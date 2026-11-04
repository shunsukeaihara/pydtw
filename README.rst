pydtw
=====

Fast Implementation of Dynamic Time Warping for Python.

install
=======

::

    pip install pydtw

interface
=========

All functions accept anything ``numpy.asarray`` understands: numpy arrays,
pandas ``Series`` / ``DataFrame``, lists. C-contiguous ``float64`` input is
used in place without a copy, including read-only views such as pandas
Series under copy-on-write; other dtypes and layouts are converted to a
C-contiguous ``float64`` copy. Inputs are never modified. Input containing
``NaN`` raises ``ValueError``. 1-d functions take sequences of shape
``(len,)``, 2-d functions take multivariate sequences of shape ``(len, dim)``
with the same ``dim``. The cost of a path is the sum of the local distance
(``metric``) over the aligned pairs.

Alignment
---------

::

    dtw1d(a, b, window=None, ub=None, return_cost_matrix=True, metric="abs")
    dtw2d(a, b, window=None, metric="euclidean", ub=None, return_cost_matrix=True)
        -> (cost_matrix, cost, alignment_a, alignment_b)

Full cost matrix plus traceback. ``cost_matrix`` is an ``(len(a), len(b))``
``ndarray``; ``alignment_a[k]`` is aligned with ``alignment_b[k]``. With
``return_cost_matrix=False`` the matrix is not built: the alignment comes from
a one-byte-per-cell direction matrix, ``cost_matrix`` is ``None``, memory
drops to an eighth and long sequences get several times faster (see below).

Distance only
-------------

::

    dtw1d_dist(a, b, window=None, ub=None, method="auto", metric="abs", dtype=numpy.float64)
    dtw2d_dist(a, b, window=None, metric="euclidean", ub=None, method="auto", dtype=numpy.float64)
        -> cost

``O(len(a) + len(b))`` memory, pruning, and a SIMD kernel when the CPU has
AVX2 or NEON. Use this when the alignment is not needed, e.g. for
nearest-neighbour search or distance matrices.

Batches
-------

::

    dtw1d_dist_batch(X, Y, window=None, ub=None, metric="abs", dtype=numpy.float64)
    dtw2d_dist_batch(X, Y, window=None, metric="euclidean", ub=None, dtype=numpy.float64)
        -> costs, shape (N,)
    dtw1d_dist_matrix(X, Y=None, window=None, metric="abs", dtype=numpy.float64)
    dtw2d_dist_matrix(X, Y=None, window=None, metric="euclidean", dtype=numpy.float64)
        -> costs, shape (N, M)

``*_batch`` computes the N pairs ``(X[k], Y[k])`` with one pair per SIMD lane
(``X`` is ``(N, n)`` or ``(N, n, dim)``, ``Y`` is ``(N, m)`` or
``(N, m, dim)``; a batch of size 1 is broadcast). ``ub`` may be a scalar or
one bound per pair. ``*_matrix`` computes all pairs between the rows of ``X``
and ``Y``; with ``Y=None`` the symmetric matrix of ``X`` with itself. Series
inside a batch must share their length; group ragged data by length. Batches
pay off for many short series; for a few long series the per-pair kernel with
a good ``ub`` is faster.

::

    simd_info() -> {"simd": "avx2" | "neon" | "scalar", "lanes_float64": ..., "lanes_float32": ...}

Benchmark (``python bench/bench_dist.py``, 1-d, ``n = m = 5000``, best of 3,
one core with AVX2). ``eap`` / ``pruned`` are the row kernels, ``simd`` the
anti-diagonal kernel (default when available), ``no prune`` is ``simd`` with
``ub=numpy.inf``.

=====================================  =========  =========  =========  =========  =========
case                                         eap     pruned   simd f64   simd f32   no prune
=====================================  =========  =========  =========  =========  =========
uniform noise                            0.059 s    0.057 s    0.013 s    0.006 s    0.012 s
random walk                              0.057 s    0.059 s    0.013 s    0.006 s    0.012 s
similar (shifted sine + noise)           0.021 s    0.022 s    0.004 s    0.002 s    0.012 s
near-identical (a, a + 0.01 noise)       0.001 s    0.003 s    0.000 s    0.000 s    0.013 s
random walk, window = 10%                0.011 s    0.011 s    0.001 s    0.001 s    0.001 s
2-d dim = 8 random walk                  0.094 s    0.096 s    0.046 s    0.033 s    0.046 s
1-NN search, 101 candidates, n = 1000    0.088 s    0.090 s    0.013 s
=====================================  =========  =========  =========  =========  =========

In the 1-NN row the best distance so far is passed as ``ub``, so candidates
that cannot win are abandoned early. The same search as one
``dtw1d_dist_batch`` call takes 0.092 s (float64) / 0.046 s (float32): a
batch cannot exploit a tightening bound, so it wins only on short series.
All pairs of 200 series of length 100: 0.997 s as a loop over
``dtw1d_dist(method="eap")``, 0.159 s with ``dtw1d_dist_matrix`` (0.086 s in
float32).

Alignment, same machine (``no matrix`` is ``return_cost_matrix=False``;
``pruned_dtw1d`` now gets its bound from the SIMD kernel):

====================================  =========  ==========  =========  ==========
case                                      dtw1d   no matrix     pruned   pr+no mat
====================================  =========  ==========  =========  ==========
uniform noise, n = 5000                 0.097 s     0.061 s    0.105 s     0.062 s
random walk, n = 5000                   0.093 s     0.056 s    0.093 s     0.058 s
similar (shifted sine + noise)          0.096 s     0.058 s    0.064 s     0.013 s
near-identical (a, a + 0.01 noise)      0.094 s     0.063 s    0.052 s     0.003 s
random walk, n = 20000                  13.8 s      2.5 s
====================================  =========  ==========  =========  ==========

At ``n = m = 20000`` the float64 cost matrix is 3.0 GiB and the direction
matrix 0.38 GiB.

Pruned alignment
----------------

::

    pruned_dtw1d(a, b, window=None, method="auto", return_cost_matrix=True, metric="abs")
    pruned_dtw2d(a, b, window=None, metric="euclidean", method="auto", return_cost_matrix=True)
        -> (cost_matrix, cost, alignment_a, alignment_b)

Same result as ``dtw1d`` / ``dtw2d``. The exact cost is computed first with
``dtw*_dist`` and then passed as ``ub`` to the alignment, so the full matrix
only expands cells that can lie on an optimal path.

Arguments
---------

``window``
    Sakoe-Chiba window. ``None`` searches the whole matrix; an integer ``w``
    only evaluates cells with ``abs(i - j) <= w``. A window smaller than
    ``abs(len(a) - len(b))`` cannot reach the last cell and is widened to that
    value with a warning.

``ub``
    Upper bound on the cost. Cells whose partial cost exceeds it are pruned,
    which is exact as long as the true cost is at most ``ub``. When the true
    cost exceeds ``ub``, ``dtw*_dist`` returns ``inf`` and ``dtw1d`` /
    ``dtw2d`` return ``(cost_matrix, inf, [], [])``. In ``dtw*_dist``,
    ``None`` (default) derives a bound from a simple warping path, which keeps
    the result exact; pass ``numpy.inf`` to disable pruning. In ``dtw1d`` /
    ``dtw2d``, ``None`` disables pruning. Typical sources of a bound are the
    best distance found so far in a nearest-neighbour search, or the cost
    with a narrower ``window``.

``method``
    Kernel for the distance-only functions. ``"auto"`` (default) picks the
    SIMD anti-diagonal kernel when the CPU supports AVX2 (x86-64) or NEON
    (AArch64) and EAPrunedDTW otherwise. ``"simd"`` forces the anti-diagonal
    kernel (with a scalar fallback), ``"eap"`` (EAPrunedDTW, Herrmann & Webb
    2021) and ``"pruned"`` (PrunedDTW, Silva & Batista 2016) are the
    row-oriented kernels. All return the exact cost.

``dtype``
    ``numpy.float64`` (default) or ``numpy.float32`` for the SIMD kernels.
    float32 doubles the lanes and halves the memory traffic; the cost is then
    accurate to about 1e-6 relative (more for very long paths), and the
    result is still returned as a Python float / float32 array.

``return_cost_matrix``
    ``False`` skips the cost matrix (``None`` is returned in its place) and
    derives the alignment from a direction matrix of one byte per cell.

``metric``
    Local distance between aligned elements. 1-d: ``"abs"`` (default,
    ``|a[i] - b[j]|``; ``"cityblock"`` and ``"euclidean"`` are accepted as
    aliases) or ``"sqeuclidean"`` (``(a[i] - b[j]) ** 2``). 2-d, between
    rows: ``"euclidean"`` (default), ``"sqeuclidean"``, ``"cityblock"``,
    ``"chebyshev"``. Each metric is compiled into its own specialisation of
    the inner loop, so none of them pays for the choice at run time.

Cells outside the window or pruned are ``inf`` in ``cost_matrix``. Cells above
``ub`` are not guaranteed to hold their exact value, only a value above ``ub``.
``cost_matrix`` is ``None`` when ``return_cost_matrix=False``.

SIMD
----

The distance kernels are compiled three times: scalar, AVX2 + FMA and NEON.
At first use the CPU is probed (CPUID on x86-64; NEON is always present on
AArch64) and the best available version is selected; the wheels therefore
run on any x86-64 or AArch64 machine. ``PYDTW_SIMD=scalar|avx2|neon|auto``
in the environment overrides the choice (an unavailable choice falls back to
scalar), which the test suite uses to check the scalar path. All kernels give
the same results up to floating-point rounding.

Deprecated
----------

``constrained_dtw1d(a, b, constraint)`` and
``constrained_dtw2d(a, b, constraint, metric)`` are aliases for
``dtw1d(a, b, window=constraint)`` and ``dtw2d(a, b, window=constraint,
metric=metric)``.

usage
=====

Alignment for two 1-dimensional sequences
-----------------------------------------

::

    from pydtw import dtw1d
    import numpy as np
    a = np.random.rand(10)
    b = np.random.rand(15)
    cost_matrix, cost, alignment_a, alignment_b = dtw1d(a, b)

Alignment for two 2-dimensional sequences
-----------------------------------------

::

    from pydtw import dtw2d
    import numpy as np

    a = np.random.rand(10, 4)
    b = np.random.rand(15, 4)
    cost_matrix, cost, alignment_a, alignment_b = dtw2d(a, b)

Nearest neighbour with pruning
------------------------------

::

    from pydtw import dtw1d_dist, pruned_dtw1d

    best, best_i = float("inf"), None
    for i, c in enumerate(candidates):
        d = dtw1d_dist(query, c, window=50, ub=best)   # inf once it exceeds best
        if d < best:
            best, best_i = d, i
    _, cost, al_q, al_c = pruned_dtw1d(query, candidates[best_i], window=50,
                                      return_cost_matrix=False)

development
===========

::

    pip install -e .
    pytest
