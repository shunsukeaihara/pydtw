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
with the same ``dim``. The cost of a path is the sum of ``abs(a[i] - b[j])``
(1-d) or the Euclidean distance between rows (2-d) over the aligned pairs.

Alignment
---------

::

    dtw1d(a, b, window=None, ub=None, return_cost_matrix=True)
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

    dtw1d_dist(a, b, window=None, ub=None, method="pruned")
    dtw2d_dist(a, b, window=None, metric="euclidean", ub=None, method="pruned")
        -> cost

Two-row buffer (``O(len(b))`` memory instead of ``O(len(a) * len(b))``) with
pruning. Use this when the alignment is not needed, e.g. for nearest-neighbour
search or distance matrices.

Benchmark (``python bench/bench_dist.py``, 1-d, ``n = m = 5000``, best of 3,
one core; ``no prune`` is ``ub=numpy.inf``, the others use the automatic
bound). The inner loop runs at about 2.3 ns per cell independent of the data;
pruning costs nothing on hard pairs and pays off as soon as the sequences are
similar. ``pruned`` is the default because it was the fastest in every case.

=====================================  =========  =========  =========
case                                    no prune     pruned        eap
=====================================  =========  =========  =========
uniform noise                            0.058 s    0.059 s    0.096 s
random walk                              0.058 s    0.054 s    0.102 s
similar (shifted sine + noise)           0.062 s    0.023 s    0.036 s
near-identical (a, a + 0.01 noise)       0.062 s    0.003 s    0.001 s
random walk, window = 10%                0.012 s    0.012 s    0.019 s
2-d dim = 8 random walk                  0.133 s    0.138 s    0.146 s
1-NN search, 101 candidates, n = 1000    0.250 s    0.094 s    0.154 s
=====================================  =========  =========  =========

In the 1-NN row the best distance so far is passed as ``ub``, so candidates
that cannot win are abandoned early.

Alignment, same machine (``no matrix`` is ``return_cost_matrix=False``):

====================================  =========  ==========  =========  ==========
case                                      dtw1d   no matrix     pruned   pr+no mat
====================================  =========  ==========  =========  ==========
uniform noise, n = 5000                 0.096 s     0.104 s    0.142 s     0.146 s
random walk, n = 5000                   0.098 s     0.105 s    0.144 s     0.135 s
similar (shifted sine + noise)          0.094 s     0.101 s    0.074 s     0.033 s
near-identical (a, a + 0.01 noise)      0.101 s     0.099 s    0.052 s     0.007 s
random walk, n = 20000                  11.9 s      2.2 s
====================================  =========  ==========  =========  ==========

At ``n = m = 20000`` the float64 cost matrix is 3.0 GiB and the direction
matrix 0.38 GiB.

Pruned alignment
----------------

::

    pruned_dtw1d(a, b, window=None, method="pruned", return_cost_matrix=True)
    pruned_dtw2d(a, b, window=None, metric="euclidean", method="pruned", return_cost_matrix=True)
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
    Pruning scheme for the two-row implementation: ``"pruned"`` (PrunedDTW,
    Silva & Batista 2016, default) or ``"eap"`` (EAPrunedDTW, Herrmann & Webb
    2021). Both return the exact cost.

``return_cost_matrix``
    ``False`` skips the cost matrix (``None`` is returned in its place) and
    derives the alignment from a direction matrix of one byte per cell.

``metric``
    Distance between rows for 2-d sequences. Only ``"euclidean"`` for now.

Cells outside the window or pruned are ``inf`` in ``cost_matrix``. Cells above
``ub`` are not guaranteed to hold their exact value, only a value above ``ub``.
``cost_matrix`` is ``None`` when ``return_cost_matrix=False``.

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
