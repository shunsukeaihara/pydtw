# -*- coding: utf-8 -*-
# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# distutils: language = c++
import warnings

import numpy as np
cimport cython
from libc.math cimport fabs, sqrt, INFINITY
from libc.float cimport DBL_EPSILON
from libcpp.vector cimport vector
from libc.math cimport NAN


cdef extern from "simd/dtw_simd.hpp":
    int pydtw_simd_level() noexcept nogil
    const char* pydtw_simd_name() noexcept nogil
    int pydtw_simd_lanes(int itemsize) noexcept nogil
    double pydtw_dtw_ad_f64(const double* a, size_t n, const double* b, size_t m, size_t dim,
                            int metric, size_t w, double ub, int pruning) noexcept nogil
    float pydtw_dtw_ad_f32(const float* a, size_t n, const float* b, size_t m, size_t dim,
                           int metric, size_t w, float ub, int pruning) noexcept nogil
    void pydtw_dtw_batch_f64(const double* X, const double* Y, size_t N, size_t n, size_t m,
                             size_t dim, int metric, size_t w, const double* ub, int pruning,
                             double* out) noexcept nogil
    void pydtw_dtw_batch_f32(const float* X, const float* Y, size_t N, size_t n, size_t m,
                             size_t dim, int metric, size_t w, const float* ub, int pruning,
                             float* out) noexcept nogil


# ---------------------------------------------------------------------------
# Local distances
#
# Each metric is a distinct (empty) struct type; the cores are fused over the
# tag type, so Cython emits one specialised C function per metric and the
# distance is inlined with no per-cell dispatch on any compiler.
# ---------------------------------------------------------------------------

ctypedef struct AbsMetric:          # 1-d: |a - b|
    char _
ctypedef struct SqDiffMetric:       # 1-d: (a - b) ** 2
    char _
ctypedef struct EuclideanMetric:    # n-d: sqrt(sum (a - b) ** 2)
    char _
ctypedef struct SqEuclideanMetric:  # n-d: sum (a - b) ** 2
    char _
ctypedef struct CityblockMetric:    # n-d: sum |a - b|
    char _
ctypedef struct ChebyshevMetric:    # n-d: max |a - b|
    char _

ctypedef fused metric_t:
    AbsMetric
    SqDiffMetric
    EuclideanMetric
    SqEuclideanMetric
    CityblockMetric
    ChebyshevMetric

cdef enum:
    M_ABS = 0
    M_SQDIFF = 1
    M_EUCLIDEAN = 2
    M_SQEUCLIDEAN = 3
    M_CITYBLOCK = 4
    M_CHEBYSHEV = 5


cdef inline double cell_dist(metric_t* tag, const double* a, const double* b,
                             Py_ssize_t dim) noexcept nogil:
    cdef Py_ssize_t i
    cdef double tmp, d = 0.0
    if metric_t is AbsMetric:
        return fabs(a[0] - b[0])
    elif metric_t is SqDiffMetric:
        tmp = a[0] - b[0]
        return tmp * tmp
    elif metric_t is EuclideanMetric:
        for i in range(dim):
            tmp = a[i] - b[i]
            d += tmp * tmp
        return sqrt(d)
    elif metric_t is SqEuclideanMetric:
        for i in range(dim):
            tmp = a[i] - b[i]
            d += tmp * tmp
        return d
    elif metric_t is CityblockMetric:
        for i in range(dim):
            d += fabs(a[i] - b[i])
        return d
    else:
        for i in range(dim):
            tmp = fabs(a[i] - b[i])
            if tmp > d:
                d = tmp
        return d


# ---------------------------------------------------------------------------
# Argument handling
# ---------------------------------------------------------------------------

def check_constraint(a_shape, b_shape, constraint=0, warn=True):
    """Return a Sakoe-Chiba window that is large enough for the given lengths.

    A window ``w`` only allows cells with ``abs(i - j) <= w``; the last cell
    ``(a_shape - 1, b_shape - 1)`` is therefore reachable only if
    ``w >= abs(a_shape - b_shape)``.
    """
    min_size = abs(a_shape - b_shape)
    if constraint < min_size:
        if warn:
            warnings.warn(
                "Constraint {} too small for sequences length {} and {}; using {}".format(
                    constraint, a_shape, b_shape, min_size))
        constraint = min_size
    return constraint


cdef Py_ssize_t _resolve_window(Py_ssize_t n, Py_ssize_t m, window) except -1:
    if window is None:
        return max(n, m)
    window = int(window)
    if window < 0:
        raise ValueError("window must be non-negative, got {}".format(window))
    return check_constraint(n, m, window)


cdef _as_sequence(x, int ndim, str name):
    # No copy for C-contiguous float64 input (including read-only views such
    # as pandas Series under copy-on-write); everything else is converted.
    arr = np.ascontiguousarray(x, dtype=np.float64)
    if arr.ndim != ndim:
        raise ValueError("{} must be {}-dimensional, got shape {}".format(name, ndim, arr.shape))
    if arr.shape[0] == 0:
        raise ValueError("{} must not be empty".format(name))
    if np.isnan(arr).any():
        raise ValueError("{} contains NaN".format(name))
    return arr


cdef _as_pair_2d(a, b):
    a = _as_sequence(a, 2, "a")
    b = _as_sequence(b, 2, "b")
    if a.shape[1] != b.shape[1]:
        raise ValueError('Matrices must have same dimension. a={}, b={}'.format(a.shape[1], b.shape[1]))
    return a, b


_METRICS_1D = {"abs": M_ABS, "cityblock": M_ABS, "euclidean": M_ABS, "sqeuclidean": M_SQDIFF}
_METRICS_2D = {"euclidean": M_EUCLIDEAN, "sqeuclidean": M_SQEUCLIDEAN,
               "cityblock": M_CITYBLOCK, "chebyshev": M_CHEBYSHEV}


cdef int _metric_id(metric, int ndim) except -1:
    table = _METRICS_1D if ndim == 1 else _METRICS_2D
    try:
        return table[metric]
    except (KeyError, TypeError):
        raise ValueError("unrecognized metric {!r}; choose from {}".format(
            metric, ", ".join(sorted(table))))


cdef enum:
    # Pruning schemes; values match PYDTW_PRUNE_* in simd/dtw_simd.hpp.
    METHOD_NONE = 0
    METHOD_PRUNED = 1
    METHOD_EAP = 2


cdef int _method_id(method) except -1:
    if method == 'eap':
        return METHOD_EAP
    if method == 'pruned':
        return METHOD_PRUNED
    if method == 'none':
        return METHOD_NONE
    raise ValueError("method must be 'eap', 'pruned' or 'none', got {!r}".format(method))


cdef bint _is_float32(dtype) except -1:
    dt = np.dtype(dtype)
    if dt == np.float64:
        return False
    if dt == np.float32:
        return True
    raise ValueError("dtype must be float64 or float32, got {!r}".format(dtype))


cdef inline double _slack(double cost) noexcept:
    """A bound for the alignment pass of pruned_dtw*: the cost from the
    distance kernel plus a relative 1e-9, because the SIMD kernel and the
    row kernel may round differently (fused multiply-add, vector sums) and
    an exact bound could prune the optimum. The alignment pass returns its
    own exact cost, so the slack never changes the result."""
    return cost + 1e-9 * cost if cost != INFINITY else cost


cdef double _bound(ub) except? -1.0:
    cdef double bound = INFINITY if ub is None else float(ub)
    if bound != bound:
        raise ValueError("ub must not be NaN")
    return bound


# ---------------------------------------------------------------------------
# Core 1: row-oriented DP with PrunedDTW pruning (Silva & Batista, 2016)
#
# One routine serves three uses, selected by how the rows are stored:
#   * full cost matrix        : nrows = n + 1, dirs = NULL
#   * cost only               : nrows = 2 (two row buffers), dirs = NULL
#   * alignment without matrix: nrows = 2 plus an (n, m) uint8 direction matrix
# The branch on dirs is loop-invariant and hoisted by the compiler, so the
# variants do not pay for each other.
# ---------------------------------------------------------------------------

cdef double pruned_rows_core(metric_t* tag, const double* a, Py_ssize_t n, const double* b,
                             Py_ssize_t m, Py_ssize_t dim, Py_ssize_t w, double ub,
                             double* D, Py_ssize_t ld, Py_ssize_t nrows,
                             unsigned char* dirs) noexcept nogil:
    """Row i (1-based, 0 = virtual row above the first element) lives at
    ``D + (i % nrows) * ld`` and has m + 1 entries; column 0 is a virtual
    column left of the first element. Cells with cost above ``ub`` are pruned:
    ``sc`` is the first column of the next row that can still be reached,
    ``ec`` is one past the last unpruned column of the previous row, and a row
    is abandoned as soon as a pruned cell is at or beyond ``ec``. Pruned and
    never-visited cells read as INFINITY. Cells outside the Sakoe-Chiba window
    ``w`` are never visited. ``dirs[(i-1)*m + (j-1)]`` receives 0 (diagonal),
    1 (from i-1) or 2 (from j-1), ties resolved in that order.
    Returns the final cost, or INFINITY when it exceeds ``ub``.
    """
    cdef Py_ssize_t i, j, beg, end, fill_to, sc = 1, ec = 1, ec_next = 1
    cdef double* prev = D
    cdef double* cur
    cdef unsigned char* drow = NULL
    cdef const double* ai
    cdef bint smaller_found, c, c2
    cdef double v, left, mn, up, diag
    cdef unsigned char d
    for j in range(m + 1):
        prev[j] = INFINITY
    prev[0] = 0.0
    for i in range(1, n + 1):
        beg = max(sc, i - w)
        end = min(m, i + w)
        if beg > end:
            return INFINITY
        cur = D + (i % nrows) * ld
        ai = a + (i - 1) * dim
        if dirs != NULL:
            drow = dirs + (i - 1) * m
        cur[beg - 1] = INFINITY
        left = INFINITY
        diag = prev[beg - 1]
        smaller_found = False
        j = beg
        while j <= end:
            up = prev[j]
            # Branch-free selection (compiles to min/cmov): a data-dependent
            # branch here mispredicts on noisy series.
            c = up < diag
            mn = up if c else diag
            c2 = left < mn
            mn = left if c2 else mn
            # d = 2 if c2 else (1 if c else 0), without a branch
            d = <unsigned char>((c2 << 1) | (c & (not c2)))
            diag = up
            v = cell_dist(tag, ai, b + (j - 1) * dim, dim) + mn
            cur[j] = v
            left = v
            if dirs != NULL:
                drow[j - 1] = d
            if v > ub:
                if not smaller_found:
                    sc = j + 1
                if j >= ec:
                    break
            else:
                smaller_found = True
                ec_next = j + 1
            j += 1
        if not smaller_found:
            return INFINITY
        # Cells after the abandoned one, plus one guard cell right of the
        # window, must read as INFINITY from the next row.
        if j <= end:
            j += 1
        fill_to = min(m, end + 1)
        while j <= fill_to:
            cur[j] = INFINITY
            j += 1
        ec = ec_next
        prev = cur
    v = prev[m]
    return v if v <= ub else INFINITY


# ---------------------------------------------------------------------------
# Core 2: EAPrunedDTW (Herrmann & Webb, 2021), cost only, two row buffers
# ---------------------------------------------------------------------------

cdef double path_upper_bound(metric_t* tag, const double* a, Py_ssize_t n, const double* b,
                             Py_ssize_t m, Py_ssize_t dim) noexcept nogil:
    """Cost of one feasible warping path: the diagonal, then along the last
    row or column. Any feasible path cost is an upper bound of the DTW cost,
    and this path stays inside every window wide enough for the last cell.

    The partial sums are accumulated in the same order the DP accumulates them
    along this path, so the DP reproduces the bound bit for bit and the optimum
    is never pruned by rounding.
    """
    cdef Py_ssize_t i, k = min(n, m)
    cdef double s = 0.0
    for i in range(k):
        s += cell_dist(tag, a + i * dim, b + i * dim, dim)
    for i in range(k, n):
        s += cell_dist(tag, a + i * dim, b + (m - 1) * dim, dim)
    for i in range(k, m):
        s += cell_dist(tag, a + (n - 1) * dim, b + i * dim, dim)
    return s


cdef double eap_pruned_dtw_core(metric_t* tag, const double* a, Py_ssize_t n, const double* b,
                                Py_ssize_t m, Py_ssize_t dim, Py_ssize_t w, double cutoff,
                                double* prev, double* cur) noexcept nogil:
    """``prev`` and ``cur`` point one past the start of buffers of length m + 1
    whose first slot holds INFINITY, so ``prev[-1]`` is a valid read.

    Every path ends in the last cell, so a partial cost above
    ``ub = cutoff - d(a[n-1], b[m-1])`` can never finish at or below
    ``cutoff``; such cells are pruned. Each row is processed in three phases:
    columns below ``prev_pp`` (one past the last unpruned column of the
    previous row) use top, diagonal and left; the column at ``prev_pp`` has no
    valid top; columns beyond it can only come from the left, so the row is
    abandoned at the first pruned cell there. Leading pruned cells advance
    ``next_start`` so later rows skip them. Rows without any unpruned cell
    abandon the whole computation. Returns INFINITY when the cost exceeds
    ``cutoff``.
    """
    cdef double d_last = cell_dist(tag, a + (n - 1) * dim, b + (m - 1) * dim, dim)
    # Slack of two ulps of the cutoff makes the comparison robust to rounding:
    # a path whose final cost rounds to <= cutoff is never pruned early.
    cdef double ub = cutoff - d_last + 2.0 * DBL_EPSILON * cutoff
    cdef Py_ssize_t i, j, jbeg, jend, pp_stop, last
    cdef Py_ssize_t next_start = 0, prev_pp = 0, curr_pp = 0
    cdef bint found
    cdef double cost, left, mn, up, diag
    cdef const double* ai
    cdef double* tmp

    # Row 0: every cell is reached from the left only.
    jend = min(m - 1, w)
    cost = 0.0
    j = 0
    while j <= jend:
        cost += cell_dist(tag, a, b + j * dim, dim)
        cur[j] = cost
        if cost > ub:
            break
        curr_pp = j + 1
        j += 1
    last = j if j <= jend else jend

    for i in range(1, n):
        if curr_pp == 0:
            return INFINITY
        prev_pp = curr_pp
        tmp = prev
        prev = cur
        cur = tmp
        ai = a + i * dim
        jbeg = max(next_start, i - w)
        jend = min(m - 1, i + w)
        if jbeg > prev_pp or jbeg > jend:
            return INFINITY
        cur[jbeg - 1] = INFINITY
        curr_pp = 0
        found = False
        left = INFINITY
        j = jbeg
        # Phase 1: top, diagonal and left are all available.
        pp_stop = min(jend, prev_pp - 1)
        diag = prev[j - 1]
        while j <= pp_stop:
            up = prev[j]
            mn = diag
            diag = up
            if up < mn:
                mn = up
            if left < mn:
                mn = left
            cost = cell_dist(tag, ai, b + j * dim, dim) + mn
            cur[j] = cost
            left = cost
            if cost <= ub:
                found = True
                curr_pp = j + 1
            elif not found:
                next_start = j + 1
            j += 1
        # Phase 2: the column at prev_pp has no valid top cell.
        if j == prev_pp and j <= jend:
            mn = prev[j - 1]
            if left < mn:
                mn = left
            cost = cell_dist(tag, ai, b + j * dim, dim) + mn
            cur[j] = cost
            left = cost
            if cost <= ub:
                found = True
                curr_pp = j + 1
            elif not found:
                next_start = j + 1
            j += 1
        # Phase 3: only the left cell is available; stop at the first pruned one.
        while j <= jend:
            cost = cell_dist(tag, ai, b + j * dim, dim) + left
            cur[j] = cost
            left = cost
            if cost > ub:
                break
            found = True
            curr_pp = j + 1
            j += 1
        last = j if j <= jend else jend

    if last == m - 1 and cur[m - 1] <= cutoff:
        return cur[m - 1]
    return INFINITY


# ---------------------------------------------------------------------------
# Traceback
# ---------------------------------------------------------------------------

cdef _finish_path(vector[Py_ssize_t]& a, vector[Py_ssize_t]& b):
    align_a = list(a)
    align_b = list(b)
    align_a.reverse()
    align_b.reverse()
    return align_a, align_b


cdef traceback_matrix(const double[:, ::1] cost_mat):
    """Walk back from the last cell of the padded (n+1, m+1) cost matrix."""
    cdef Py_ssize_t i = cost_mat.shape[0] - 1
    cdef Py_ssize_t j = cost_mat.shape[1] - 1
    cdef vector[Py_ssize_t] a
    cdef vector[Py_ssize_t] b
    cdef double diag, up, left
    a.push_back(i - 1)
    b.push_back(j - 1)
    while i > 1 or j > 1:
        if i == 1:
            j -= 1
        elif j == 1:
            i -= 1
        else:
            diag = cost_mat[i - 1, j - 1]
            up = cost_mat[i - 1, j]
            left = cost_mat[i, j - 1]
            if diag <= up and diag <= left:
                i -= 1
                j -= 1
            elif up <= left:
                i -= 1
            else:
                j -= 1
        a.push_back(i - 1)
        b.push_back(j - 1)
    return _finish_path(a, b)


cdef traceback_dirs(const unsigned char[:, ::1] dirs):
    """Walk back along the recorded predecessor directions of an (n, m) matrix."""
    cdef Py_ssize_t i = dirs.shape[0] - 1
    cdef Py_ssize_t j = dirs.shape[1] - 1
    cdef vector[Py_ssize_t] a
    cdef vector[Py_ssize_t] b
    cdef unsigned char d
    a.push_back(i)
    b.push_back(j)
    while i > 0 or j > 0:
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            d = dirs[i, j]
            if d == 0:
                i -= 1
                j -= 1
            elif d == 1:
                i -= 1
            else:
                j -= 1
        a.push_back(i)
        b.push_back(j)
    return _finish_path(a, b)


# ---------------------------------------------------------------------------
# Drivers: dispatch the metric id to the specialised core
# ---------------------------------------------------------------------------

cdef double run_rows(int mid, const double* a, Py_ssize_t n, const double* b, Py_ssize_t m,
                     Py_ssize_t dim, Py_ssize_t w, double ub, double* D, Py_ssize_t ld,
                     Py_ssize_t nrows, unsigned char* dirs) noexcept nogil:
    if mid == M_ABS:
        return pruned_rows_core[AbsMetric](NULL, a, n, b, m, dim, w, ub, D, ld, nrows, dirs)
    elif mid == M_SQDIFF:
        return pruned_rows_core[SqDiffMetric](NULL, a, n, b, m, dim, w, ub, D, ld, nrows, dirs)
    elif mid == M_EUCLIDEAN:
        return pruned_rows_core[EuclideanMetric](NULL, a, n, b, m, dim, w, ub, D, ld, nrows, dirs)
    elif mid == M_SQEUCLIDEAN:
        return pruned_rows_core[SqEuclideanMetric](NULL, a, n, b, m, dim, w, ub, D, ld, nrows, dirs)
    elif mid == M_CITYBLOCK:
        return pruned_rows_core[CityblockMetric](NULL, a, n, b, m, dim, w, ub, D, ld, nrows, dirs)
    else:
        return pruned_rows_core[ChebyshevMetric](NULL, a, n, b, m, dim, w, ub, D, ld, nrows, dirs)


cdef double run_eap(int mid, const double* a, Py_ssize_t n, const double* b, Py_ssize_t m,
                    Py_ssize_t dim, Py_ssize_t w, double cutoff, double* prev,
                    double* cur) noexcept nogil:
    if mid == M_ABS:
        return eap_pruned_dtw_core[AbsMetric](NULL, a, n, b, m, dim, w, cutoff, prev, cur)
    elif mid == M_SQDIFF:
        return eap_pruned_dtw_core[SqDiffMetric](NULL, a, n, b, m, dim, w, cutoff, prev, cur)
    elif mid == M_EUCLIDEAN:
        return eap_pruned_dtw_core[EuclideanMetric](NULL, a, n, b, m, dim, w, cutoff, prev, cur)
    elif mid == M_SQEUCLIDEAN:
        return eap_pruned_dtw_core[SqEuclideanMetric](NULL, a, n, b, m, dim, w, cutoff, prev, cur)
    elif mid == M_CITYBLOCK:
        return eap_pruned_dtw_core[CityblockMetric](NULL, a, n, b, m, dim, w, cutoff, prev, cur)
    else:
        return eap_pruned_dtw_core[ChebyshevMetric](NULL, a, n, b, m, dim, w, cutoff, prev, cur)


cdef double run_upper_bound(int mid, const double* a, Py_ssize_t n, const double* b,
                            Py_ssize_t m, Py_ssize_t dim) noexcept nogil:
    if mid == M_ABS:
        return path_upper_bound[AbsMetric](NULL, a, n, b, m, dim)
    elif mid == M_SQDIFF:
        return path_upper_bound[SqDiffMetric](NULL, a, n, b, m, dim)
    elif mid == M_EUCLIDEAN:
        return path_upper_bound[EuclideanMetric](NULL, a, n, b, m, dim)
    elif mid == M_SQEUCLIDEAN:
        return path_upper_bound[SqEuclideanMetric](NULL, a, n, b, m, dim)
    elif mid == M_CITYBLOCK:
        return path_upper_bound[CityblockMetric](NULL, a, n, b, m, dim)
    else:
        return path_upper_bound[ChebyshevMetric](NULL, a, n, b, m, dim)


cdef _dtw_path(const double[:, ::1] a, const double[:, ::1] b, Py_ssize_t w, double ub,
               int mid, bint return_cost_matrix):
    cdef Py_ssize_t n = a.shape[0], m = b.shape[0], dim = a.shape[1]
    cdef double[:, ::1] cost_mat
    cdef unsigned char[:, ::1] dirs
    cdef vector[double] rows
    cdef double cost
    if return_cost_matrix:
        full = np.full((n + 1, m + 1), np.inf, dtype=np.float64)
        cost_mat = full
        with nogil:
            cost = run_rows(mid, &a[0, 0], n, &b[0, 0], m, dim, w, ub,
                            &cost_mat[0, 0], m + 1, n + 1, NULL)
        if cost == INFINITY:
            return full[1:, 1:], cost, [], []
        align_a, align_b = traceback_matrix(cost_mat)
        return full[1:, 1:], cost, align_a, align_b
    dirs_np = np.empty((n, m), dtype=np.uint8)
    dirs = dirs_np
    rows = vector[double](2 * (m + 1))
    with nogil:
        cost = run_rows(mid, &a[0, 0], n, &b[0, 0], m, dim, w, ub,
                        rows.data(), m + 1, 2, &dirs[0, 0])
    if cost == INFINITY:
        return None, cost, [], []
    align_a, align_b = traceback_dirs(dirs)
    return None, cost, align_a, align_b


cdef double _dtw_dist_simd(a, b, Py_ssize_t w, ub, int mid, int pruning, bint float32):
    """Anti-diagonal SIMD kernel (C++). a, b are C-contiguous 2-d arrays."""
    cdef const double[:, ::1] a64
    cdef const double[:, ::1] b64
    cdef const float[:, ::1] a32
    cdef const float[:, ::1] b32
    cdef double bound = NAN if ub is None else _bound(ub)
    cdef double result
    cdef Py_ssize_t n = a.shape[0], m = b.shape[0], dim = a.shape[1]
    if float32:
        a32 = np.ascontiguousarray(a, dtype=np.float32)
        b32 = np.ascontiguousarray(b, dtype=np.float32)
        with nogil:
            result = pydtw_dtw_ad_f32(&a32[0, 0], n, &b32[0, 0], m, dim, mid, w, <float>bound, pruning)
    else:
        a64 = a
        b64 = b
        with nogil:
            result = pydtw_dtw_ad_f64(&a64[0, 0], n, &b64[0, 0], m, dim, mid, w, bound, pruning)
    return result


cdef double _dtw_dist(a, b, Py_ssize_t w, ub, int mid, method, dtype, bint simd):
    """Dispatch to the anti-diagonal SIMD kernel (when the CPU has one, or for
    float32) or to the scalar row kernels. ``simd=False`` forces the row
    kernels; it exists for benchmarking."""
    cdef const double[:, ::1] av
    cdef const double[:, ::1] bv
    cdef Py_ssize_t n = a.shape[0], m = b.shape[0], dim = a.shape[1]
    cdef vector[double] buf0
    cdef vector[double] buf1
    cdef vector[double] rows
    cdef double bound, result
    cdef bint auto_ub = ub is None
    cdef int pruning = _method_id(method)
    cdef bint float32 = _is_float32(dtype)
    if float32 and not simd:
        raise ValueError("dtype=float32 is only available with simd=True")
    if simd and (float32 or pydtw_simd_level() > 0):
        return _dtw_dist_simd(a, b, w, ub, mid, pruning, float32)
    av = a
    bv = b
    bound = _bound(ub)
    # One leading INFINITY slot so that the EAP core may read index -1.
    buf0 = vector[double](m + 2, INFINITY)
    buf1 = vector[double](m + 2, INFINITY)
    if pruning != METHOD_EAP:
        rows = vector[double](2 * (m + 1))
    with nogil:
        if auto_ub and pruning != METHOD_NONE:
            bound = run_upper_bound(mid, &av[0, 0], n, &bv[0, 0], m, dim)
        if pruning == METHOD_EAP:
            result = run_eap(mid, &av[0, 0], n, &bv[0, 0], m, dim, w, bound,
                             buf0.data() + 1, buf1.data() + 1)
        elif pruning == METHOD_PRUNED:
            result = run_rows(mid, &av[0, 0], n, &bv[0, 0], m, dim, w, bound,
                              rows.data(), m + 1, 2, NULL)
        else:
            result = run_rows(mid, &av[0, 0], n, &bv[0, 0], m, dim, w, INFINITY,
                              rows.data(), m + 1, 2, NULL)
            if result > bound:
                result = INFINITY
    return result


cdef _batch_inputs(X, Y, int ndim, str name_x, str name_y, dtype):
    """Validate (N, len[, dim]) batches, broadcast a batch of one, return
    C-contiguous arrays of the requested dtype."""
    cdef bint float32 = _is_float32(dtype)
    dt = np.float32 if float32 else np.float64
    X = np.ascontiguousarray(X, dtype=dt)
    Y = np.ascontiguousarray(Y, dtype=dt)
    if X.ndim != ndim + 1 or Y.ndim != ndim + 1:
        raise ValueError("{} and {} must be {}-dimensional batches, got shapes {} and {}".format(
            name_x, name_y, ndim + 1, X.shape, Y.shape))
    if X.shape[1] == 0 or Y.shape[1] == 0:
        raise ValueError("sequences must not be empty")
    if ndim == 2 and X.shape[2] != Y.shape[2]:
        raise ValueError("Matrices must have same dimension. {}={}, {}={}".format(
            name_x, X.shape[2], name_y, Y.shape[2]))
    if X.shape[0] != Y.shape[0]:
        if X.shape[0] == 1:
            X = np.ascontiguousarray(np.broadcast_to(X, (Y.shape[0],) + X.shape[1:]))
        elif Y.shape[0] == 1:
            Y = np.ascontiguousarray(np.broadcast_to(Y, (X.shape[0],) + Y.shape[1:]))
        else:
            raise ValueError("batch sizes differ: {} vs {}".format(X.shape[0], Y.shape[0]))
    if np.isnan(X).any():
        raise ValueError("{} contains NaN".format(name_x))
    if np.isnan(Y).any():
        raise ValueError("{} contains NaN".format(name_y))
    return X, Y, float32


cdef _batch_bounds(ub, Py_ssize_t N, dt):
    if ub is None:
        return np.full(N, np.nan, dtype=dt)
    ubs = np.asarray(ub, dtype=np.float64)
    if ubs.ndim == 0:
        ubs = np.full(N, float(ubs), dtype=np.float64)
    elif ubs.shape != (N,):
        raise ValueError("ub must be a scalar or have shape ({},), got {}".format(N, ubs.shape))
    if np.isnan(ubs).any():
        raise ValueError("ub must not be NaN")
    return np.ascontiguousarray(ubs, dtype=dt)


cdef _dtw_dist_batch(X, Y, Py_ssize_t w, ub, int mid, int pruning, bint float32):
    """X, Y: C-contiguous (N, len, dim) arrays of the same dtype."""
    cdef const double[:, :, ::1] X64
    cdef const double[:, :, ::1] Y64
    cdef const double[::1] ub64
    cdef double[::1] out64
    cdef const float[:, :, ::1] X32
    cdef const float[:, :, ::1] Y32
    cdef const float[::1] ub32
    cdef float[::1] out32
    cdef Py_ssize_t N = X.shape[0], n = X.shape[1], m = Y.shape[1], dim = X.shape[2]
    if N == 0:
        return np.empty(0, dtype=np.float32 if float32 else np.float64)
    if float32:
        out = np.empty(N, dtype=np.float32)
        X32, Y32, ub32, out32 = X, Y, _batch_bounds(ub, N, np.float32), out
        with nogil:
            pydtw_dtw_batch_f32(&X32[0, 0, 0], &Y32[0, 0, 0], N, n, m, dim, mid, w,
                                &ub32[0], pruning, &out32[0])
    else:
        out = np.empty(N, dtype=np.float64)
        X64, Y64, ub64, out64 = X, Y, _batch_bounds(ub, N, np.float64), out
        with nogil:
            pydtw_dtw_batch_f64(&X64[0, 0, 0], &Y64[0, 0, 0], N, n, m, dim, mid, w,
                                &ub64[0], pruning, &out64[0])
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def dtw1d(a, b, window=None, ub=None, return_cost_matrix=True, metric="abs"):
    """DTW alignment of two 1-d sequences.

    ``window``: Sakoe-Chiba window; ``None`` searches the full matrix, an
    integer ``w`` only cells with ``abs(i - j) <= w`` (widened with a
    warning when smaller than ``abs(len(a) - len(b))``).
    ``ub``: upper bound on the cost; cells above it are pruned. When the
    cost exceeds ``ub`` the result is ``(cost_matrix, inf, [], [])``.
    ``return_cost_matrix``: ``False`` returns ``None`` instead of the matrix
    and computes the alignment from a byte-per-cell direction matrix, which
    is faster and needs an eighth of the memory.
    ``metric``: local distance, ``"abs"`` (default, ``|a[i] - b[j]|``) or
    ``"sqeuclidean"`` (``(a[i] - b[j]) ** 2``).

    Returns ``(cost_matrix, cost, alignment_a, alignment_b)``; cells outside
    the window or pruned are ``inf`` in ``cost_matrix``.
    """
    a = _as_sequence(a, 1, "a")
    b = _as_sequence(b, 1, "b")
    w = _resolve_window(a.shape[0], b.shape[0], window)
    return _dtw_path(a.reshape(-1, 1), b.reshape(-1, 1), w, _bound(ub), _metric_id(metric, 1),
                     return_cost_matrix)


def dtw2d(a, b, window=None, metric="euclidean", ub=None, return_cost_matrix=True):
    """DTW alignment of two multivariate sequences of shape (len, dim).

    ``metric`` is the distance between rows: ``"euclidean"`` (default),
    ``"sqeuclidean"``, ``"cityblock"`` or ``"chebyshev"``. See
    :func:`dtw1d` for ``window``, ``ub`` and ``return_cost_matrix``.
    """
    a, b = _as_pair_2d(a, b)
    w = _resolve_window(a.shape[0], b.shape[0], window)
    return _dtw_path(a, b, w, _bound(ub), _metric_id(metric, 2), return_cost_matrix)


def constrained_dtw1d(a, b, constraint=0):
    """Deprecated alias for ``dtw1d(a, b, window=constraint)``."""
    return dtw1d(a, b, window=constraint)


def constrained_dtw2d(a, b, constraint=0, metric="euclidean"):
    """Deprecated alias for ``dtw2d(a, b, window=constraint, metric=metric)``."""
    return dtw2d(a, b, window=constraint, metric=metric)


def dtw1d_dist(a, b, window=None, ub=None, method="eap", metric="abs", dtype=np.float64,
               simd=True):
    """DTW cost of two 1-d sequences without computing the alignment.

    Uses O(len(a) + len(b)) memory. ``window`` and ``metric`` as in
    :func:`dtw1d`. ``ub`` is an upper bound on the cost: cells above it are
    pruned and ``inf`` is returned when the true cost exceeds it. ``None``
    (default) derives a bound from a simple warping path, which keeps the
    result exact; ``np.inf`` disables pruning.

    ``method`` is the pruning scheme: ``"eap"`` (EAPrunedDTW, Herrmann &
    Webb 2021, default), ``"pruned"`` (PrunedDTW, Silva & Batista 2016) or
    ``"none"``. All are exact and all are implemented in both kernels. The
    SIMD anti-diagonal kernel is used automatically when the CPU supports
    AVX2 or NEON, the scalar row kernels otherwise; ``simd=False`` forces
    the row kernels (for benchmarking).
    ``dtype``: precision of the computation and its buffers, ``numpy.float64``
    (default) or ``numpy.float32``; float32 doubles the SIMD lanes at the
    price of about 1e-6 relative error and always runs on the anti-diagonal
    kernel.
    """
    a = _as_sequence(a, 1, "a")
    b = _as_sequence(b, 1, "b")
    w = _resolve_window(a.shape[0], b.shape[0], window)
    return _dtw_dist(a.reshape(-1, 1), b.reshape(-1, 1), w, ub, _metric_id(metric, 1), method,
                     dtype, simd)


def dtw2d_dist(a, b, window=None, metric="euclidean", ub=None, method="eap", dtype=np.float64,
               simd=True):
    """DTW cost of two multivariate sequences of shape (len, dim) without the alignment.

    See :func:`dtw2d` for ``metric`` and :func:`dtw1d_dist` for ``window``,
    ``ub``, ``method``, ``dtype`` and ``simd``.
    """
    a, b = _as_pair_2d(a, b)
    w = _resolve_window(a.shape[0], b.shape[0], window)
    return _dtw_dist(a, b, w, ub, _metric_id(metric, 2), method, dtype, simd)


def dtw1d_dist_batch(X, Y, window=None, ub=None, method="eap", metric="abs", dtype=np.float64):
    """DTW costs of N pairs ``(X[k], Y[k])`` of 1-d sequences, one pair per SIMD lane.

    ``X`` has shape ``(N, n)`` and ``Y`` shape ``(N, m)``; a batch of size 1
    is broadcast against the other. ``ub`` is ``None`` (bound derived per
    pair, exact), a scalar or an array of ``N`` bounds; pairs whose cost
    exceeds their bound get ``inf``. ``method`` as in :func:`dtw1d_dist`
    (here it only changes how early a pair can be abandoned). Returns an
    array of ``N`` costs with the requested ``dtype``.
    """
    X, Y, float32 = _batch_inputs(X, Y, 1, "X", "Y", dtype)
    w = _resolve_window(X.shape[1], Y.shape[1], window)
    return _dtw_dist_batch(X[:, :, None], Y[:, :, None], w, ub, _metric_id(metric, 1),
                           _method_id(method), float32)


def dtw2d_dist_batch(X, Y, window=None, metric="euclidean", ub=None, method="eap",
                     dtype=np.float64):
    """DTW costs of N pairs of multivariate sequences, ``X`` of shape ``(N, n, dim)``
    and ``Y`` of shape ``(N, m, dim)``. See :func:`dtw1d_dist_batch`."""
    X, Y, float32 = _batch_inputs(X, Y, 2, "X", "Y", dtype)
    w = _resolve_window(X.shape[1], Y.shape[1], window)
    return _dtw_dist_batch(X, Y, w, ub, _metric_id(metric, 2), _method_id(method), float32)


cdef _dist_matrix(X, Y, int ndim, window, int mid, int pruning, dtype):
    symmetric = Y is None
    if symmetric:
        Y = X
    float32 = _is_float32(dtype)
    dt = np.float32 if float32 else np.float64
    X = np.ascontiguousarray(X, dtype=dt)
    Y = np.ascontiguousarray(Y, dtype=dt)
    if X.ndim != ndim + 1 or Y.ndim != ndim + 1:
        raise ValueError("X and Y must be {}-dimensional batches, got shapes {} and {}".format(
            ndim + 1, X.shape, Y.shape))
    if ndim == 1:
        X, Y = X[:, :, None], Y[:, :, None]
    cdef Py_ssize_t N = X.shape[0], M = Y.shape[0], i
    out = np.empty((N, M), dtype=dt)
    if N == 0 or M == 0:
        return out
    w = _resolve_window(X.shape[1], Y.shape[1], window)
    for i in range(N):
        if symmetric:
            Xi, Yi, float32 = _batch_inputs(X[i:i + 1], Y[i:], 2, "X", "Y", dt)
            row = _dtw_dist_batch(Xi, Yi, w, None, mid, pruning, float32)
            out[i, i:] = row
            out[i:, i] = row
        else:
            Xi, Yi, float32 = _batch_inputs(X[i:i + 1], Y, 2, "X", "Y", dt)
            out[i] = _dtw_dist_batch(Xi, Yi, w, None, mid, pruning, float32)
    return out


def dtw1d_dist_matrix(X, Y=None, window=None, method="eap", metric="abs", dtype=np.float64):
    """All-pairs DTW costs between the rows of ``X`` (shape ``(N, n)``) and
    ``Y`` (shape ``(M, m)``), as an ``(N, M)`` array. ``Y=None`` computes the
    symmetric matrix of ``X`` with itself (only the upper triangle is
    evaluated). Uses the batch kernel with a per-pair bound."""
    return _dist_matrix(X, Y, 1, window, _metric_id(metric, 1), _method_id(method), dtype)


def dtw2d_dist_matrix(X, Y=None, window=None, metric="euclidean", method="eap", dtype=np.float64):
    """All-pairs DTW costs between batches of multivariate sequences,
    ``X`` of shape ``(N, n, dim)`` and ``Y`` of shape ``(M, m, dim)``.
    See :func:`dtw1d_dist_matrix`."""
    return _dist_matrix(X, Y, 2, window, _metric_id(metric, 2), _method_id(method), dtype)


def simd_info():
    """Which SIMD kernel is active and how many lanes it has.

    Returns ``{"simd": "avx2" | "neon" | "scalar", "lanes_float64": int,
    "lanes_float32": int}``. The environment variable ``PYDTW_SIMD``
    (``scalar``, ``avx2``, ``neon`` or ``auto``) overrides the detection when
    the module is first used."""
    return {"simd": pydtw_simd_name().decode(),
            "lanes_float64": pydtw_simd_lanes(8),
            "lanes_float32": pydtw_simd_lanes(4)}


def pruned_dtw1d(a, b, window=None, method="eap", return_cost_matrix=True, metric="abs"):
    """Same result as :func:`dtw1d`, computed with pruning.

    Runs :func:`dtw1d_dist` first and then :func:`dtw1d` with ``ub`` set to
    that cost, so the full matrix only expands cells that can lie on an
    optimal path.
    """
    a = _as_sequence(a, 1, "a")
    b = _as_sequence(b, 1, "b")
    w = _resolve_window(a.shape[0], b.shape[0], window)
    cdef int mid = _metric_id(metric, 1)
    a2, b2 = a.reshape(-1, 1), b.reshape(-1, 1)
    cost = _dtw_dist(a2, b2, w, None, mid, method, np.float64, True)
    return _dtw_path(a2, b2, w, _slack(cost), mid, return_cost_matrix)


def pruned_dtw2d(a, b, window=None, metric="euclidean", method="eap", return_cost_matrix=True):
    """Same result as :func:`dtw2d`, computed with pruning. See :func:`pruned_dtw1d`."""
    a, b = _as_pair_2d(a, b)
    w = _resolve_window(a.shape[0], b.shape[0], window)
    cdef int mid = _metric_id(metric, 2)
    cost = _dtw_dist(a, b, w, None, mid, method, np.float64, True)
    return _dtw_path(a, b, w, _slack(cost), mid, return_cost_matrix)
