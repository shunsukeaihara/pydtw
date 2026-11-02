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


ctypedef double (*metric_ptr)(const double*, const double*, Py_ssize_t) noexcept nogil


cdef inline double d_min(double a, double b, double c) noexcept nogil:
    if a < b and a < c:
        return a
    elif b < c:
        return b
    else:
        return c


cdef inline int d_argmin(double a, double b, double c) noexcept nogil:
    if a <= b and a <= c:
        return 0
    elif b <= c:
        return 1
    else:
        return 2


cdef double abs_distance(const double* a, const double* b, Py_ssize_t dim) noexcept nogil:
    return fabs(a[0] - b[0])


cdef double euclidean_distance(const double* a, const double* b, Py_ssize_t dim) noexcept nogil:
    cdef Py_ssize_t i
    cdef double tmp, d = 0.0
    for i in range(dim):
        tmp = a[i] - b[i]
        d += tmp * tmp
    return sqrt(d)


def check_constraint(a_shape, b_shape, constraint=0, warn=True):
    """Return a Sakoe-Chiba band width that is large enough for the given lengths.

    A band of width ``c`` only allows cells with ``abs(i - j) <= c``; the last
    cell ``(a_shape - 1, b_shape - 1)`` is therefore reachable only if
    ``c >= abs(a_shape - b_shape)``.
    """
    min_size = abs(a_shape - b_shape)
    if constraint < min_size:
        if warn:
            warnings.warn(
                "Constraint {} too small for sequences length {} and {}; using {}".format(
                    constraint, a_shape, b_shape, min_size))
        constraint = min_size
    return constraint


cdef _as_sequence(x, int ndim, str name):
    arr = np.ascontiguousarray(x, dtype=np.float64)
    if arr.ndim != ndim:
        raise ValueError("{} must be {}-dimensional, got shape {}".format(name, ndim, arr.shape))
    if arr.shape[0] == 0:
        raise ValueError("{} must not be empty".format(name))
    return arr


def dtw1d(a, b):
    """Unconstrained DTW between two 1-d sequences.

    Returns ``(cost_matrix, cost, alignment_a, alignment_b)``.
    """
    a = _as_sequence(a, 1, "a")
    b = _as_sequence(b, 1, "b")
    return __dtw1d(a, b, max(a.shape[0], b.shape[0]))


def constrained_dtw1d(a, b, constraint=0):
    """DTW between two 1-d sequences restricted to a Sakoe-Chiba band.

    Only cells with ``abs(i - j) <= constraint`` are evaluated.
    """
    a = _as_sequence(a, 1, "a")
    b = _as_sequence(b, 1, "b")
    constraint = check_constraint(a.shape[0], b.shape[0], constraint)
    return __dtw1d(a, b, constraint)


cdef __dtw1d(double[::1] a, double[::1] b, Py_ssize_t constraint):
    full = np.empty((a.shape[0] + 1, b.shape[0] + 1), dtype=np.float64)
    cdef double[:, ::1] cost_mat = full
    with nogil:
        fill_cost_mat_1d(cost_mat, a, b, constraint)
    align_a, align_b, cost = traceback(cost_mat)
    return full[1:, 1:], cost, align_a, align_b


cdef void fill_cost_mat_1d(double[:, ::1] cost_mat, double[::1] a, double[::1] b,
                           Py_ssize_t constraint) noexcept nogil:
    cdef Py_ssize_t i, j, j0, j1
    cdef Py_ssize_t n = cost_mat.shape[0], m = cost_mat.shape[1]
    cost_mat[:] = INFINITY
    cost_mat[0, 0] = 0.0
    for i in range(1, n):
        j0 = max(1, i - constraint)
        j1 = min(m, i + constraint + 1)
        for j in range(j0, j1):
            cost_mat[i, j] = fabs(a[i - 1] - b[j - 1]) + \
                d_min(cost_mat[i - 1, j], cost_mat[i, j - 1], cost_mat[i - 1, j - 1])


def dtw2d(a, b, metric="euclidean"):
    """Unconstrained DTW between two multivariate sequences of shape (len, dim).

    Returns ``(cost_matrix, cost, alignment_a, alignment_b)``.
    """
    a = _as_sequence(a, 2, "a")
    b = _as_sequence(b, 2, "b")
    return __dtw2d(a, b, max(a.shape[0], b.shape[0]), metric)


def constrained_dtw2d(a, b, constraint=0, metric="euclidean"):
    """DTW between two multivariate sequences restricted to a Sakoe-Chiba band."""
    a = _as_sequence(a, 2, "a")
    b = _as_sequence(b, 2, "b")
    constraint = check_constraint(a.shape[0], b.shape[0], constraint)
    return __dtw2d(a, b, constraint, metric)


cdef __dtw2d(double[:, ::1] a, double[:, ::1] b, Py_ssize_t constraint, metric):
    if a.shape[1] != b.shape[1]:
        raise ValueError('Matrices must have same dimension. a={}, b={}'.format(a.shape[1], b.shape[1]))
    cdef metric_ptr dist_func
    if metric == 'euclidean':
        dist_func = &euclidean_distance
    else:
        raise ValueError("unrecognized metric: {!r}".format(metric))
    full = np.empty((a.shape[0] + 1, b.shape[0] + 1), dtype=np.float64)
    cdef double[:, ::1] cost_mat = full
    with nogil:
        fill_cost_mat_2d(cost_mat, a, b, constraint, dist_func)
    align_a, align_b, cost = traceback(cost_mat)
    return full[1:, 1:], cost, align_a, align_b


cdef void fill_cost_mat_2d(double[:, ::1] cost_mat, double[:, ::1] a, double[:, ::1] b,
                           Py_ssize_t constraint, metric_ptr dist_func) noexcept nogil:
    cdef Py_ssize_t i, j, j0, j1
    cdef Py_ssize_t n = cost_mat.shape[0], m = cost_mat.shape[1], dim = a.shape[1]
    cost_mat[:] = INFINITY
    cost_mat[0, 0] = 0.0
    for i in range(1, n):
        j0 = max(1, i - constraint)
        j1 = min(m, i + constraint + 1)
        for j in range(j0, j1):
            cost_mat[i, j] = dist_func(&a[i - 1, 0], &b[j - 1, 0], dim) + \
                d_min(cost_mat[i - 1, j], cost_mat[i, j - 1], cost_mat[i - 1, j - 1])


cdef traceback(double[:, ::1] cost_mat):
    """Walk back from the last cell of the padded (n+1, m+1) matrix.

    Returns 0-based alignment index lists in forward order and the total cost.
    """
    cdef Py_ssize_t i = cost_mat.shape[0] - 1
    cdef Py_ssize_t j = cost_mat.shape[1] - 1
    cdef double cost = cost_mat[i, j]
    cdef vector[Py_ssize_t] a
    cdef vector[Py_ssize_t] b
    cdef int match
    a.push_back(i - 1)
    b.push_back(j - 1)
    while i > 1 or j > 1:
        if i == 1:
            j -= 1
        elif j == 1:
            i -= 1
        else:
            match = d_argmin(cost_mat[i - 1, j - 1], cost_mat[i - 1, j], cost_mat[i, j - 1])
            if match == 0:
                i -= 1
                j -= 1
            elif match == 1:
                i -= 1
            else:
                j -= 1
        a.push_back(i - 1)
        b.push_back(j - 1)
    align_a = list(a)
    align_b = list(b)
    align_a.reverse()
    align_b.reverse()
    return align_a, align_b, cost


# ---------------------------------------------------------------------------
# Distance-only API: two-row buffer (O(min(n, m)) memory) + PrunedDTW
# ---------------------------------------------------------------------------

cdef double path_upper_bound(const double* a, Py_ssize_t n, const double* b, Py_ssize_t m,
                             Py_ssize_t dim, metric_ptr dist_func) noexcept nogil:
    """Cost of one feasible warping path: the diagonal, then along the last
    row or column. Any feasible path cost is an upper bound of the DTW cost,
    and this path stays inside every band wide enough for the last cell.

    The partial sums are accumulated in the same order the DP accumulates them
    along this path, so the DP reproduces the bound bit for bit and the optimum
    is never pruned by rounding.
    """
    cdef Py_ssize_t i, k = min(n, m)
    cdef double s = 0.0
    for i in range(k):
        s += dist_func(a + i * dim, b + i * dim, dim)
    for i in range(k, n):
        s += dist_func(a + i * dim, b + (m - 1) * dim, dim)
    for i in range(k, m):
        s += dist_func(a + (n - 1) * dim, b + i * dim, dim)
    return s


cdef double pruned_dtw_core(const double* a, Py_ssize_t n, const double* b, Py_ssize_t m,
                            Py_ssize_t dim, metric_ptr dist_func, Py_ssize_t w, double ub,
                            double* prev, double* cur) noexcept nogil:
    """PrunedDTW (Silva & Batista, 2016) on two rows of length m + 1.

    Column 0 is a virtual column left of the first element; ``prev[0] = 0``
    seeds the recursion. Cells whose cost exceeds ``ub`` are pruned: ``sc`` is
    the first column of the next row that can still be reached, ``ec`` is one
    past the last unpruned column of the previous row, and a row is abandoned
    as soon as a pruned cell is at or beyond ``ec`` (everything to its right
    could only be reached through pruned cells). Returns INFINITY when the DTW
    cost exceeds ``ub``.
    """
    cdef Py_ssize_t i, j, beg, end, fill_to, sc = 1, ec = 1, ec_next = 1
    cdef bint smaller_found
    cdef double v
    cdef double* tmp
    for j in range(m + 1):
        prev[j] = INFINITY
    prev[0] = 0.0
    for i in range(1, n + 1):
        beg = max(sc, i - w)
        end = min(m, i + w)
        if beg > end:
            return INFINITY
        cur[beg - 1] = INFINITY
        smaller_found = False
        j = beg
        while j <= end:
            v = dist_func(a + (i - 1) * dim, b + (j - 1) * dim, dim) + \
                d_min(prev[j], cur[j - 1], prev[j - 1])
            cur[j] = v
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
        # Cells after the abandoned one, plus one guard cell to the right of the
        # band, must read as INFINITY from the next row.
        if j <= end:
            j += 1
        fill_to = min(m, end + 1)
        while j <= fill_to:
            cur[j] = INFINITY
            j += 1
        ec = ec_next
        tmp = prev
        prev = cur
        cur = tmp
    v = prev[m]
    return v if v <= ub else INFINITY


cdef double eap_pruned_dtw_core(const double* a, Py_ssize_t n, const double* b, Py_ssize_t m,
                                Py_ssize_t dim, metric_ptr dist_func, Py_ssize_t w, double cutoff,
                                double* prev, double* cur) noexcept nogil:
    """EAPrunedDTW (Herrmann & Webb, 2021) on two row buffers.

    ``prev`` and ``cur`` point one past the start of buffers of length m + 1
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
    cdef double d_last = dist_func(a + (n - 1) * dim, b + (m - 1) * dim, dim)
    # Slack of two ulps of the cutoff makes the comparison robust to rounding:
    # a path whose final cost rounds to <= cutoff is never pruned early.
    cdef double ub = cutoff - d_last + 2.0 * DBL_EPSILON * cutoff
    cdef Py_ssize_t i, j, jbeg, jend, pp_stop, last
    cdef Py_ssize_t next_start = 0, prev_pp = 0, curr_pp = 0
    cdef bint found
    cdef double cost, left
    cdef const double* ai
    cdef double* tmp

    # Row 0: every cell is reached from the left only.
    jend = min(m - 1, w)
    cost = 0.0
    j = 0
    while j <= jend:
        cost += dist_func(a, b + j * dim, dim)
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
        while j <= pp_stop:
            cost = dist_func(ai, b + j * dim, dim) + d_min(prev[j - 1], prev[j], left)
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
            cost = dist_func(ai, b + j * dim, dim) + (prev[j - 1] if prev[j - 1] < left else left)
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
            cost = dist_func(ai, b + j * dim, dim) + left
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


cdef int _method_id(method) except -1:
    if method == 'eap':
        return 0
    if method == 'pruned':
        return 1
    raise ValueError("method must be 'eap' or 'pruned', got {!r}".format(method))


cdef double _dtw_dist(double[:, ::1] a, double[:, ::1] b, Py_ssize_t w, ub, metric_ptr dist_func,
                      method):
    cdef Py_ssize_t n = a.shape[0], m = b.shape[0], dim = a.shape[1]
    # One leading INFINITY slot so that the cores may read index -1.
    cdef vector[double] buf0 = vector[double](m + 2, INFINITY)
    cdef vector[double] buf1 = vector[double](m + 2, INFINITY)
    cdef double bound = INFINITY, result
    cdef bint auto_ub = ub is None
    cdef int method_id = _method_id(method)
    if not auto_ub:
        bound = ub
    with nogil:
        if auto_ub:
            bound = path_upper_bound(&a[0, 0], n, &b[0, 0], m, dim, dist_func)
        if method_id == 0:
            result = eap_pruned_dtw_core(&a[0, 0], n, &b[0, 0], m, dim, dist_func, w, bound,
                                         buf0.data() + 1, buf1.data() + 1)
        else:
            result = pruned_dtw_core(&a[0, 0], n, &b[0, 0], m, dim, dist_func, w, bound,
                                     buf0.data() + 1, buf1.data() + 1)
    return result


cdef metric_ptr _metric(metric) except NULL:
    if metric == 'euclidean':
        return &euclidean_distance
    raise ValueError("unrecognized metric: {!r}".format(metric))


def dtw1d_dist(a, b, ub=None, method="eap"):
    """DTW cost of two 1-d sequences without computing the path.

    Uses O(len(b)) memory. ``ub`` is an upper bound on the cost: cells that
    exceed it are pruned, and ``inf`` is returned when the true cost exceeds
    it. ``None`` (default) derives a bound from a simple warping path, which
    keeps the result exact; ``np.inf`` disables pruning.

    ``method`` selects the pruning scheme: ``"eap"`` (EAPrunedDTW, Herrmann &
    Webb 2021, default) or ``"pruned"`` (PrunedDTW, Silva & Batista 2016).
    Both are exact.
    """
    a = _as_sequence(a, 1, "a")
    b = _as_sequence(b, 1, "b")
    return _dtw_dist(a.reshape(-1, 1), b.reshape(-1, 1), max(a.shape[0], b.shape[0]), ub,
                     &abs_distance, method)


def constrained_dtw1d_dist(a, b, constraint=0, ub=None, method="eap"):
    """Like :func:`dtw1d_dist`, restricted to a Sakoe-Chiba band."""
    a = _as_sequence(a, 1, "a")
    b = _as_sequence(b, 1, "b")
    constraint = check_constraint(a.shape[0], b.shape[0], constraint)
    return _dtw_dist(a.reshape(-1, 1), b.reshape(-1, 1), constraint, ub, &abs_distance, method)


def dtw2d_dist(a, b, metric="euclidean", ub=None, method="eap"):
    """DTW cost of two multivariate sequences of shape (len, dim) without the path.

    See :func:`dtw1d_dist` for ``ub`` and ``method``.
    """
    a = _as_sequence(a, 2, "a")
    b = _as_sequence(b, 2, "b")
    if a.shape[1] != b.shape[1]:
        raise ValueError('Matrices must have same dimension. a={}, b={}'.format(a.shape[1], b.shape[1]))
    return _dtw_dist(a, b, max(a.shape[0], b.shape[0]), ub, _metric(metric), method)


def constrained_dtw2d_dist(a, b, constraint=0, metric="euclidean", ub=None, method="eap"):
    """Like :func:`dtw2d_dist`, restricted to a Sakoe-Chiba band."""
    a = _as_sequence(a, 2, "a")
    b = _as_sequence(b, 2, "b")
    if a.shape[1] != b.shape[1]:
        raise ValueError('Matrices must have same dimension. a={}, b={}'.format(a.shape[1], b.shape[1]))
    constraint = check_constraint(a.shape[0], b.shape[0], constraint)
    return _dtw_dist(a, b, constraint, ub, _metric(metric), method)
