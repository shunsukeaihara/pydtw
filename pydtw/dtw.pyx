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
