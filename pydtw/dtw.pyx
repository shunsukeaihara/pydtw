# -*- coding: utf-8 -*-
# distutils: language = c++
# distutils: sources = dtw.cpp
import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport fabs, sqrt, INFINITY
from libcpp.vector cimport vector
import warnings

ctypedef double (*metric_ptr)(double[::1] a, double[::1])

cdef inline double d_min(double a, double b, double c):
    if a < b and a < c:
        return a
    elif b < c:
        return b
    else:
        return c

cdef inline int d_argmin(double  a, double b, double c):
    if a <= b and a <= c:
        return 0
    elif b <= c:
        return 1
    else:
        return 2


@cython.boundscheck(False)
@cython.wraparound(False)
cdef double euclidean_distance(double[::1] a, double[::1] b):
    cdef int i
    cdef double tmp, d
    d = 0
    for i in range(a.shape[0]):
        tmp = a[i] - b[i]
        d += tmp * tmp
    return sqrt(d)


@cython.boundscheck(False)
@cython.wraparound(False)
def check_constraint(a_shape, b_shape, constraint=0, warn=True):
    cdef int min_size = abs(a_shape - b_shape) + 1
    if constraint < min_size:
        if warn:
            warnings.warn("Constraint {} too small for sequences length {} and {}; using {}".format(constraint, a_shape, b_shape, min_size))
        constraint = min_size
    return constraint


@cython.boundscheck(False)
@cython.wraparound(False)
def dtw1d(np.ndarray[np.float64_t, ndim=1, mode="c"] a, np.ndarray[np.float64_t, ndim=1, mode="c"] b):
    cdef int constraint = abs(a.shape[0] - b.shape[0]) + 1
    cost_mat, cost, align_a, align_b = __dtw1d(a, b, constraint)
    return cost_mat, cost, align_a, align_b


@cython.boundscheck(False)
@cython.wraparound(False)
def constrained_dtw1d(np.ndarray[np.float64_t, ndim=1, mode="c"] a, np.ndarray[np.float64_t, ndim=1, mode="c"] b,
                       constraint=0):
    constraint = check_constraint(a.shape[0],b.shape[0],constraint)
    cost_mat, cost, align_a, align_b = __dtw1d(a, b, constraint)
    return cost_mat, cost, align_a, align_b


@cython.boundscheck(False)
@cython.wraparound(False)
cdef __dtw1d(np.ndarray[np.float64_t, ndim=1, mode="c"] a, np.ndarray[np.float64_t, ndim=1, mode="c"] b,
             int constraint):
    cdef double[:, ::1] cost_mat = create_cost_mat_1d(a, b, constraint)
    align_a, align_b, cost = traceback(cost_mat, a.shape[0], b.shape[0])
    align_a.reverse()
    align_b.reverse()
    return cost_mat, cost, align_a, align_b


@cython.boundscheck(False)
@cython.wraparound(False)
cdef double[:, ::1] create_cost_mat_1d(double[::1] a, double[::1]b, int constraint):
    cdef double[:, ::1] cost_mat = np.empty((a.shape[0] + 1, b.shape[0] + 1), dtype=np.float64)
    cost_mat[:] = INFINITY
    cost_mat[0, 0] = 0
    cdef int i, j
    for i in range(1, cost_mat.shape[0]):
        for j in range(max(1, i-constraint), min(cost_mat.shape[1], i+constraint+1)):
            cost_mat[i, j] = fabs(a[i - 1] - b[j - 1]) +\
                d_min(cost_mat[i - 1, j], cost_mat[i, j - 1], cost_mat[i - 1, j - 1])
    return cost_mat[1:, 1:]


@cython.boundscheck(False)
@cython.wraparound(False)
def dtw2d(np.ndarray[np.float64_t, ndim=2, mode="c"] a,
          np.ndarray[np.float64_t, ndim=2, mode="c"] b, metric="euclidean"):
    cdef int constraint = abs(a.shape[0] - b.shape[0]) + 1
    cost_mat, cost, align_a, align_b = __dtw2d(a, b, constraint, metric)
    return cost_mat, cost, align_a, align_b


@cython.boundscheck(False)
@cython.wraparound(False)
def constrained_dtw2d(np.ndarray[np.float64_t, ndim=2, mode="c"] a,
                       np.ndarray[np.float64_t, ndim=2, mode="c"] b, constraint=0, metric="euclidean"):
    constraint = check_constraint(a.shape[0],b.shape[0],constraint)
    cost_mat, cost, align_a, align_b = __dtw2d(a, b, constraint, metric)
    return cost_mat, cost, align_a, align_b


@cython.boundscheck(False)
@cython.wraparound(False)
cdef __dtw2d(np.ndarray[np.float64_t, ndim=2, mode="c"] a,
             np.ndarray[np.float64_t, ndim=2, mode="c"] b, int constraint, metric):
    assert a.shape[1] == b.shape[1], 'Matrices must have same dimention. a={}, b={}'.format(a.shape[1], b.shape[1])
    cdef metric_ptr dist_func
    if metric == 'euclidean':
        dist_func = &euclidean_distance
    else:
        raise ValueError("unrecognized metric")
    cdef double[:, ::1] cost_mat = create_cost_mat_2d(a, b, constraint, dist_func)
    align_a, align_b, cost = traceback(cost_mat, a.shape[0], b.shape[0])
    align_a.reverse()
    align_b.reverse()
    return cost_mat, cost, align_a, align_b


@cython.boundscheck(False)
@cython.wraparound(False)
cdef double[:, ::1] create_cost_mat_2d(double[:, ::1] a, double[:, ::1] b, int constraint, metric_ptr dist_func):
    cdef double[:, ::1] cost_mat = np.empty((a.shape[0] + 1, b.shape[0] + 1), dtype=np.float64)
    cost_mat[:] = INFINITY
    cost_mat[0, 0] = 0
    cdef int i, j
    for i in range(1, cost_mat.shape[0]):
        for j in range(max(1, i-constraint), min(cost_mat.shape[1], i+constraint+1)):
            cost_mat[i, j] = dist_func(a[i - 1], b[j - 1]) +\
                d_min(cost_mat[i - 1, j], cost_mat[i, j - 1], cost_mat[i - 1, j - 1])
    return cost_mat[1:, 1:]


@cython.boundscheck(False)
@cython.wraparound(False)
cdef traceback(double[:, ::1] cost_mat, int ilen, int jlen):
    cdef int i, j
    i = ilen - 1
    j = jlen - 1
    cdef double cost = cost_mat[i, j]
    cdef vector[int] a
    cdef vector[int] b
    a.push_back(i)
    b.push_back(j)
    cdef int match
    while (i > 0 or j > 0):
        match = d_argmin(cost_mat[i - 1, j - 1], cost_mat[i - 1, j], cost_mat[i, j - 1])
        if match == 0:
            i -= 1
            j -= 1
        elif match == 1:
            i -= 1
        else:
            j -= 1
        a.push_back(i)
        b.push_back(j)
    return a, b, cost

