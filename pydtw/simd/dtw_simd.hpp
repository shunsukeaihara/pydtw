// SIMD kernels for pydtw: runtime-dispatched (scalar / AVX2+FMA / NEON),
// float64 and float32, anti-diagonal single-pair cost and lane-parallel
// pair batches. Plain C interface so Cython can call it.
#pragma once
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Active instruction set: 0 scalar, 1 AVX2 (+FMA), 2 NEON. Detected once;
// the environment variable PYDTW_SIMD=scalar|avx2|neon|auto overrides it
// (an unavailable choice falls back to scalar).
int pydtw_simd_level(void);
const char* pydtw_simd_name(void);
// Vector lanes for an element of `itemsize` bytes (8 or 4) at the active level.
int pydtw_simd_lanes(int itemsize);

// Metric ids shared with dtw.pyx.
enum {
    PYDTW_M_ABS = 0, PYDTW_M_SQDIFF = 1, PYDTW_M_EUCLIDEAN = 2,
    PYDTW_M_SQEUCLIDEAN = 3, PYDTW_M_CITYBLOCK = 4, PYDTW_M_CHEBYSHEV = 5
};

// DTW cost of one pair by anti-diagonal sweeps. a is (n, dim) row-major,
// b is (m, dim). Cells outside the Sakoe-Chiba window `w` are skipped; cells
// whose cost exceeds `ub` are pruned. Returns INFINITY when the cost exceeds
// `ub`. Exact otherwise. ub = NaN means "derive a bound from a simple warping
// path" (exact result, usually much faster); ub = INFINITY disables pruning.
double pydtw_dtw_ad_f64(const double* a, size_t n, const double* b, size_t m,
                        size_t dim, int metric, size_t w, double ub);
float pydtw_dtw_ad_f32(const float* a, size_t n, const float* b, size_t m,
                       size_t dim, int metric, size_t w, float ub);

// DTW cost of N pairs (X[k], Y[k]) computed with one pair per vector lane.
// X is (N, n, dim) row-major, Y is (N, m, dim); ub has N entries (NaN: derive
// a bound, INFINITY: no pruning); out receives N costs (INFINITY where the
// cost exceeds ub[k]).
void pydtw_dtw_batch_f64(const double* X, const double* Y, size_t N, size_t n, size_t m,
                         size_t dim, int metric, size_t w, const double* ub, double* out);
void pydtw_dtw_batch_f32(const float* X, const float* Y, size_t N, size_t n, size_t m,
                         size_t dim, int metric, size_t w, const float* ub, float* out);

#ifdef __cplusplus
}
#endif
