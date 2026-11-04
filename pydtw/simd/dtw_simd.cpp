// See dtw_simd.hpp. The generic kernels live in kernels.inl and are included
// once per instruction set below, inside a region compiled for that set.
#include "dtw_simd.hpp"

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64)
#  define PYDTW_X86 1
#  include <immintrin.h>
#  if defined(_MSC_VER)
#    include <intrin.h>
#  endif
#elif defined(__aarch64__) || defined(_M_ARM64)
#  define PYDTW_ARM64 1
#  include <arm_neon.h>
#endif

// ---------------------------------------------------------------------------
// Detection
// ---------------------------------------------------------------------------
#if defined(PYDTW_X86)
static bool cpu_has_avx2_fma() {
#  if defined(_MSC_VER)
    int info[4];
    __cpuid(info, 0);
    if (info[0] < 7) return false;
    __cpuid(info, 1);
    const bool osxsave = (info[2] & (1 << 27)) != 0;
    const bool avx = (info[2] & (1 << 28)) != 0;
    const bool fma = (info[2] & (1 << 12)) != 0;
    if (!(osxsave && avx && fma)) return false;
    if ((_xgetbv(0) & 0x6) != 0x6) return false;  // OS saves XMM and YMM state
    __cpuidex(info, 7, 0);
    return (info[1] & (1 << 5)) != 0;             // AVX2
#  else
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
#  endif
}
#endif

static int detect_level() {
    int best = 0;
#if defined(PYDTW_X86)
    if (cpu_has_avx2_fma()) best = 1;
#elif defined(PYDTW_ARM64)
    best = 2;  // NEON (with float64 lanes) is part of the AArch64 base ISA
#endif
    const char* force = std::getenv("PYDTW_SIMD");
    if (force && *force) {
        if (!std::strcmp(force, "scalar")) return 0;
        if (!std::strcmp(force, "avx2")) return best == 1 ? 1 : 0;
        if (!std::strcmp(force, "neon")) return best == 2 ? 2 : 0;
    }
    return best;
}

extern "C" int pydtw_simd_level(void) {
    static const int level = detect_level();
    return level;
}

extern "C" const char* pydtw_simd_name(void) {
    switch (pydtw_simd_level()) {
        case 1: return "avx2";
        case 2: return "neon";
        default: return "scalar";
    }
}

extern "C" int pydtw_simd_lanes(int itemsize) {
    const int bytes = pydtw_simd_level() == 1 ? 32 : pydtw_simd_level() == 2 ? 16 : itemsize;
    return bytes / itemsize;
}

// ---------------------------------------------------------------------------
// Scalar traits (also the portable reference for the other two)
// ---------------------------------------------------------------------------
namespace pydtw_scalar {
template <class T_> struct V {
    using T = T_;
    using v = T_;
    static constexpr int L = 1;
    static inline v load(const T* p) { return *p; }
    static inline void store(T* p, v x) { *p = x; }
    static inline v set1(T x) { return x; }
    static inline v add(v a, v b) { return a + b; }
    static inline v sub(v a, v b) { return a - b; }
    static inline v mul(v a, v b) { return a * b; }
    static inline v min(v a, v b) { return b < a ? b : a; }
    static inline v max(v a, v b) { return b > a ? b : a; }
    static inline v abs(v a) { return std::fabs(a); }
    static inline v sqrt(v a) { return std::sqrt(a); }
    static inline T hmin(v a) { return a; }
};
#include "kernels.inl"
}  // namespace pydtw_scalar

// ---------------------------------------------------------------------------
// AVX2 + FMA traits. Only this region is compiled with AVX2 enabled (GCC and
// clang via pragmas; MSVC allows the intrinsics without /arch).
// ---------------------------------------------------------------------------
#if defined(PYDTW_X86)
#  if defined(__clang__)
#    pragma clang attribute push(__attribute__((target("avx2,fma"))), apply_to = function)
#  elif defined(__GNUC__)
#    pragma GCC push_options
#    pragma GCC target("avx2,fma")
#  endif
namespace pydtw_avx2 {
template <class T> struct V;
template <> struct V<double> {
    using T = double;
    using v = __m256d;
    static constexpr int L = 4;
    static inline v load(const T* p) { return _mm256_loadu_pd(p); }
    static inline void store(T* p, v x) { _mm256_storeu_pd(p, x); }
    static inline v set1(T x) { return _mm256_set1_pd(x); }
    static inline v add(v a, v b) { return _mm256_add_pd(a, b); }
    static inline v sub(v a, v b) { return _mm256_sub_pd(a, b); }
    static inline v mul(v a, v b) { return _mm256_mul_pd(a, b); }
    static inline v min(v a, v b) { return _mm256_min_pd(a, b); }
    static inline v max(v a, v b) { return _mm256_max_pd(a, b); }
    static inline v abs(v a) { return _mm256_andnot_pd(_mm256_set1_pd(-0.0), a); }
    static inline v sqrt(v a) { return _mm256_sqrt_pd(a); }
    static inline T hmin(v a) {
        __m128d lo = _mm256_castpd256_pd128(a), hi = _mm256_extractf128_pd(a, 1);
        __m128d m = _mm_min_pd(lo, hi);
        m = _mm_min_pd(m, _mm_unpackhi_pd(m, m));
        return _mm_cvtsd_f64(m);
    }
};
template <> struct V<float> {
    using T = float;
    using v = __m256;
    static constexpr int L = 8;
    static inline v load(const T* p) { return _mm256_loadu_ps(p); }
    static inline void store(T* p, v x) { _mm256_storeu_ps(p, x); }
    static inline v set1(T x) { return _mm256_set1_ps(x); }
    static inline v add(v a, v b) { return _mm256_add_ps(a, b); }
    static inline v sub(v a, v b) { return _mm256_sub_ps(a, b); }
    static inline v mul(v a, v b) { return _mm256_mul_ps(a, b); }
    static inline v min(v a, v b) { return _mm256_min_ps(a, b); }
    static inline v max(v a, v b) { return _mm256_max_ps(a, b); }
    static inline v abs(v a) { return _mm256_andnot_ps(_mm256_set1_ps(-0.0f), a); }
    static inline v sqrt(v a) { return _mm256_sqrt_ps(a); }
    static inline T hmin(v a) {
        __m128 lo = _mm256_castps256_ps128(a), hi = _mm256_extractf128_ps(a, 1);
        __m128 m = _mm_min_ps(lo, hi);
        m = _mm_min_ps(m, _mm_movehl_ps(m, m));
        m = _mm_min_ps(m, _mm_shuffle_ps(m, m, 0x55));
        return _mm_cvtss_f32(m);
    }
};
#include "kernels.inl"
}  // namespace pydtw_avx2
#  if defined(__clang__)
#    pragma clang attribute pop
#  elif defined(__GNUC__)
#    pragma GCC pop_options
#  endif
#endif  // PYDTW_X86

// ---------------------------------------------------------------------------
// NEON traits (AArch64 only; always available there)
// ---------------------------------------------------------------------------
#if defined(PYDTW_ARM64)
namespace pydtw_neon {
template <class T> struct V;
template <> struct V<double> {
    using T = double;
    using v = float64x2_t;
    static constexpr int L = 2;
    static inline v load(const T* p) { return vld1q_f64(p); }
    static inline void store(T* p, v x) { vst1q_f64(p, x); }
    static inline v set1(T x) { return vdupq_n_f64(x); }
    static inline v add(v a, v b) { return vaddq_f64(a, b); }
    static inline v sub(v a, v b) { return vsubq_f64(a, b); }
    static inline v mul(v a, v b) { return vmulq_f64(a, b); }
    static inline v min(v a, v b) { return vminq_f64(a, b); }
    static inline v max(v a, v b) { return vmaxq_f64(a, b); }
    static inline v abs(v a) { return vabsq_f64(a); }
    static inline v sqrt(v a) { return vsqrtq_f64(a); }
    static inline T hmin(v a) { return vminvq_f64(a); }
};
template <> struct V<float> {
    using T = float;
    using v = float32x4_t;
    static constexpr int L = 4;
    static inline v load(const T* p) { return vld1q_f32(p); }
    static inline void store(T* p, v x) { vst1q_f32(p, x); }
    static inline v set1(T x) { return vdupq_n_f32(x); }
    static inline v add(v a, v b) { return vaddq_f32(a, b); }
    static inline v sub(v a, v b) { return vsubq_f32(a, b); }
    static inline v mul(v a, v b) { return vmulq_f32(a, b); }
    static inline v min(v a, v b) { return vminq_f32(a, b); }
    static inline v max(v a, v b) { return vmaxq_f32(a, b); }
    static inline v abs(v a) { return vabsq_f32(a); }
    static inline v sqrt(v a) { return vsqrtq_f32(a); }
    static inline T hmin(v a) { return vminvq_f32(a); }
};
#include "kernels.inl"
}  // namespace pydtw_neon
#endif  // PYDTW_ARM64

// ---------------------------------------------------------------------------
// C entry points
// ---------------------------------------------------------------------------
template <class T>
static T dtw_ad_entry(const T* a, size_t n, const T* b, size_t m, size_t dim, int metric,
                      size_t w, T ub, int pruning) {
    switch (pydtw_simd_level()) {
#if defined(PYDTW_X86)
        case 1: return pydtw_avx2::dtw_ad_dispatch<pydtw_avx2::V<T>>(a, n, b, m, dim, metric, w, ub, pruning);
#endif
#if defined(PYDTW_ARM64)
        case 2: return pydtw_neon::dtw_ad_dispatch<pydtw_neon::V<T>>(a, n, b, m, dim, metric, w, ub, pruning);
#endif
        default: return pydtw_scalar::dtw_ad_dispatch<pydtw_scalar::V<T>>(a, n, b, m, dim, metric, w, ub, pruning);
    }
}

template <class T>
static void dtw_batch_entry(const T* X, const T* Y, size_t N, size_t n, size_t m, size_t dim,
                            int metric, size_t w, const T* ub, int pruning, T* out) {
    switch (pydtw_simd_level()) {
#if defined(PYDTW_X86)
        case 1: pydtw_avx2::dtw_batch_dispatch<pydtw_avx2::V<T>>(X, Y, N, n, m, dim, metric, w, ub, pruning, out); return;
#endif
#if defined(PYDTW_ARM64)
        case 2: pydtw_neon::dtw_batch_dispatch<pydtw_neon::V<T>>(X, Y, N, n, m, dim, metric, w, ub, pruning, out); return;
#endif
        default: pydtw_scalar::dtw_batch_dispatch<pydtw_scalar::V<T>>(X, Y, N, n, m, dim, metric, w, ub, pruning, out); return;
    }
}

extern "C" double pydtw_dtw_ad_f64(const double* a, size_t n, const double* b, size_t m,
                                   size_t dim, int metric, size_t w, double ub, int pruning) {
    return dtw_ad_entry<double>(a, n, b, m, dim, metric, w, ub, pruning);
}
extern "C" float pydtw_dtw_ad_f32(const float* a, size_t n, const float* b, size_t m,
                                  size_t dim, int metric, size_t w, float ub, int pruning) {
    return dtw_ad_entry<float>(a, n, b, m, dim, metric, w, ub, pruning);
}
extern "C" void pydtw_dtw_batch_f64(const double* X, const double* Y, size_t N, size_t n, size_t m,
                                    size_t dim, int metric, size_t w, const double* ub, int pruning,
                                    double* out) {
    dtw_batch_entry<double>(X, Y, N, n, m, dim, metric, w, ub, pruning, out);
}
extern "C" void pydtw_dtw_batch_f32(const float* X, const float* Y, size_t N, size_t n, size_t m,
                                    size_t dim, int metric, size_t w, const float* ub, int pruning,
                                    float* out) {
    dtw_batch_entry<float>(X, Y, N, n, m, dim, metric, w, ub, pruning, out);
}
