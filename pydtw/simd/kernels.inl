// Generic DTW kernels, templated on a vector-traits type W (see dtw_simd.cpp).
// This file is #included once per instruction set inside a namespace whose
// functions are compiled for that instruction set, so the templates below are
// defined (and therefore compiled) with the right target options.
//
// W must provide:
//   using T  (element type), using v (vector), static constexpr int L (lanes)
//   load(const T*) -> v, store(T*, v), set1(T) -> v, add, sub, mul, min, max,
//   abs(v), sqrt(v), hmin(v) -> T
// Metric ids follow PYDTW_M_*.

template <class W, int METRIC>
static inline typename W::v local_dist(const typename W::T* a, const typename W::T* b,
                                       size_t dim, size_t sa, size_t sb) {
    // a, b point at component 0 of the current group of L cells (or lanes);
    // component d is at a[d * sa] and b[d * sb].
    using v = typename W::v;
    v d;
    if (METRIC == PYDTW_M_ABS) {
        return W::abs(W::sub(W::load(a), W::load(b)));
    } else if (METRIC == PYDTW_M_SQDIFF) {
        v t = W::sub(W::load(a), W::load(b));
        return W::mul(t, t);
    } else if (METRIC == PYDTW_M_EUCLIDEAN || METRIC == PYDTW_M_SQEUCLIDEAN) {
        v t = W::sub(W::load(a), W::load(b));
        d = W::mul(t, t);
        for (size_t k = 1; k < dim; ++k) {
            t = W::sub(W::load(a + k * sa), W::load(b + k * sb));
            d = W::add(d, W::mul(t, t));
        }
        return METRIC == PYDTW_M_EUCLIDEAN ? W::sqrt(d) : d;
    } else if (METRIC == PYDTW_M_CITYBLOCK) {
        d = W::abs(W::sub(W::load(a), W::load(b)));
        for (size_t k = 1; k < dim; ++k)
            d = W::add(d, W::abs(W::sub(W::load(a + k * sa), W::load(b + k * sb))));
        return d;
    } else {  // chebyshev
        d = W::abs(W::sub(W::load(a), W::load(b)));
        for (size_t k = 1; k < dim; ++k)
            d = W::max(d, W::abs(W::sub(W::load(a + k * sa), W::load(b + k * sb))));
        return d;
    }
}

// Scalar helper for the very first cell and for the final check.
template <class T>
static inline T scalar_dist(const T* a, const T* b, size_t dim, int metric) {
    T d = 0;
    switch (metric) {
        case PYDTW_M_ABS: return std::fabs(a[0] - b[0]);
        case PYDTW_M_SQDIFF: { T t = a[0] - b[0]; return t * t; }
        case PYDTW_M_EUCLIDEAN:
        case PYDTW_M_SQEUCLIDEAN:
            for (size_t k = 0; k < dim; ++k) { T t = a[k] - b[k]; d += t * t; }
            return metric == PYDTW_M_EUCLIDEAN ? std::sqrt(d) : d;
        case PYDTW_M_CITYBLOCK:
            for (size_t k = 0; k < dim; ++k) d += std::fabs(a[k] - b[k]);
            return d;
        default:
            for (size_t k = 0; k < dim; ++k) { T t = std::fabs(a[k] - b[k]); if (t > d) d = t; }
            return d;
    }
}

// Cost of one feasible warping path (diagonal, then along the last row or
// column): an upper bound of the DTW cost that lies inside every window wide
// enough for the last cell. Used when the caller passes NaN as the bound.
template <class T>
static T path_bound(const T* a, size_t n, const T* b, size_t m, size_t dim, int metric) {
    const size_t k = std::min(n, m);
    T s = 0;
    for (size_t i = 0; i < k; ++i) s += scalar_dist<T>(a + i * dim, b + i * dim, dim, metric);
    for (size_t i = k; i < n; ++i) s += scalar_dist<T>(a + i * dim, b + (m - 1) * dim, dim, metric);
    for (size_t i = k; i < m; ++i) s += scalar_dist<T>(a + (n - 1) * dim, b + i * dim, dim, metric);
    return s;
}

// Pruning threshold: the bound plus a small relative slack. The bound may come
// from scalar code (or from another kernel) whose rounding differs slightly
// from this kernel's vector arithmetic; the slack keeps the optimum from being
// pruned by such differences. It only loosens pruning, never the result.
template <class T>
static inline T effective_bound(T ub) {
    if (std::isinf(ub)) return ub;
    const T rel = sizeof(T) == 8 ? T(1e-9) : T(1e-4);
    return ub + rel * std::fabs(ub);
}

// ---------------------------------------------------------------------------
// Anti-diagonal single pair
// ---------------------------------------------------------------------------
// Cells on anti-diagonal K (i + j = K) are independent, so they are computed
// L at a time. Diagonal K is stored indexed by row i in a buffer with one
// INFINITY slot in front (index -1). For cell (i, K - i):
//   up   = D[K-1][i-1]   (cell (i-1, j))
//   left = D[K-1][i]     (cell (i, j-1))
//   diag = D[K-2][i-1]   (cell (i-1, j-1))
// so all three predecessors are contiguous loads at offsets i-1 / i.
// b is stored reversed (br[t] = b[m-1-t]) so that b[K-i] = br[m-1-K+i] is an
// ascending contiguous load as well. Both inputs are transposed to (dim, len)
// and padded by L so vector loads may overrun the live range; the overrun
// results land in buffer slots that are reset to INFINITY before anything
// reads them. Pruning: only cells reachable from "live" cells (cost <= ub) of
// the two previous diagonals are computed, as in PrunedDTW.

// Pruning modes (PYDTW_PRUNE_*): 0 none, 1 PrunedDTW (cells above ub are
// dead), 2 EAPrunedDTW (every path still has to pay the last cell, so cells
// above ub - d(a[n-1], b[m-1]) are dead already).
template <class T>
static inline T prune_threshold(int pruning, T ub, T d_last) {
    if (pruning == 0 || std::isinf(ub)) return std::numeric_limits<T>::infinity();
    return effective_bound<T>(pruning == 2 ? ub - d_last : ub);
}

template <class W, int METRIC>
static typename W::T dtw_ad(const typename W::T* a, size_t n, const typename W::T* b, size_t m,
                            size_t dim, size_t w, typename W::T ub, int pruning) {
    using T = typename W::T;
    using v = typename W::v;
    constexpr size_t L = W::L;
    const T INF = std::numeric_limits<T>::infinity();
    if (std::isnan(ub)) ub = path_bound<T>(a, n, b, m, dim, (int)METRIC);
    const T ub_final = effective_bound<T>(ub);
    const T d_last = scalar_dist<T>(a + (n - 1) * dim, b + (m - 1) * dim, dim, (int)METRIC);
    const T ub_eff = prune_threshold<T>(pruning, ub, d_last);

    const size_t na = n + L, mb = m + L;
    std::vector<T> at(dim * na, T(0)), brt(dim * mb, T(0));
    for (size_t i = 0; i < n; ++i)
        for (size_t d = 0; d < dim; ++d) at[d * na + i] = a[i * dim + d];
    for (size_t t = 0; t < m; ++t)
        for (size_t d = 0; d < dim; ++d) brt[d * mb + t] = b[(m - 1 - t) * dim + d];

    // Three diagonal buffers, each n + 2 + L long, with D[-1] at index 0.
    const size_t blen = n + 2 + L;
    std::vector<T> store(3 * blen, INF);
    T* buf[3] = { store.data() + 1, store.data() + blen + 1, store.data() + 2 * blen + 1 };
    // computed range [clo, chi] and live range [llo, lhi] (llo > lhi: empty)
    // of the diagonals K-1 (index 1) and K-2 (index 2) relative to the current.
    long clo1, chi1, llo1, lhi1, clo2 = 0, chi2 = -1, llo2 = 0, lhi2 = -1;

    // Diagonal 0: the single cell (0, 0).
    T v0 = scalar_dist<T>(a, b, dim, (int)METRIC);
    buf[1][0] = v0;
    clo1 = chi1 = 0;
    if (v0 <= ub_eff) { llo1 = lhi1 = 0; } else { llo1 = 0; lhi1 = -1; }
    if (n == 1 && m == 1) return v0 <= ub_final ? v0 : INF;

    const long nl = (long)n, ml = (long)m, wl = (long)w;
    const size_t last_K = n + m - 2;
    for (size_t K = 1; K <= last_K; ++K) {
        const long Kl = (long)K;
        // reachable rows from live cells: up/left from K-1 give [llo1, lhi1+1],
        // diagonal from K-2 gives [llo2+1, lhi2+1]
        long lo = LONG_MAX, hi = LONG_MIN;
        if (llo1 <= lhi1) { lo = llo1; hi = lhi1 + 1; }
        if (llo2 <= lhi2) { lo = std::min(lo, llo2 + 1); hi = std::max(hi, lhi2 + 1); }
        if (lo > hi) return INF;  // nothing alive on the two previous diagonals
        // matrix bounds and Sakoe-Chiba window |i - j| <= w  <=>  |2i - K| <= w
        lo = std::max(lo, std::max(0L, Kl - (ml - 1)));
        hi = std::min(hi, std::min(nl - 1, Kl));
        if (Kl > wl) lo = std::max(lo, (Kl - wl + 1) / 2);
        hi = std::min(hi, (Kl + wl) / 2);
        if (lo > hi) {
            // No cell of this diagonal is inside the window (e.g. odd K with
            // w = 0); paths skip it with a diagonal step, so just record an
            // empty diagonal and go on.
            T* t = buf[2]; buf[2] = buf[1]; buf[1] = buf[0]; buf[0] = t;
            clo2 = clo1; chi2 = chi1; llo2 = llo1; lhi2 = lhi1;
            clo1 = 0; chi1 = -1; llo1 = 0; lhi1 = -1;
            continue;
        }

        T* Dk1 = buf[1];  // K-1
        T* Dk2 = buf[2];  // K-2
        T* Dn = buf[0];   // K (reusing the K-3 buffer)
        // Reset predecessor slots that were not computed on their diagonal:
        // reads from K-1 cover [lo-1, hi], from K-2 cover [lo-1, hi-1].
        for (long x = lo - 1; x < clo1 && x <= hi; ++x) Dk1[x] = INF;
        for (long x = std::max(chi1 + 1, lo - 1); x <= hi; ++x) Dk1[x] = INF;
        for (long x = lo - 1; x < clo2 && x <= hi - 1; ++x) Dk2[x] = INF;
        for (long x = std::max(chi2 + 1, lo - 1); x <= hi - 1; ++x) Dk2[x] = INF;

        const T* bbase = brt.data() + (ml - 1 - Kl);  // + i gives br[m-1-K+i]
        for (long i = lo; i <= hi; i += (long)L) {
            v dist = local_dist<W, METRIC>(at.data() + i, bbase + i, dim, na, mb);
            v up = W::load(Dk1 + i - 1);
            v left = W::load(Dk1 + i);
            v diag = W::load(Dk2 + i - 1);
            W::store(Dn + i, W::add(dist, W::min(W::min(up, left), diag)));
        }
        // live range on this diagonal
        long llo = lo, lhi = hi;
        while (llo <= hi && !(Dn[llo] <= ub_eff)) ++llo;
        while (lhi >= llo && !(Dn[lhi] <= ub_eff)) --lhi;
        // rotate: K becomes K-1, K-1 becomes K-2
        T* t = buf[2]; buf[2] = buf[1]; buf[1] = buf[0]; buf[0] = t;
        clo2 = clo1; chi2 = chi1; llo2 = llo1; lhi2 = lhi1;
        clo1 = lo; chi1 = hi; llo1 = llo; lhi1 = lhi;
    }
    // Final cell (n-1, m-1) lives on diagonal last_K at row n-1.
    if (clo1 <= nl - 1 && nl - 1 <= chi1) {
        T r = buf[1][nl - 1];
        return r <= ub_final ? r : INF;
    }
    return INF;
}

// ---------------------------------------------------------------------------
// Pair batch: lane k of every vector belongs to pair k of the chunk
// ---------------------------------------------------------------------------
// Inputs are repacked lane-major: A[(i * dim + d) * L + lane], B likewise, so
// one load fetches component d of element i for all L pairs. Rows prev/cur are
// (m + 1) vectors. The dependency on the left cell is per lane, so the latency
// of the chain is shared by L pairs. Pruning: a pair is dead once its cost
// exceeds ub[k]; the row loop stops when every lane is dead.

template <class W, int METRIC>
static void dtw_batch(const typename W::T* X, const typename W::T* Y, size_t N, size_t n, size_t m,
                      size_t dim, size_t w, const typename W::T* ub, int pruning,
                      typename W::T* out) {
    using T = typename W::T;
    using v = typename W::v;
    constexpr size_t L = W::L;
    const T INF = std::numeric_limits<T>::infinity();
    std::vector<T> A(n * dim * L), B(m * dim * L), rows(2 * (m + 1) * L);
    alignas(64) T ubl[L], ubf[L], res[L];

    for (size_t c = 0; c < N; c += L) {
        const size_t lanes = std::min(L, N - c);
        for (size_t l = 0; l < L; ++l) {
            const size_t k = c + std::min(l, lanes - 1);  // pad with the last pair
            const T* xk = X + k * n * dim;
            const T* yk = Y + k * m * dim;
            for (size_t i = 0; i < n; ++i)
                for (size_t d = 0; d < dim; ++d) A[(i * dim + d) * L + l] = xk[i * dim + d];
            for (size_t j = 0; j < m; ++j)
                for (size_t d = 0; d < dim; ++d) B[(j * dim + d) * L + l] = yk[j * dim + d];
            T u = ub[k];
            if (std::isnan(u)) u = path_bound<T>(xk, n, yk, m, dim, (int)METRIC);
            ubf[l] = effective_bound<T>(u);
            ubl[l] = prune_threshold<T>(pruning, u,
                scalar_dist<T>(xk + (n - 1) * dim, yk + (m - 1) * dim, dim, (int)METRIC));
        }
        const v ubv = W::load(ubl);
        T* prev = rows.data();
        T* cur = rows.data() + (m + 1) * L;
        for (size_t j = 0; j <= m; ++j) W::store(prev + j * L, W::set1(INF));
        W::store(prev, W::set1(T(0)));
        bool dead = false;
        for (size_t i = 1; i <= n && !dead; ++i) {
            const size_t beg = i > w ? i - w : 1;
            const size_t end = std::min(m, i + w);
            const T* ai = A.data() + (i - 1) * dim * L;
            W::store(cur + (beg - 1) * L, W::set1(INF));
            v left = W::set1(INF), rowmin = W::set1(INF);
            for (size_t j = beg; j <= end; ++j) {
                v dist = local_dist<W, METRIC>(ai, B.data() + (j - 1) * dim * L, dim, L, L);
                v up = W::load(prev + j * L);
                v diag = W::load(prev + (j - 1) * L);
                left = W::add(dist, W::min(W::min(up, left), diag));
                W::store(cur + j * L, left);
                rowmin = W::min(rowmin, left);
            }
            if (end < m) W::store(cur + (end + 1) * L, W::set1(INF));
            // All lanes above their pruning threshold -> nothing can finish
            // within bound. Not applied to the last row: its final cell is
            // judged against the bound itself below (the EAP threshold is
            // tightened by the last cell's own cost).
            if (i < n) dead = W::hmin(W::sub(rowmin, ubv)) > T(0);
            T* t = prev; prev = cur; cur = t;
        }
        W::store(res, W::load(prev + m * L));
        for (size_t l = 0; l < lanes; ++l)
            out[c + l] = (!dead && res[l] <= ubf[l]) ? res[l] : INF;
    }
}

template <class W>
static typename W::T dtw_ad_dispatch(const typename W::T* a, size_t n, const typename W::T* b, size_t m,
                                     size_t dim, int metric, size_t w, typename W::T ub, int pruning) {
    switch (metric) {
        case PYDTW_M_ABS:         return dtw_ad<W, PYDTW_M_ABS>(a, n, b, m, dim, w, ub, pruning);
        case PYDTW_M_SQDIFF:      return dtw_ad<W, PYDTW_M_SQDIFF>(a, n, b, m, dim, w, ub, pruning);
        case PYDTW_M_EUCLIDEAN:   return dtw_ad<W, PYDTW_M_EUCLIDEAN>(a, n, b, m, dim, w, ub, pruning);
        case PYDTW_M_SQEUCLIDEAN: return dtw_ad<W, PYDTW_M_SQEUCLIDEAN>(a, n, b, m, dim, w, ub, pruning);
        case PYDTW_M_CITYBLOCK:   return dtw_ad<W, PYDTW_M_CITYBLOCK>(a, n, b, m, dim, w, ub, pruning);
        default:                  return dtw_ad<W, PYDTW_M_CHEBYSHEV>(a, n, b, m, dim, w, ub, pruning);
    }
}

template <class W>
static void dtw_batch_dispatch(const typename W::T* X, const typename W::T* Y, size_t N, size_t n,
                               size_t m, size_t dim, int metric, size_t w,
                               const typename W::T* ub, int pruning, typename W::T* out) {
    switch (metric) {
        case PYDTW_M_ABS:         dtw_batch<W, PYDTW_M_ABS>(X, Y, N, n, m, dim, w, ub, pruning, out); break;
        case PYDTW_M_SQDIFF:      dtw_batch<W, PYDTW_M_SQDIFF>(X, Y, N, n, m, dim, w, ub, pruning, out); break;
        case PYDTW_M_EUCLIDEAN:   dtw_batch<W, PYDTW_M_EUCLIDEAN>(X, Y, N, n, m, dim, w, ub, pruning, out); break;
        case PYDTW_M_SQEUCLIDEAN: dtw_batch<W, PYDTW_M_SQEUCLIDEAN>(X, Y, N, n, m, dim, w, ub, pruning, out); break;
        case PYDTW_M_CITYBLOCK:   dtw_batch<W, PYDTW_M_CITYBLOCK>(X, Y, N, n, m, dim, w, ub, pruning, out); break;
        default:                  dtw_batch<W, PYDTW_M_CHEBYSHEV>(X, Y, N, n, m, dim, w, ub, pruning, out); break;
    }
}
