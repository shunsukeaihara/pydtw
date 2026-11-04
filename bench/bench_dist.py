# -*- coding: utf-8 -*-
"""Benchmark of the distance-only API (row kernels vs SIMD anti-diagonal kernel,
float64 vs float32, pair batches) and of the alignment API.

Run with ``python bench/bench_dist.py`` after ``pip install -e .``.
"""
import time

import numpy as np

from pydtw import (dtw1d, dtw1d_dist, dtw2d_dist, pruned_dtw1d, dtw1d_dist_batch,
                   dtw1d_dist_matrix, simd_info)


def best_of(f, *args, reps=3, **kw):
    f(*args, **kw)
    times = []
    for _ in range(reps):
        start = time.perf_counter()
        f(*args, **kw)
        times.append(time.perf_counter() - start)
    return min(times)


def fmt(*ts):
    return " ".join("{:8.3f}s".format(t) for t in ts)


def main():
    info = simd_info()
    print("SIMD: {} ({} float64 lanes, {} float32 lanes)".format(
        info["simd"], info["lanes_float64"], info["lanes_float32"]))
    rng = np.random.default_rng(0)
    n = 5000
    a = np.cumsum(rng.normal(size=n))
    cases = {
        "uniform noise": (rng.random(n), rng.random(n)),
        "random walk": (np.cumsum(rng.normal(size=n)), np.cumsum(rng.normal(size=n))),
        "similar (shifted sine + noise)": (
            np.sin(np.linspace(0, 50, n)) + 0.05 * rng.normal(size=n),
            np.sin(np.linspace(0.2, 50.2, n)) + 0.05 * rng.normal(size=n)),
        "near-identical (a, a + 0.01 noise)": (a, a + 0.01 * rng.normal(size=n)),
    }
    print("\n1-d distance, n = m = {}".format(n))
    print("{:34s} {:>9s} {:>9s} {:>9s} {:>9s} {:>9s}".format(
        "case", "eap", "pruned", "simd f64", "simd f32", "no prune"))
    for name, (x, y) in cases.items():
        print("{:34s} {}".format(name, fmt(
            best_of(dtw1d_dist, x, y, method="eap"),
            best_of(dtw1d_dist, x, y, method="pruned"),
            best_of(dtw1d_dist, x, y, method="simd"),
            best_of(dtw1d_dist, x, y, method="simd", dtype=np.float32),
            best_of(dtw1d_dist, x, y, method="simd", ub=np.inf))))
    x, y = cases["random walk"]
    print("{:34s} {}".format("random walk, window = 10%", fmt(
        best_of(dtw1d_dist, x, y, window=n // 10, method="eap"),
        best_of(dtw1d_dist, x, y, window=n // 10, method="pruned"),
        best_of(dtw1d_dist, x, y, window=n // 10, method="simd"),
        best_of(dtw1d_dist, x, y, window=n // 10, method="simd", dtype=np.float32),
        best_of(dtw1d_dist, x, y, window=n // 10, method="simd", ub=np.inf))))
    x = np.cumsum(rng.normal(size=(n, 8)), axis=0)
    y = np.cumsum(rng.normal(size=(n, 8)), axis=0)
    print("{:34s} {}".format("2-d dim = 8 random walk", fmt(
        best_of(dtw2d_dist, x, y, method="eap"),
        best_of(dtw2d_dist, x, y, method="pruned"),
        best_of(dtw2d_dist, x, y, method="simd"),
        best_of(dtw2d_dist, x, y, method="simd", dtype=np.float32),
        best_of(dtw2d_dist, x, y, method="simd", ub=np.inf))))

    n = 1000
    q = np.cumsum(rng.normal(size=n))
    cands = [np.cumsum(rng.normal(size=n)) for _ in range(100)] + [q + 0.1 * rng.normal(size=n)]
    C = np.array(cands)

    def nn(method):
        best = np.inf
        for c in cands:
            best = min(best, dtw1d_dist(q, c, ub=best, method=method))
        return best

    print("\n1-NN search, {} candidates, n = {}".format(len(cands), n))
    print("{:34s} {:>9s} {:>9s} {:>9s} {:>9s} {:>9s}".format(
        "", "eap", "pruned", "simd f64", "batch f64", "batch f32"))
    print("{:34s} {}".format("best-so-far as ub / batch", fmt(
        best_of(nn, "eap"), best_of(nn, "pruned"), best_of(nn, "simd"),
        best_of(lambda: dtw1d_dist_batch(q[None, :], C).min()),
        best_of(lambda: dtw1d_dist_batch(q[None, :], C, dtype=np.float32).min()))))

    N, n = 200, 100
    X = np.cumsum(rng.normal(size=(N, n)), axis=1)
    print("\ndistance matrix, {} x {} series of length {}".format(N, N, n))
    print("{:34s} {:>9s} {:>9s} {:>9s}".format("", "eap loop", "matrix f64", "matrix f32"))
    print("{:34s} {}".format("all pairs", fmt(
        best_of(lambda: [[dtw1d_dist(x, y, method="eap") for y in X] for x in X], reps=1),
        best_of(dtw1d_dist_matrix, X),
        best_of(dtw1d_dist_matrix, X, dtype=np.float32))))

    n = 5000
    print("\nalignment, 1-d, n = m = {}".format(n))
    print("{:34s} {:>9s} {:>9s} {:>9s} {:>9s}".format("case", "dtw1d", "no matrix", "pruned", "pr+no mat"))
    for name, (x, y) in cases.items():
        print("{:34s} {}".format(name, fmt(
            best_of(dtw1d, x, y), best_of(dtw1d, x, y, return_cost_matrix=False),
            best_of(pruned_dtw1d, x, y), best_of(pruned_dtw1d, x, y, return_cost_matrix=False))))


if __name__ == "__main__":
    main()
