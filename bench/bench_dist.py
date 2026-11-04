# -*- coding: utf-8 -*-
"""Benchmark of the distance-only API: pruning scheme (none / pruned / eap)
x kernel (scalar row kernels via simd=False, SIMD float64, SIMD float32),
plus batches, distance matrices and the alignment API.

Run with ``python bench/bench_dist.py`` after ``pip install -e .``.
"""
import time

import numpy as np

from pydtw import (dtw1d, dtw1d_dist, dtw2d_dist, pruned_dtw1d, dtw1d_dist_batch,
                   dtw1d_dist_matrix, simd_info)

METHODS = ("none", "pruned", "eap")
KERNELS = (("scalar", dict(simd=False)), ("simd f64", dict()), ("simd f32", dict(dtype=np.float32)))


def best_of(f, *args, reps=3, **kw):
    f(*args, **kw)
    times = []
    for _ in range(reps):
        start = time.perf_counter()
        f(*args, **kw)
        times.append(time.perf_counter() - start)
    return min(times)


def grid(f, *args, **kw):
    """9 timings: for each kernel, the three pruning schemes."""
    out = []
    for _, kkw in KERNELS:
        for method in METHODS:
            out.append(best_of(f, *args, method=method, **kkw, **kw))
    return out


def header(label):
    print("{:34s} {:>29s}   {:>29s}   {:>29s}".format(label, "scalar", "simd f64", "simd f32"))
    print("{:34s} {}".format("", "   ".join("{:>9s} {:>9s} {:>9s}".format(*METHODS) for _ in KERNELS)))


def row(name, ts):
    cells = ["{:8.3f}s".format(t) for t in ts]
    print("{:34s} {}   {}   {}".format(name, " ".join(cells[0:3]), " ".join(cells[3:6]), " ".join(cells[6:9])))


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
    print("\n1-d distance, n = m = {} (seconds, best of 3)".format(n))
    header("case")
    for name, (x, y) in cases.items():
        row(name, grid(dtw1d_dist, x, y))
    x, y = cases["random walk"]
    row("random walk, window = 10%", grid(dtw1d_dist, x, y, window=n // 10))
    x = np.cumsum(rng.normal(size=(n, 8)), axis=0)
    y = np.cumsum(rng.normal(size=(n, 8)), axis=0)
    row("2-d dim = 8 random walk", grid(dtw2d_dist, x, y))

    n = 1000
    q = np.cumsum(rng.normal(size=n))
    cands = [np.cumsum(rng.normal(size=n)) for _ in range(100)] + [q + 0.1 * rng.normal(size=n)]
    C = np.array(cands)

    def nn(method, **kw):
        best = np.inf
        for c in cands:
            best = min(best, dtw1d_dist(q, c, ub=best, method=method, **kw))
        return best

    print("\n1-NN search, {} candidates, n = {} (best-so-far as ub)".format(len(cands), n))
    header("")
    row("per-pair dtw1d_dist", grid(nn))
    print("{:34s} {}".format("dtw1d_dist_batch (f64 / f32)", "   ".join(
        "{:8.3f}s".format(best_of(lambda **kw: dtw1d_dist_batch(q[None, :], C, **kw).min(), **kw))
        for kw in (dict(), dict(dtype=np.float32)))))

    N, n = 200, 100
    X = np.cumsum(rng.normal(size=(N, n)), axis=1)
    print("\ndistance matrix, {} x {} series of length {}".format(N, N, n))
    print("{:34s} {:>9s} {:>9s} {:>9s}".format("", "loop", "matrix f64", "matrix f32"))
    print("{:34s} {:8.3f}s {:8.3f}s {:8.3f}s".format(
        "all pairs",
        best_of(lambda: [[dtw1d_dist(x, y) for y in X] for x in X], reps=1),
        best_of(dtw1d_dist_matrix, X),
        best_of(dtw1d_dist_matrix, X, dtype=np.float32)))

    n = 5000
    print("\nalignment, 1-d, n = m = {}".format(n))
    print("{:34s} {:>9s} {:>9s} {:>9s} {:>9s}".format("case", "dtw1d", "no matrix", "pruned", "pr+no mat"))
    for name, (x, y) in cases.items():
        print("{:34s} {:8.3f}s {:8.3f}s {:8.3f}s {:8.3f}s".format(
            name, best_of(dtw1d, x, y), best_of(dtw1d, x, y, return_cost_matrix=False),
            best_of(pruned_dtw1d, x, y), best_of(pruned_dtw1d, x, y, return_cost_matrix=False)))


if __name__ == "__main__":
    main()
