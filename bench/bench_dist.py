# -*- coding: utf-8 -*-
"""Benchmark of the distance-only API (no pruning vs PrunedDTW vs EAPrunedDTW)
and of the alignment API (with and without the cost matrix).

Run with ``python bench/bench_dist.py`` after ``pip install -e .``.
"""
import time

import numpy as np

from pydtw import dtw1d, dtw1d_dist, dtw2d_dist, pruned_dtw1d


def best_of(f, *args, reps=3, **kw):
    f(*args, **kw)
    times = []
    for _ in range(reps):
        start = time.perf_counter()
        f(*args, **kw)
        times.append(time.perf_counter() - start)
    return min(times)


def row(name, f, x, y, **kw):
    none = best_of(f, x, y, ub=np.inf, **kw)
    pruned = best_of(f, x, y, method="pruned", **kw)
    eap = best_of(f, x, y, method="eap", **kw)
    print("{:34s} {:8.3f}s {:8.3f}s {:8.3f}s".format(name, none, pruned, eap))


def main():
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
    print("1-d, n = m = {}".format(n))
    print("{:34s} {:>9s} {:>9s} {:>9s}".format("case", "no prune", "pruned", "eap"))
    for name, (x, y) in cases.items():
        row(name, dtw1d_dist, x, y)
    x, y = cases["random walk"]
    row("random walk, window = 10%", dtw1d_dist, x, y, window=n // 10)
    x = np.cumsum(rng.normal(size=(n, 8)), axis=0)
    y = np.cumsum(rng.normal(size=(n, 8)), axis=0)
    row("2-d dim = 8 random walk", dtw2d_dist, x, y)

    n = 1000
    q = np.cumsum(rng.normal(size=n))
    cands = [np.cumsum(rng.normal(size=n)) for _ in range(100)] + [q + 0.1 * rng.normal(size=n)]

    def nn(method):
        best = np.inf
        for c in cands:
            best = min(best, dtw1d_dist(q, c, ub=best, method=method))
        return best

    def nn_no_prune():
        return min(dtw1d_dist(q, c, ub=np.inf) for c in cands)

    print("\n1-NN search, {} candidates, n = {}".format(len(cands), n))
    print("{:34s} {:8.3f}s {:8.3f}s {:8.3f}s".format(
        "best-so-far as ub", best_of(nn_no_prune), best_of(nn, "pruned"), best_of(nn, "eap")))

    n = 5000
    print("\nalignment, 1-d, n = m = {}".format(n))
    print("{:34s} {:>9s} {:>9s} {:>9s} {:>9s}".format("case", "dtw1d", "no matrix", "pruned", "pr+no mat"))
    for name, (x, y) in cases.items():
        print("{:34s} {:8.3f}s {:8.3f}s {:8.3f}s {:8.3f}s".format(
            name, best_of(dtw1d, x, y), best_of(dtw1d, x, y, return_cost_matrix=False),
            best_of(pruned_dtw1d, x, y), best_of(pruned_dtw1d, x, y, return_cost_matrix=False)))


if __name__ == "__main__":
    main()
