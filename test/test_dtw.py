# -*- coding: utf-8 -*-
import warnings
from unittest import TestCase

import numpy as np
import pytest

from pydtw import (dtw1d, dtw2d, dtw1d_dist, dtw2d_dist, pruned_dtw1d, pruned_dtw2d,
                   constrained_dtw1d, constrained_dtw2d)


def reference_dtw(a, b, band=None):
    """Plain O(n*m) numpy DTW used as an oracle in the tests."""
    n, m = len(a), len(b)
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if band is not None and abs(i - j) > band:
                continue
            if a.ndim == 1:
                d = abs(a[i - 1] - b[j - 1])
            else:
                d = np.linalg.norm(a[i - 1] - b[j - 1])
            D[i, j] = d + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    i, j = n, m
    pa, pb = [i - 1], [j - 1]
    while i > 1 or j > 1:
        if i == 1:
            j -= 1
        elif j == 1:
            i -= 1
        else:
            k = int(np.argmin([D[i - 1, j - 1], D[i - 1, j], D[i, j - 1]]))
            if k == 0:
                i -= 1
                j -= 1
            elif k == 1:
                i -= 1
            else:
                j -= 1
        pa.append(i - 1)
        pb.append(j - 1)
    return D[1:, 1:], D[n, m], pa[::-1], pb[::-1]


A2D = np.array([[0.30590702, 0.71122566, 0.87785326, 0.41896773, 0.27411357],
                [0.78599928, 0.13459698, 0.7441022, 0.79060971, 0.9681483],
                [0.67467294, 0.65713991, 0.06253549, 0.2692394, 0.34000727],
                [0.79171092, 0.05383535, 0.85633251, 0.79434776, 0.07521822],
                [0.04116069, 0.93189229, 0.56063383, 0.15773727, 0.41580332],
                [0.88198997, 0.70688883, 0.04622127, 0.6324347, 0.94977589],
                [0.45343914, 0.58697004, 0.56443057, 0.53493055, 0.03360396],
                [0.53693851, 0.34162195, 0.26667218, 0.44254594, 0.08195404],
                [0.63299858, 0.54636604, 0.15679107, 0.53235207, 0.10908979],
                [0.94692708, 0.03055877, 0.52066543, 0.15592486, 0.90842773],
                [0.63713889, 0.13715326, 0.12382141, 0.18114644, 0.64222287],
                [0.18344581, 0.36095063, 0.04299809, 0.93636622, 0.97176932],
                [0.46324373, 0.03592446, 0.2815672, 0.024673, 0.03278622],
                [0.40219816, 0.37011199, 0.89306112, 0.69104287, 0.65363817],
                [0.12373541, 0.66056791, 0.72306298, 0.51247937, 0.89520386],
                [0.62731672, 0.06151905, 0.20113791, 0.54437935, 0.24038752],
                [0.6650012, 0.93281123, 0.45220594, 0.5920064, 0.84662455],
                [0.35575232, 0.53527631, 0.26486243, 0.05665991, 0.37931886],
                [0.7680193, 0.38466873, 0.42141126, 0.43820938, 0.18742516],
                [0.02902629, 0.61636535, 0.14813542, 0.80206397, 0.96151947]])
B2D = np.array([[0.37700537, 0.06839722, 0.90342088, 0.67260341, 0.21236238],
                [0.81431271, 0.30039987, 0.7422115, 0.81856106, 0.73533936],
                [0.0785568, 0.20518932, 0.37183274, 0.66525787, 0.91777226],
                [0.26032762, 0.43689793, 0.50414443, 0.62953974, 0.26879355],
                [0.0772129, 0.6953331, 0.94603823, 0.04144669, 0.64313459],
                [0.33403999, 0.58964715, 0.85784944, 0.1092154, 0.90425047],
                [0.57409039, 0.99600997, 0.5179884, 0.39531699, 0.14252428],
                [0.79165219, 0.57988299, 0.62778483, 0.69332613, 0.63002323],
                [0.0249018, 0.73852518, 0.98462038, 0.24437005, 0.89890168],
                [0.37329552, 0.7420235, 0.51391262, 0.18437844, 0.34257216],
                [0.7842619, 0.21165934, 0.95643329, 0.399339, 0.93537423],
                [0.13890876, 0.32683409, 0.90477286, 0.82666108, 0.01006905],
                [0.90557923, 0.28472431, 0.09018984, 0.86671066, 0.90903727],
                [0.54199152, 0.83542679, 0.65840321, 0.95418474, 0.87883887],
                [0.18411973, 0.78678018, 0.39588103, 0.92501485, 0.87820573],
                [0.83208939, 0.70154756, 0.96767817, 0.16727018, 0.49564328],
                [0.47970899, 0.59563789, 0.25823924, 0.43123425, 0.16967742],
                [0.83565957, 0.35983831, 0.73003322, 0.38112664, 0.61784956],
                [0.89832668, 0.45160468, 0.56442061, 0.47974733, 0.39944765],
                [0.68100611, 0.26754648, 0.79144153, 0.14882468, 0.29260584],
                [0.49658543, 0.88300153, 0.4789712, 0.75601603, 0.64980613],
                [0.98528348, 0.4945167, 0.60385308, 0.64908159, 0.40022932],
                [0.7663773, 0.73506873, 0.94322472, 0.72975117, 0.23751136],
                [0.00794367, 0.72420966, 0.07036621, 0.5213072, 0.60074303],
                [0.78067221, 0.48692555, 0.38134388, 0.16889658, 0.47186969]])


class DtwTest(TestCase):
    def test_dtw1d(self):
        a = np.array([0, 0, 1, 1, 2, 4, 2, 1, 2, 0], dtype=np.float64)
        b = np.array([1, 1, 1, 2, 2, 2, 2, 3, 2, 0], dtype=np.float64)
        expected_a = [0, 1, 2, 3, 4, 4, 4, 4, 5, 6, 7, 8, 9]
        expected_b = [0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 8, 9]
        for f in (dtw1d, pruned_dtw1d):
            cost_mat, cost, al_a, al_b = f(a, b)
            self.assertEqual(cost, 4.0)
            self.assertEqual(al_a, expected_a)
            self.assertEqual(al_b, expected_b)
            cost_mat, cost, al_a, al_b = f(a, b, window=2)
            self.assertEqual(cost, 4.0)
            self.assertEqual(al_a, expected_a)
            self.assertEqual(al_b, expected_b)
        cost_mat, cost, al_a, al_b = constrained_dtw1d(a, b, 2)
        self.assertEqual(cost, 4.0)
        self.assertEqual(al_a, expected_a)

    def test_dtw2d(self):
        expected_a = [0, 1, 2, 3, 4, 5, 6, 6, 7, 8, 9, 10, 11, 11, 11,
                      12, 12, 13, 14, 15, 16, 17, 18, 19, 19]
        expected_b = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                      15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
        for f in (dtw2d, pruned_dtw2d):
            cost_mat, cost, al_a, al_b = f(A2D, B2D)
            self.assertEqual(al_a, expected_a)
            self.assertEqual(al_b, expected_b)
            cost_mat, cost, al_a, al_b = f(A2D, B2D, window=5)
            self.assertEqual(al_a, expected_a)
            self.assertEqual(al_b, expected_b)
        cost_mat, cost, al_a, al_b = constrained_dtw2d(A2D, B2D, 5)
        self.assertEqual(al_a, expected_a)

    def test_matches_reference_1d(self):
        rng = np.random.default_rng(0)
        for n, m in [(1, 1), (1, 7), (7, 1), (10, 10), (10, 25), (40, 9), (64, 64)]:
            a, b = rng.random(n), rng.random(m)
            ref_cm, ref_cost, ref_a, ref_b = reference_dtw(a, b)
            for f in (dtw1d, pruned_dtw1d):
                cm, cost, al_a, al_b = f(a, b)
                self.assertAlmostEqual(cost, ref_cost, msg=(n, m, f.__name__))
                self.assertEqual(al_a, ref_a, (n, m, f.__name__))
                self.assertEqual(al_b, ref_b, (n, m, f.__name__))
            np.testing.assert_allclose(dtw1d(a, b)[0], ref_cm)

    def test_matches_reference_2d(self):
        rng = np.random.default_rng(1)
        for n, m, d in [(1, 1, 3), (1, 6, 2), (6, 1, 2), (12, 12, 4), (10, 30, 1), (33, 8, 5)]:
            a, b = rng.random((n, d)), rng.random((m, d))
            ref_cm, ref_cost, ref_a, ref_b = reference_dtw(a, b)
            for f in (dtw2d, pruned_dtw2d):
                cm, cost, al_a, al_b = f(a, b)
                self.assertAlmostEqual(cost, ref_cost, msg=(n, m, d, f.__name__))
                self.assertEqual(al_a, ref_a, (n, m, d, f.__name__))
                self.assertEqual(al_b, ref_b, (n, m, d, f.__name__))
            np.testing.assert_allclose(dtw2d(a, b)[0], ref_cm)

    def test_window_matches_reference(self):
        rng = np.random.default_rng(2)
        for n, m, band in [(20, 20, 0), (20, 20, 3), (20, 30, 10), (30, 20, 12), (15, 15, 100)]:
            a, b = rng.random(n), rng.random(m)
            ref_cm, ref_cost, ref_a, ref_b = reference_dtw(a, b, band)
            for f in (dtw1d, pruned_dtw1d):
                cm, cost, al_a, al_b = f(a, b, window=band)
                self.assertAlmostEqual(cost, ref_cost, msg=(n, m, band, f.__name__))
                self.assertEqual(al_a, ref_a, (n, m, band, f.__name__))
                self.assertEqual(al_b, ref_b, (n, m, band, f.__name__))
            np.testing.assert_allclose(dtw1d(a, b, window=band)[0], ref_cm)
            a2, b2 = rng.random((n, 3)), rng.random((m, 3))
            ref_cm, ref_cost, ref_a, ref_b = reference_dtw(a2, b2, band)
            for f in (dtw2d, pruned_dtw2d):
                cm, cost, al_a, al_b = f(a2, b2, window=band)
                self.assertAlmostEqual(cost, ref_cost, msg=(n, m, band, f.__name__))
                self.assertEqual(al_a, ref_a, (n, m, band, f.__name__))
                self.assertEqual(al_b, ref_b, (n, m, band, f.__name__))

    def test_window_too_small_is_widened(self):
        a, b = np.arange(5.0), np.arange(9.0)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cm, cost, al_a, al_b = dtw1d(a, b, window=1)
        self.assertEqual(len(w), 1)
        self.assertTrue(np.isfinite(cost))
        self.assertEqual(al_a[-1], 4)
        self.assertEqual(al_b[-1], 8)
        # abs(n - m) is enough for the end cell to be reachable: no warning.
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            dtw1d(a, b, window=4)
        self.assertEqual(len(w), 0)
        with self.assertRaises(ValueError):
            dtw1d(a, b, window=-1)

    def test_upper_bound_on_alignment(self):
        a = np.array([0, 0, 1, 1, 2, 4, 2, 1, 2, 0], dtype=np.float64)
        b = np.array([1, 1, 1, 2, 2, 2, 2, 3, 2, 0], dtype=np.float64)
        cm, cost, al_a, al_b = dtw1d(a, b, ub=4.0)
        self.assertEqual(cost, 4.0)
        self.assertEqual(al_a, dtw1d(a, b)[2])
        cm, cost, al_a, al_b = dtw1d(a, b, ub=3.9)
        self.assertEqual(cost, np.inf)
        self.assertEqual((al_a, al_b), ([], []))
        self.assertEqual(cm.shape, (10, 10))
        cm, cost, al_a, al_b = dtw2d(A2D, B2D, ub=1.0)
        self.assertEqual(cost, np.inf)
        self.assertEqual((al_a, al_b), ([], []))
        # ub equal to the exact cost never prunes the optimum.
        rng = np.random.default_rng(4)
        for _ in range(100):
            n, m = rng.integers(1, 50, size=2)
            x, y = np.cumsum(rng.normal(size=n)), np.cumsum(rng.normal(size=m))
            ref = dtw1d(x, y)
            got = dtw1d(x, y, ub=ref[1])
            self.assertEqual(got[1], ref[1], (n, m))
            self.assertEqual(got[2:], ref[2:], (n, m))

    def test_returns_ndarray_cost_matrix(self):
        cm, cost, al_a, al_b = dtw1d(np.arange(4.0), np.arange(6.0))
        self.assertIsInstance(cm, np.ndarray)
        self.assertEqual(cm.shape, (4, 6))
        self.assertIsInstance(cost, float)
        self.assertEqual((al_a[0], al_b[0]), (0, 0))
        self.assertEqual((al_a[-1], al_b[-1]), (3, 5))

    def test_accepts_lists_ints_and_strided_input(self):
        ref = dtw1d(np.array([0.0, 1.0, 2.0]), np.array([0.0, 2.0]))[1]
        self.assertEqual(dtw1d([0, 1, 2], [0, 2])[1], ref)
        self.assertEqual(dtw1d(np.array([0, 1, 2]), np.array([0, 2]))[1], ref)
        self.assertEqual(dtw1d(np.arange(6.0)[::2] / 2, np.array([0.0, 2.0]))[1], ref)
        a = np.random.default_rng(3).random((5, 4))
        self.assertEqual(dtw2d(a[:, ::2], a[:, ::2])[1], 0.0)

    def test_invalid_input(self):
        with self.assertRaises(ValueError):
            dtw1d(np.zeros(0), np.zeros(3))
        with self.assertRaises(ValueError):
            dtw1d(np.zeros(3), np.zeros((3, 1)))
        with self.assertRaises(ValueError):
            dtw2d(np.zeros((3, 2)), np.zeros((3, 3)))
        with self.assertRaises(ValueError):
            dtw2d(np.zeros((0, 2)), np.zeros((3, 2)))
        with self.assertRaises(ValueError):
            dtw2d(np.zeros((3, 2)), np.zeros((3, 2)), metric="cosine")
        with self.assertRaises(ValueError):
            dtw1d(np.zeros(3), np.zeros(3), ub=np.nan)


class DtwDistTest(TestCase):
    METHODS = ("eap", "pruned")

    def test_matches_path_version(self):
        rng = np.random.default_rng(10)
        for n, m in [(1, 1), (1, 9), (9, 1), (16, 16), (20, 45), (50, 13), (200, 200)]:
            a, b = rng.random(n), rng.random(m)
            a2, b2 = rng.random((n, 3)), rng.random((m, 3))
            expected1 = dtw1d(a, b)[1]
            expected2 = dtw2d(a2, b2)[1]
            for method in self.METHODS:
                self.assertAlmostEqual(dtw1d_dist(a, b, method=method), expected1, msg=(n, m, method))
                self.assertAlmostEqual(dtw1d_dist(a, b, ub=np.inf, method=method), expected1)
                self.assertAlmostEqual(dtw2d_dist(a2, b2, method=method), expected2, msg=(n, m, method))
                self.assertAlmostEqual(dtw2d_dist(a2, b2, ub=np.inf, method=method), expected2)

    def test_window_matches_path_version(self):
        rng = np.random.default_rng(11)
        for n, m, band in [(20, 20, 0), (20, 20, 3), (20, 30, 10), (30, 20, 12), (15, 15, 100), (300, 280, 25)]:
            a, b = rng.random(n), rng.random(m)
            a2, b2 = rng.random((n, 2)), rng.random((m, 2))
            expected1 = dtw1d(a, b, window=band)[1]
            expected2 = dtw2d(a2, b2, window=band)[1]
            for method in self.METHODS:
                self.assertAlmostEqual(dtw1d_dist(a, b, window=band, method=method), expected1,
                                       msg=(n, m, band, method))
                self.assertAlmostEqual(dtw1d_dist(a, b, window=band, ub=np.inf, method=method), expected1)
                self.assertAlmostEqual(dtw2d_dist(a2, b2, window=band, method=method), expected2,
                                       msg=(n, m, band, method))

    def test_pruning_is_exact_for_many_random_pairs(self):
        # Random walks are the typical DTW workload; exercise the pruning logic.
        rng = np.random.default_rng(12)
        for _ in range(300):
            n, m = rng.integers(1, 40, size=2)
            a, b = np.cumsum(rng.normal(size=n)), np.cumsum(rng.normal(size=m))
            expected = dtw1d(a, b)[1]
            band = int(rng.integers(0, 40))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                expected_band = dtw1d(a, b, window=band)[1]
                for method in self.METHODS:
                    self.assertAlmostEqual(dtw1d_dist(a, b, method=method), expected, msg=(n, m, method))
                    self.assertAlmostEqual(dtw1d_dist(a, b, window=band, method=method), expected_band,
                                           msg=(n, m, band, method))

    def test_explicit_upper_bound(self):
        a = np.array([0, 0, 1, 1, 2, 4, 2, 1, 2, 0], dtype=np.float64)
        b = np.array([1, 1, 1, 2, 2, 2, 2, 3, 2, 0], dtype=np.float64)
        a2 = np.random.default_rng(13).random((30, 4))
        b2 = a2 + 0.01
        d = dtw2d(a2, b2)[1]
        for method in self.METHODS:
            self.assertEqual(dtw1d_dist(a, b, method=method), 4.0)
            self.assertEqual(dtw1d_dist(a, b, ub=4.0, method=method), 4.0)
            self.assertEqual(dtw1d_dist(a, b, ub=3.9, method=method), np.inf)
            self.assertEqual(dtw1d_dist(a, b, ub=0.0, method=method), np.inf)
            self.assertEqual(dtw1d_dist(a, a, ub=0.0, method=method), 0.0)
            self.assertEqual(dtw2d_dist(a2, b2, ub=d, method=method), d)
            self.assertEqual(dtw2d_dist(a2, b2, ub=d * 0.5, method=method), np.inf)

    def test_exact_upper_bound_from_previous_result(self):
        # ub equal to the true cost must never prune the optimum (rounding).
        rng = np.random.default_rng(14)
        for _ in range(100):
            n, m = rng.integers(1, 60, size=2)
            a, b = np.cumsum(rng.normal(size=n)), np.cumsum(rng.normal(size=m))
            d = dtw1d(a, b)[1]
            for method in self.METHODS:
                self.assertEqual(dtw1d_dist(a, b, ub=d, method=method), d, msg=(n, m, method))

    def test_invalid_input(self):
        with self.assertRaises(ValueError):
            dtw1d_dist(np.zeros(0), np.zeros(3))
        with self.assertRaises(ValueError):
            dtw2d_dist(np.zeros((3, 2)), np.zeros((3, 3)))
        with self.assertRaises(ValueError):
            dtw2d_dist(np.zeros((3, 2)), np.zeros((3, 2)), metric="cosine")
        with self.assertRaises(ValueError):
            dtw1d_dist(np.zeros(3), np.zeros(3), method="fast")


class InputHandlingTest(TestCase):
    def test_read_only_arrays(self):
        rng = np.random.default_rng(20)
        a, b = rng.random(30), rng.random(40)
        ra, rb = a.copy(), b.copy()
        ra.flags.writeable = False
        rb.flags.writeable = False
        self.assertEqual(dtw1d_dist(ra, rb), dtw1d_dist(a, b))
        self.assertEqual(dtw1d(ra, rb)[1:], dtw1d(a, b)[1:])
        a2, b2 = rng.random((30, 3)), rng.random((40, 3))
        ra2, rb2 = a2.copy(), b2.copy()
        ra2.flags.writeable = False
        rb2.flags.writeable = False
        self.assertEqual(dtw2d_dist(ra2, rb2), dtw2d_dist(a2, b2))
        self.assertEqual(pruned_dtw2d(ra2, rb2)[1:], dtw2d(a2, b2)[1:])

    def test_pandas(self):
        pd = pytest.importorskip("pandas")
        rng = np.random.default_rng(21)
        a, b = rng.random(30), rng.random(40)
        sa, sb = pd.Series(a), pd.Series(b)
        self.assertEqual(dtw1d_dist(sa, sb), dtw1d_dist(a, b))
        self.assertEqual(dtw1d(sa, sb)[1:], dtw1d(a, b)[1:])
        self.assertEqual(dtw1d(sa[5:], sb)[1:], dtw1d(a[5:], b)[1:])
        df = pd.DataFrame({"x": a, "y": a * 2})
        self.assertEqual(dtw1d_dist(df["x"], sb), dtw1d_dist(a, b))
        self.assertEqual(dtw2d_dist(df, pd.DataFrame({"x": b, "y": b * 2})),
                         dtw2d_dist(np.c_[a, a * 2], np.c_[b, b * 2]))
        self.assertEqual(dtw1d_dist(pd.Series(a, dtype="Float64"), sb), dtw1d_dist(a, b))
        # inputs are not modified
        self.assertTrue(np.array_equal(sa.to_numpy(), a))

    def test_nan_raises(self):
        a = np.array([0.0, 1.0, np.nan])
        b = np.array([0.0, 1.0, 2.0])
        for f in (dtw1d, dtw1d_dist, pruned_dtw1d):
            with self.assertRaises(ValueError):
                f(a, b)
            with self.assertRaises(ValueError):
                f(b, a)
        with self.assertRaises(ValueError):
            dtw2d(np.c_[a, b], np.c_[b, b])
        with self.assertRaises(ValueError):
            dtw2d_dist(np.c_[b, b], np.c_[a, b])


class NoCostMatrixTest(TestCase):
    def test_same_alignment_as_matrix_version(self):
        rng = np.random.default_rng(30)
        for n, m in [(1, 1), (1, 8), (8, 1), (10, 10), (20, 35), (35, 20), (120, 100)]:
            a, b = np.cumsum(rng.normal(size=n)), np.cumsum(rng.normal(size=m))
            a2, b2 = np.cumsum(rng.normal(size=(n, 3)), axis=0), np.cumsum(rng.normal(size=(m, 3)), axis=0)
            band = int(rng.integers(0, 40))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for f, x, y in ((dtw1d, a, b), (pruned_dtw1d, a, b), (dtw2d, a2, b2), (pruned_dtw2d, a2, b2)):
                    for kw in ({}, {"window": band}):
                        ref = f(x, y, **kw)
                        got = f(x, y, return_cost_matrix=False, **kw)
                        self.assertIsNone(got[0], (f.__name__, n, m, kw))
                        self.assertEqual(got[1:], ref[1:], (f.__name__, n, m, kw))
                        self.assertIsInstance(ref[0], np.ndarray)

    def test_upper_bound(self):
        a = np.array([0, 0, 1, 1, 2, 4, 2, 1, 2, 0], dtype=np.float64)
        b = np.array([1, 1, 1, 2, 2, 2, 2, 3, 2, 0], dtype=np.float64)
        ref = dtw1d(a, b)
        got = dtw1d(a, b, ub=4.0, return_cost_matrix=False)
        self.assertEqual(got[1:], ref[1:])
        got = dtw1d(a, b, ub=3.9, return_cost_matrix=False)
        self.assertEqual(got, (None, np.inf, [], []))
        rng = np.random.default_rng(31)
        for _ in range(50):
            n, m = rng.integers(1, 50, size=2)
            x, y = np.cumsum(rng.normal(size=n)), np.cumsum(rng.normal(size=m))
            ref = dtw1d(x, y)
            self.assertEqual(dtw1d(x, y, ub=ref[1], return_cost_matrix=False)[1:], ref[1:], (n, m))
