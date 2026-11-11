"""Needed budget and curve predictor on synthetic data (planning.md, steps 9-10).

Run:  PYTHONPATH=vendor python -m unittest discover -s tests -v
"""
import unittest

import numpy as np

from tok_exam.difficulty import (CurvePredictor, evaluate, needed_bins, text_features,
                                 tier_features)


class TestNeeded(unittest.TestCase):
    def test_bins(self):
        acc = np.array([[0, 0.25, 0.5, 1.0],   # first reaches 0.5 at index 2
                        [1, 1, 1, 1],          # solved with no thinking
                        [0, 0, 0.25, 0.25]])   # never
        self.assertEqual(needed_bins(acc, 0.5).tolist(), [2, 0, 4])


class TestFeatures(unittest.TestCase):
    items = [{"tier": "easy", "question": "How many apples? 3 and 4"},
             {"tier": "hard", "question": r"Find all real $x$ with $\frac{x}{2}=1$. [asy]draw();[/asy]"}]

    def test_shapes(self):
        self.assertEqual(tier_features(self.items).tolist(), [[1, 0, 0], [0, 0, 1]])
        t = text_features(self.items)
        self.assertEqual(t.shape[0], 2)
        self.assertEqual(t[1, 5], 1.0)   # has [asy]
        self.assertEqual(t[0, 2], 2)     # two numbers


class TestPredictor(unittest.TestCase):
    def test_learns_tiers(self):
        rng = np.random.default_rng(0)
        tiers = rng.integers(0, 3, 300)
        X = np.eye(3)[tiers]
        # easy solved from budget 0, moderate from 2, hard from 4 (never in a 4-budget grid)
        start = np.array([0, 2, 4])[tiers]
        acc = (np.arange(4)[None, :] >= start[:, None]).astype(float)
        model = CurvePredictor().fit(X[:200], acc[:200])
        pred = model.predict(X[200:])
        self.assertTrue((np.diff(pred, axis=1) >= 0).all())
        s = evaluate(pred, acc[200:], 0.5)
        self.assertEqual(s["bin_acc"], 1.0)
        self.assertGreater(s["spearman"], 0.99)

    def test_out_of_fold_never_sees_its_own_row(self):
        from tok_exam.difficulty import out_of_fold

        rng = np.random.default_rng(1)
        X = rng.normal(size=(60, 3))
        acc = (rng.random((60, 4)) < 0.5).astype(float)
        train = np.arange(60) < 40
        a = out_of_fold(X, acc, train, 1.0, seed=3)
        self.assertEqual(a.shape, (40, 4))
        self.assertTrue(np.allclose(a, out_of_fold(X, acc, train, 1.0, seed=3)))
        # changing one train row's label must not move that row's own prediction
        acc2 = acc.copy()
        acc2[0] = 1 - acc2[0]
        self.assertTrue(np.allclose(a[0], out_of_fold(X, acc2, train, 1.0, seed=3)[0]))

    def test_constant_column(self):
        X = np.eye(3)[[0, 1, 2, 0, 1, 2]]
        acc = np.array([[0, 1]] * 6, dtype=float)  # budget 0 never right, budget 1 always
        pred = CurvePredictor().fit(X, acc).predict(X)
        self.assertTrue(np.allclose(pred, acc))


if __name__ == "__main__":
    unittest.main()
