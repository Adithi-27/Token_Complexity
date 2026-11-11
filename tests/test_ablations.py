"""Ablation helpers (tok_exam.ablations) and the label-level kNN control."""
import unittest

import numpy as np

from tok_exam.ablations import variant_papers, with_costs
from tok_exam.difficulty import bin_metrics, knn_majority_bins

POOL = [{"id": f"{t}{i}", "tier": t, "split": s}
        for t in ("easy", "moderate", "hard") for i in range(30) for s in ("test",)]
BASE = {"composition": {"easy": 10, "moderate": 6, "hard": 4}, "marks": {"easy": 1, "moderate": 2, "hard": 4}}


class TestAblations(unittest.TestCase):
    def test_variant_composition_marks_and_exclude(self):
        spec = {"composition": {"hard": 10}, "marks": {"hard": 1}, "exclude": ["easy0", "easy1"]}
        papers = variant_papers(POOL, spec, BASE, 20, 1)
        self.assertEqual(len(papers), 20)
        for p in papers:
            tiers = [q["tier"] for q in p["questions"]]
            self.assertEqual((tiers.count("easy"), tiers.count("moderate"), tiers.count("hard")), (10, 6, 10))
            self.assertTrue(all(q["marks"] == 1 for q in p["questions"] if q["tier"] == "hard"))
            self.assertFalse({"easy0", "easy1"} & {q["id"] for q in p["questions"]})
        self.assertEqual(variant_papers(POOL, spec, BASE, 20, 1), papers)   # seeded

    def test_costs(self):
        preds = {"icl": ({}, 1.0), "probe": ({}, 128.0), "tier": ({}, 0.0)}
        self.assertEqual(with_costs(preds, "charged"), preds)
        self.assertTrue(all(c == 0 for _, c in with_costs(preds, "free").values()))
        p = with_costs(preds, "prompt", 900.0)
        self.assertEqual((p["icl"][1], p["probe"][1], p["tier"][1]), (901.0, 128.0, 0.0))


class TestLabelControl(unittest.TestCase):
    def test_majority_bin(self):
        # 4 train points: two near (0, 1) with bin 3, two near (1, 0) with bins 1 and 2
        E = np.array([[0, 1], [0.1, 1], [1, 0], [1, 0.1], [0.05, 1]], dtype=float)
        bins = np.array([3, 3, 1, 2, 0])
        train = np.array([True, True, True, True, False])
        got = knn_majority_bins(E, bins, train, 2)
        self.assertEqual(got[4], 3)          # both neighbours say 3
        self.assertEqual(got[2], 2)          # its only other close train point says 2; tie -> smaller
        m = bin_metrics(np.array([0, 1, 2]), np.array([0, 2, 2]))
        self.assertAlmostEqual(m["bin_acc"], 2 / 3)
        self.assertEqual(m["bin_within_1"], 1.0)


if __name__ == "__main__":
    unittest.main()
