"""Two-level bootstrap pieces (tok_exam.stats) on a synthetic table."""
import unittest

import numpy as np

from tok_exam.policies import Outcome
from tok_exam.stats import ByOrigin, ResampledTable, auc, holm, replicate

GRID = [0, 128, 256]


class Table:
    budgets, n_samples = GRID, 2

    def outcome(self, q, b, s):
        return Outcome(b >= 128 and s == 0, min(b, 100), 0, None)


CFG = {"papers": {"n_papers": 3, "composition": {"easy": 2, "moderate": 1, "hard": 1},
                  "marks": {"easy": 1, "moderate": 2, "hard": 4}}}
ITEMS = ([{"id": f"e{i}", "tier": "easy"} for i in range(4)]
         + [{"id": f"m{i}", "tier": "moderate"} for i in range(2)]
         + [{"id": f"h{i}", "tier": "hard"} for i in range(2)])


class TestStats(unittest.TestCase):
    def test_replicate_shapes_and_views(self):
        t, origin, papers = replicate(CFG, Table(), ITEMS, np.random.default_rng(0))
        self.assertEqual(len(papers), 3)
        for p in papers:
            ids = [q["id"] for q in p["questions"]]
            self.assertEqual(len(ids), len(set(ids)))
            self.assertEqual([q["tier"] for q in p["questions"]].count("easy"), 2)
        v = papers[0]["questions"][0]["id"]
        self.assertIn(origin[v], {it["id"] for it in ITEMS})
        self.assertEqual(t.outcome(v, 128, 0).correct, t.samples[v][0] == 0)

    def test_by_origin(self):
        d = ByOrigin({"a": [1, 2]}, {"a#0": "a", "a#1": "a"})
        self.assertEqual(d["a#1"], [1, 2])
        self.assertIn("a#0", d)
        self.assertNotIn("b#0", d)

    def test_holm(self):
        self.assertEqual(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])

    def test_auc_constant(self):
        self.assertAlmostEqual(auc([8192, 16384, 32768], [50, 50, 50]), 50.0)


if __name__ == "__main__":
    unittest.main()
