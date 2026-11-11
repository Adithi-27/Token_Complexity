"""CPU helpers of tok_exam.realbudget."""
import unittest

from tok_exam.realbudget import budget_message, compare, matched_prefix, stratified


class TestRealBudget(unittest.TestCase):
    def test_matched_prefix(self):
        self.assertEqual(matched_prefix([1, 2, 3], [1, 2, 4]), 2)
        self.assertEqual(matched_prefix([1, 2], [1, 2, 3]), 2)
        self.assertEqual(matched_prefix([], [1]), 0)

    def test_stratified_is_seeded_and_test_only(self):
        rows = {t: [{"id": f"{t}{i}", "split": "test" if i % 2 else "train"} for i in range(10)]
                for t in ("easy", "moderate", "hard")}
        a, b = stratified(rows, 3, 7), stratified(rows, 3, 7)
        self.assertEqual(a, b)
        for t, ids in a.items():
            self.assertEqual(len(ids), 3)
            self.assertTrue(all(int(i[len(t):]) % 2 for i in ids))

    def test_budget_message(self):
        cfg = {"realbudget": {"budget_line": "At most {budget}."}}
        self.assertEqual(budget_message(cfg, " Q? ", 512), "Q?\nAt most 512.")

    def test_compare(self):
        rows = [{"id": "a", "tier": "easy", "budget": 512, "correct": True, "thinking": 300,
                 "forced": False},
                {"id": "b", "tier": "hard", "budget": 512, "correct": False, "thinking": 512,
                 "forced": True}]
        out = compare(rows, {"a": {512: 0.5}, "b": {512: 0.0}})
        self.assertEqual(out["all@512"]["prompted_acc"], 0.5)
        self.assertEqual(out["all@512"]["truncation_acc"], 0.25)
        self.assertEqual(out["easy@512"]["stopped_by_itself"], 1.0)


if __name__ == "__main__":
    unittest.main()
