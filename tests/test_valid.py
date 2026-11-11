"""Valid / invalid thinking split (tok_exam.valid)."""
import unittest

from tok_exam.policies import Outcome
from tok_exam.valid import spend_split, sufficient_budget, trace_split

GRID = [0, 128, 256, 512]


class Table:
    budgets = GRID

    def __init__(self, correct, total):
        self.correct, self.total = correct, total

    def outcome(self, q, b, s):
        return Outcome(self.correct[b], min(b, self.total), 0, None)


class TestValid(unittest.TestCase):
    def test_sufficient_budget_needs_stability(self):
        self.assertEqual(sufficient_budget({0: 0, 128: 1, 256: 1, 512: 1}, GRID), 128)
        self.assertEqual(sufficient_budget({0: 1, 128: 0, 256: 1, 512: 1}, GRID), 256)
        self.assertIsNone(sufficient_budget({0: 1, 128: 1, 256: 1, 512: 0}, GRID))

    def test_trace_split(self):
        sp = trace_split(Table({0: 0, 128: 1, 256: 1, 512: 1}, 400), "q", 0)
        self.assertEqual((sp["valid"], sp["invalid"], sp["total"]), (128, 272, 400))
        sp = trace_split(Table({0: 0, 128: 1, 256: 0, 512: 0}, 400), "q", 0)
        self.assertEqual((sp["valid"], sp["invalid"], sp["unstable"]), (0, 400, True))

    def test_spend_split(self):
        sp = {"s_star": 256}
        self.assertEqual(spend_split(sp, 512, 400),
                         {"useful": 256, "overspend": 144, "hopeless": 0, "shortfall": 0})
        self.assertEqual(spend_split(sp, 128, 128)["shortfall"], 128)
        self.assertEqual(spend_split({"s_star": None}, 512, 512)["hopeless"], 512)

    def test_policy_waste_keeps_check_tokens_apart(self):
        from tok_exam.valid import policy_waste

        class Ctx:
            table = type("T", (), {"n_samples": 1})()

        import tok_exam.valid as v
        orig = v.run_policy
        v.run_policy = lambda name, ctx, paper, B, s: [
            {"id": "q", "cap": 512, "thinking": 400 + 100, "check": 100}]
        try:
            paper = {"questions": [{"id": "q", "tier": "easy"}]}
            w = policy_waste(Ctx(), [paper], [1024], ["P4_triage"], {("q", 0): {"s_star": 256}})
        finally:
            v.run_policy = orig
        r = w["P4_triage"][1024]
        self.assertEqual((r["useful"], r["overspend"], r["check"]), (256, 144, 100))


if __name__ == "__main__":
    unittest.main()
