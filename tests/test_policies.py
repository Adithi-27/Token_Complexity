"""Allocation policies on a synthetic table (planning.md, step 11).

Run:  PYTHONPATH=vendor python -m unittest discover -s tests -v
"""
import itertools
import unittest

import numpy as np

from tok_exam.policies import (PLANS, SEQUENTIAL, Context, Outcome, fill, floor_to_grid,
                               knapsack, run_policy, score)

GRID = [0, 128, 256, 512, 1024]
# question -> (tier, marks, budget from which every sample is correct; None = never)
QUESTIONS = {"e1": ("easy", 1, 0), "e2": ("easy", 1, 128), "m1": ("moderate", 2, 256),
             "m2": ("moderate", 2, 512), "h1": ("hard", 4, 1024), "h2": ("hard", 4, None)}


class FakeTable:
    budgets, n_samples = GRID, 2

    def outcome(self, q, b, s):
        start = QUESTIONS[q][2]
        ok = start is not None and b >= start
        # a correct trace stops thinking at `start`; others use the whole cap
        thinking = min(b, start) if ok else b
        return Outcome(ok, thinking, thinking + 5, "1" if ok else f"wrong{s}")

    def mean_acc(self, q):
        return [float(self.outcome(q, b, 0).correct) for b in GRID]


PAPER = {"paper_id": "test-000",
         "questions": [{"id": q, "tier": t, "marks": m} for q, (t, m, _) in QUESTIONS.items()]}
CFG = {"policies": {"knapsack_unit": 128, "triage_budget": 128,
                    "tier_ratio": {"easy": 1, "moderate": 2, "hard": 4}}}


def ctx(predicted=None):
    t = FakeTable()
    curves = predicted or {q: t.mean_acc(q) for q in QUESTIONS}
    return Context(CFG, t, predicted=curves, bandit=curves, icl_caps=EqualCaps())


class EqualCaps(dict):
    """Stand-in for the ICL paper allocations: an equal share of B."""

    def __missing__(self, key):
        _, B = key
        return {q: floor_to_grid(B / len(QUESTIONS), GRID) for q in QUESTIONS}


class TestHelpers(unittest.TestCase):
    def test_floor(self):
        self.assertEqual(floor_to_grid(300, GRID), 256)
        self.assertEqual(floor_to_grid(50, GRID), 0)

    def test_fill_stays_within_budget(self):
        # 256 + 512 = 768; the 232 left cannot buy the next step of either
        self.assertEqual(fill({"a": 300, "b": 700}, GRID, 1000), {"a": 256, "b": 512})
        # 0 + 0 with 400 left: both move up to 128; 256 is more than twice either target
        self.assertEqual(fill({"a": 100, "b": 120}, GRID, 400), {"a": 128, "b": 128})

    def test_knapsack_matches_brute_force(self):
        rng = np.random.default_rng(1)
        for _ in range(20):
            values = {q: np.sort(rng.random(len(GRID))).tolist() for q in "abc"}
            B = int(rng.integers(0, 2500))
            caps = knapsack(values, GRID, B, 128)
            got = sum(values[q][GRID.index(c)] for q, c in caps.items())
            best = max(sum(values[q][j] for q, j in zip("abc", js))
                       for js in itertools.product(range(len(GRID)), repeat=3)
                       if sum(GRID[j] for j in js) <= B)
            self.assertLessEqual(sum(caps.values()), B)
            self.assertAlmostEqual(got, best)


class TestPolicies(unittest.TestCase):
    def test_every_policy_runs_and_respects_budget(self):
        c = ctx()
        for name in list(PLANS) + list(SEQUENTIAL):
            for B in (0, 512, 2048, 8192):
                with self.subTest(name=name, B=B):
                    res = run_policy(name, c, PAPER, B, 0)
                    self.assertEqual(len(res), len(QUESTIONS))
                    # every policy, P4 with its check attempts included, stays within B
                    self.assertLessEqual(score(res)["thinking"], B)

    def test_oracle_spends_where_it_pays(self):
        # 1920 tokens: e1 (0) e2 (128) m1 (256) m2 (512) h1 (1024) = 1920 -> all but h2
        caps = PLANS["oracle"](ctx(), PAPER, 1920)
        self.assertEqual(caps["h2"], 0)
        s = score(run_policy("oracle", ctx(), PAPER, 1920, 0))
        self.assertEqual(s["marks"], 1 + 1 + 2 + 2 + 4)

    def test_oracle_beats_equal_share(self):
        for B in (512, 1024, 2048):
            o = score(run_policy("oracle", ctx(), PAPER, B, 0))["marks"]
            e = score(run_policy("P0_equal", ctx(), PAPER, B, 0))["marks"]
            self.assertGreaterEqual(o, e)

    def test_hindsight_oracle_bounds_every_policy(self):
        for B in (0, 256, 1024, 1920, 4096):
            h = score(run_policy("oracle_hindsight", ctx(), PAPER, B, 0))
            self.assertLessEqual(h["thinking"], B)
            for name in list(PLANS) + ["P1_in_order"]:
                with self.subTest(name=name, B=B):
                    o = score(run_policy(name, ctx(predicted={q: FakeTable().mean_acc(q)
                                                              for q in QUESTIONS}),
                                         PAPER, B, 0))
                    self.assertGreaterEqual(h["marks"], o["marks"])

    def test_shadow_price_fits_and_is_near_knapsack(self):
        from tok_exam.policies import shadow_price
        values = {q: [m * p for p in FakeTable().mean_acc(q)] for q, (_, m, _) in QUESTIONS.items()}
        for B in (0, 128, 512, 1000, 1920, 4096):
            caps = shadow_price(values, GRID, B)
            self.assertLessEqual(sum(caps.values()), B)
            got = sum(values[q][GRID.index(b)] for q, b in caps.items())
            best = knapsack(values, GRID, B, 128)
            opt = sum(values[q][GRID.index(b)] for q, b in best.items())
            self.assertLessEqual(got, opt + 1e-9)
        # flat curves: identical questions with value only at the top budget still get it
        flat = {f"x{i}": [0, 0, 0, 0, 1] for i in range(3)}
        self.assertEqual(sorted(shadow_price(flat, GRID, 2100).values()), [0, 1024, 1024])
        # 1920 is exactly enough for every solvable question; h2 is never worth a token
        caps = shadow_price(values, GRID, 1920)
        self.assertEqual(caps["h2"], 0)
        self.assertEqual(caps["h1"], 1024)

    def test_in_order_runs_out(self):
        res = run_policy("P1_in_order", ctx(), PAPER, 1024, 0)
        self.assertLessEqual(score(res)["thinking"], 1024)

    def test_rollover_passes_savings_on(self):
        # equal share of 1920 over 6 questions is 256 each; e1 and e2 use 0 and 128,
        # so 384 roll over and let a later question reach a higher cap
        plain = {r["id"]: r for r in run_policy("P0_equal", ctx(), PAPER, 1920, 0)}
        roll = {r["id"]: r for r in run_policy("P0_equal+roll", ctx(), PAPER, 1920, 0)}
        self.assertLessEqual(score(list(roll.values()))["thinking"], 1920)
        self.assertGreater(max(r["cap"] for r in roll.values()), 256)
        self.assertGreaterEqual(score(list(roll.values()))["marks"],
                                score(list(plain.values()))["marks"])

    def test_every_rollover_variant_respects_budget(self):
        for name in PLANS:
            for B in (0, 512, 2048, 8192):
                with self.subTest(name=name, B=B):
                    res = run_policy(name + "+roll", ctx(), PAPER, B, 1)
                    self.assertLessEqual(score(res)["thinking"], B)

    def test_triage_keeps_agreeing_answers(self):
        # e1, e2 agree at 128 ("1" == "1"); the rest disagree (wrong0 vs wrong1)
        res = {r["id"]: r for r in run_policy("P4_triage", ctx(), PAPER, 4096, 0)}
        self.assertEqual(res["e1"]["cap"], 128)
        self.assertTrue(res["e1"]["correct"] and res["e2"]["correct"])
        self.assertTrue(res["m2"]["correct"])

    def test_triage_closed_rejects_answers_cut_at_the_cap(self):
        # h1 is cut at every cap below 1024 and both samples force the same wrong answer:
        # P4 trusts the agreement and freezes h1; the closed variant sends it to pass 2
        class AgreeWrong(FakeTable):
            def outcome(self, q, b, s):
                o = super().outcome(q, b, s)
                return o._replace(pred="x") if q == "h1" and not o.correct else o

        c = Context(CFG, AgreeWrong(), predicted={q: FakeTable().mean_acc(q) for q in QUESTIONS})
        plain = {r["id"]: r for r in run_policy("P4_triage", c, PAPER, 4096, 0)}
        closed = {r["id"]: r for r in run_policy("P4_triage_closed", c, PAPER, 4096, 0)}
        self.assertFalse(plain["h1"]["correct"])
        self.assertTrue(closed["h1"]["correct"])
        self.assertLessEqual(score(list(closed.values()))["thinking"], 4096)

    def test_equal_share_spends_its_budget(self):
        # 6 questions, B = 900: B/n = 150 floors to 128 (768 planned); the 132 left buys
        # one step 128 -> 256, so P0 never leaves a whole affordable grid step unused
        caps = PLANS["P0_equal"](ctx(), PAPER, 900)
        self.assertEqual(sorted(caps.values()), [128] * 5 + [256])
        caps = PLANS["P0_equal"](ctx(), PAPER, 100)     # B/n = 16: all floors to 0 ...
        self.assertEqual(sum(caps.values()), 0)          # ... and 128 > 2 * 16, so no step up
        caps = PLANS["P0_equal"](ctx(), PAPER, 1200)    # B/n = 200: 432 left buys 3 steps
        self.assertEqual(sorted(caps.values()), [128, 128, 128, 256, 256, 256])
        # ties go by printed order, not by tier
        from tok_exam.policies import paper_order
        first = [q["id"] for q in paper_order(PAPER)][:3]
        self.assertEqual({q for q, c in caps.items() if c == 256}, set(first))

    def test_triage_charges_predictor_cost(self):
        c = ctx()
        c.predictor_cost = 100.0
        for name in ("P4_triage", "P4_triage_closed"):
            for B in (256, 1536, 4096):
                res = run_policy(name, c, PAPER, B, 0)
                self.assertLessEqual(score(res)["thinking"] + 100 * len(QUESTIONS), max(B, 600))
        # a predictor that costs more than B leaves nothing to think with
        self.assertEqual(score(run_policy("P4_triage", c, PAPER, 300, 0))["thinking"], 0)

    def test_triage_reports_check_tokens(self):
        res = run_policy("P4_triage", ctx(), PAPER, 4096, 0)
        self.assertTrue(all(0 <= r["check"] <= r["thinking"] for r in res))
        self.assertGreater(sum(r["check"] for r in res), 0)
        self.assertTrue(all(r["check"] == 0 for r in run_policy("P3_knapsack", ctx(), PAPER, 4096, 0)))

if __name__ == "__main__":
    unittest.main()
