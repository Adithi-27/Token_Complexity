"""CPU parts of tok_exam.icl: labels, exemplars, prompts, parsing, fitting caps."""
import unittest

import numpy as np

from tok_exam.icl import (Exemplars, bin_names, curve_from_probs, fit_caps, labels_from_traces,
                          paper_exemplars, parse_allocation, question_message)

GRID = [0, 128, 256, 512, 1024]
CFG = {"icl": {"question_header": "H"}}


def item(i, split="train", tier="easy"):
    return {"id": f"q{i}", "split": split, "tier": tier, "question": f"question {i}"}


class TestLabels(unittest.TestCase):
    def test_second_shortest_correct_trace(self):
        rows = [{"id": "a", "thinking_tokens": t, "correct": c}
                for t, c in [(100, True), (300, True), (50, False), (900, True)]]
        self.assertEqual(labels_from_traces(rows, GRID, 0.5)["a"], 3)   # 300 -> 512
        rows = [{"id": "b", "thinking_tokens": 100, "correct": i == 0} for i in range(4)]
        self.assertEqual(labels_from_traces(rows, GRID, 0.5)["b"], len(GRID))

    def test_label_rounds_up_to_grid(self):
        rows = [{"id": "a", "thinking_tokens": t, "correct": True} for t in (10, 129, 129, 129)]
        self.assertEqual(labels_from_traces(rows, GRID, 0.5)["a"], 2)   # 129 -> 256


class TestExemplars(unittest.TestCase):
    def setUp(self):
        self.pool = [item(i) for i in range(6)] + [item(9, "test")]
        self.labels = {it["id"]: i % 3 for i, it in enumerate(self.pool)}
        emb = np.eye(7)
        emb[6, 2] = 0.9      # the test question is closest to q2
        self.emb = emb

    def test_never_self_never_test(self):
        for sel in ("random", "retrieved"):
            ex = Exemplars(self.pool, self.labels, sel, 0, self.emb)
            for it in self.pool:
                picked = ex.pick(it, 4)
                self.assertEqual(len(picked), 4)
                self.assertNotIn(it["id"], [p["id"] for p in picked])
                self.assertTrue(all(p["split"] == "train" for p in picked))

    def test_retrieved_puts_nearest_last(self):
        ex = Exemplars(self.pool, self.labels, "retrieved", 0, self.emb)
        self.assertEqual(ex.pick(self.pool[-1], 3)[-1]["id"], "q2")

    def test_random_is_seeded(self):
        a = Exemplars(self.pool, self.labels, "random", 7).pick(self.pool[0], 3)
        b = Exemplars(self.pool, self.labels, "random", 7).pick(self.pool[0], 3)
        self.assertEqual(a, b)

    def test_prompt_lists_labels(self):
        ex = Exemplars(self.pool, self.labels, "random", 0)
        msg = question_message(CFG, self.pool[-1], ex.pick(self.pool[-1], 2), self.labels, GRID)
        self.assertTrue(msg.endswith("Thinking tokens needed:"))
        self.assertIn("F) never", msg)

    def test_paper_exemplars_distinct(self):
        ex = Exemplars(self.pool, self.labels, "retrieved", 0, self.emb)
        pool = {it["id"]: it for it in self.pool}
        paper = {"questions": [{"id": "q9"}, {"id": "q0"}]}
        got = paper_exemplars(ex, paper, pool, 3)
        self.assertEqual(len({g["id"] for g in got}), 3)


class TestParsing(unittest.TestCase):
    def test_parse(self):
        self.assertEqual(parse_allocation("Q1: 128\nQ2: 1,024\nQ3 - 0\nQ9: 5", 3),
                         {1: 128, 2: 1024, 3: 0})
        self.assertEqual(parse_allocation(None, 3), {})

    def test_fit_caps_respects_budget(self):
        paper = {"questions": [{"id": c} for c in "abc"]}
        for alloc in ({1: 1024, 2: 1024, 3: 1024}, {1: 100}, {}, {1: 99999}):
            for B in (0, 300, 1000, 5000):
                caps = fit_caps(paper, alloc, GRID, B)
                self.assertLessEqual(sum(caps.values()), B)
                self.assertTrue(all(c in GRID for c in caps.values()))

    def test_curve(self):
        self.assertEqual(curve_from_probs(np.array([0.5, 0.5, 0, 0, 0, 0])).tolist(),
                         [0.5, 1, 1, 1, 1])
        self.assertEqual(len(bin_names(GRID)), 6)


if __name__ == "__main__":
    unittest.main()
