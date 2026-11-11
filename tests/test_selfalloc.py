"""P5 prompt and answer parsing (planning.md, step 11).

Run:  PYTHONPATH=vendor python -m unittest discover -s tests -v
"""
import unittest

from tok_exam.config import load_config
from tok_exam.selfalloc import paper_message, parse_answers, score_paper

PAPER = {"paper_id": "test-000", "questions": [
    {"id": "a", "tier": "easy", "marks": 1, "gold": "18"},
    {"id": "b", "tier": "moderate", "marks": 2, "gold": r"\frac{1}{2}"},
    {"id": "c", "tier": "hard", "marks": 4, "gold": "70"}]}
POOL = {"a": {"question": "Q about eggs "}, "b": {"question": "Half?"}, "c": {"question": "Bases"}}


class TestSelfAlloc(unittest.TestCase):
    def test_message(self):
        msg = paper_message(load_config(), PAPER, POOL, 4096)
        self.assertIn("4096 tokens", msg)
        self.assertIn("7 marks", msg)
        self.assertIn("Q1 (1 mark). Q about eggs", msg)
        self.assertIn("Q3 (4 marks). Bases", msg)
        self.assertIn('"Q<k>: \\boxed{answer}" for k = 1 to 3', msg)

    def test_parse(self):
        text = ("Here are my answers.\nQ1: \\boxed{18}\n**Q2:** I think \\boxed{0.4} no, "
                "\\boxed{\\frac{1}{2}}\nQ3. skipped\n")
        self.assertEqual(parse_answers(text, 3), {1: "18", 2: r"\frac{1}{2}"})
        self.assertEqual(parse_answers(None, 3), {})
        self.assertEqual(parse_answers("Q4: \\boxed{1}", 3), {})

    def test_slice_and_pending(self):
        from tok_exam.selfalloc import paper_slice, pending_jobs

        papers = [{"paper_id": f"p{i}"} for i in range(10)]
        self.assertEqual([p["paper_id"] for p in paper_slice(papers, "3:6")], ["p3", "p4", "p5"])
        self.assertEqual(len(paper_slice(papers, ":4")), 4)
        self.assertEqual(len(paper_slice(papers, "8:")), 2)
        self.assertIs(paper_slice(papers, None), papers)
        done = {("p0", 2048, 0), ("p0", 2048, 1), ("p1", 4096, 0)}
        todo = pending_jobs(papers[:2], [2048, 4096], 2, done)
        self.assertEqual(len(todo), 8 - 3)
        self.assertNotIn(("p0", 2048, 1), {(p["paper_id"], B, i) for p, B, i in todo})

    def test_paired_comparison(self):
        from tok_exam.selfalloc import paired_comparison

        p5 = [{"paper_id": p, "budget": B, "sample_idx": s, "score": sc, "thinking": 100, "cut": False,
               "per_question": [{"pred": "1"}, {"pred": None}]}
              for p, sc in (("a", 60.0), ("b", 40.0)) for B in (2048,) for s in (0, 1)]
        runs = [{"policy": "P0", "paper_id": "a", "budget": 2048, "score": 50.0},
                {"policy": "P0", "paper_id": "b", "budget": 2048, "score": 50.0},
                {"policy": "P9", "paper_id": "a", "budget": 2048, "score": 50.0}]   # b missing
        c = paired_comparison(p5, runs, ["P0", "P9"], n_boot=50)
        self.assertEqual(c["papers"], 2)
        self.assertAlmostEqual(c["p5"]["2048"], 50.0)
        self.assertAlmostEqual(c["p5_parsed"]["2048"], 0.5)
        self.assertAlmostEqual(c["policies"]["P0"]["2048"]["diff"], 0.0)
        self.assertEqual(c["policies"]["P0"]["2048"]["p5_wins"], 1)
        self.assertNotIn("P9", c["policies"])

    def test_score(self):
        got, per_q = score_paper(PAPER, {1: "18", 2: "0.5"})
        self.assertEqual(got, 3)
        self.assertEqual([q["correct"] for q in per_q], [True, True, False])


if __name__ == "__main__":
    unittest.main()
