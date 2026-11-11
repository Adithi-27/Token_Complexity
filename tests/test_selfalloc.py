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

    def test_score(self):
        got, per_q = score_paper(PAPER, {1: "18", 2: "0.5"})
        self.assertEqual(got, 3)
        self.assertEqual([q["correct"] for q in per_q], [True, True, False])


if __name__ == "__main__":
    unittest.main()
