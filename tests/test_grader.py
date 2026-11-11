"""Hand-written outputs for the grader and the row format (planning.md, steps 4-6).

Run:  PYTHONPATH=vendor python -m unittest discover -s tests -v
"""
import unittest

from tok_exam.config import load_config
from tok_exam.grader import grade, is_equiv, last_boxed, normalise_gsm8k_gold
from tok_exam.papers import sample_papers, total_marks
from tok_exam.prompt import build_user_message
from tok_exam.records import FIELDS, make_row, split_thinking

# (name, answer text, gold, expected correct)
CASES = [
    ("integer", r"So the answer is $\boxed{18}$.", "18", True),
    ("wrong integer", r"\boxed{17}", "18", False),
    ("no box", "Therefore the answer is 18.", "18", False),
    ("several boxes, last wins", r"First \boxed{12}, correcting: \boxed{18}", "18", True),
    ("dollar sign", r"\boxed{\$18}", "18", True),
    ("unit text", r"\boxed{18 \text{ dollars}}", "18", True),
    ("thousands separator", r"\boxed{1,024}", "1024", True),
    ("trailing zeros", r"\boxed{18.00}", "18", True),
    ("percent", r"\boxed{25\%}", "25", True),
    ("fraction vs decimal", r"\boxed{0.75}", r"\frac{3}{4}", True),
    ("dfrac", r"\boxed{\dfrac{14}{3}}", r"\frac{14}{3}", True),
    ("tuple with left/right", r"\boxed{(3, \frac{\pi}{2})}", r"\left( 3, \frac{\pi}{2} \right)", True),
    ("tuple wrong", r"\boxed{(3, \pi)}", r"\left( 3, \frac{\pi}{2} \right)", False),
    ("radical", r"\boxed{3\sqrt{13}}", r"3\sqrt{13}", True),
    ("complex", r"\boxed{-2+7i}", "-2 + 7i", True),
    ("polynomial reordered", r"\boxed{-1 + x - x^2 + x^3 - x^4 + x^5}",
     "x^5 - x^4 + x^3 - x^2 + x - 1", True),
    ("text answer", r"\boxed{\text{Evelyn}}", r"\text{Evelyn}", True),
    ("text answer wrong", r"\boxed{\text{Bob}}", r"\text{Evelyn}", False),
    ("list", r"\boxed{3,5,7}", "3, 5, 7", True),
    ("degrees", r"\boxed{40^\circ}", "40", True),
    ("empty box", r"The answer goes in \boxed{}", "18", False),
    ("unclosed last box falls back", r"\boxed{18} and then \boxed{20", "18", True),
]


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        return text.split()


class TestGrader(unittest.TestCase):
    def test_cases(self):
        for name, text, gold, expected in CASES:
            with self.subTest(name):
                self.assertEqual(grade(text, gold)[1], expected)

    def test_last_boxed(self):
        self.assertEqual(last_boxed(r"\boxed{\frac{1}{2}}"), r"\frac{1}{2}")
        self.assertIsNone(last_boxed(None))
        self.assertFalse(is_equiv(None, "1"))

    def test_gsm8k_gold(self):
        self.assertEqual(normalise_gsm8k_gold("blah\n#### 1,234"), "1234")
        self.assertEqual(normalise_gsm8k_gold("#### -5"), "-5")
        with self.assertRaises(ValueError):
            normalise_gsm8k_gold("#### 3.5")


class TestRecords(unittest.TestCase):
    item = {"id": "gsm8k-1", "dataset": "gsm8k", "tier": "easy", "split": "test",
            "question": "q", "gold": "18"}

    def test_split(self):
        self.assertEqual(split_thinking("a b c</think>\n\\boxed{1}"), ("a b c", "\n\\boxed{1}"))
        self.assertEqual(split_thinking("<think>\na b</think>x"), ("\na b", "x"))
        self.assertEqual(split_thinking("never closes"), ("never closes", None))

    def test_row(self):
        row = make_row(self.item, 3, "P", "one two three</think>\n\\boxed{18}", FakeTokenizer())
        self.assertEqual(tuple(row), FIELDS)
        self.assertEqual((row["thinking_tokens"], row["pred"], row["correct"]), (3, "18", True))

    def test_box_inside_thinking_does_not_count(self):
        row = make_row(self.item, 0, "P", "I think \\boxed{18} but ran out", FakeTokenizer())
        self.assertEqual((row["pred"], row["correct"], row["answer_tokens"]), (None, False, 0))


class TestPromptAndPapers(unittest.TestCase):
    def test_prompt(self):
        cfg = load_config()
        self.assertEqual(build_user_message(cfg, " Q "),
                         "Q\nPlease reason step by step, and put your final answer within \\boxed{}.")

    def test_papers(self):
        import random
        cfg = load_config()
        pc = cfg["papers"]
        pool = [{"id": f"{t}-{s}-{i}", "tier": t, "split": s}
                for t in ("easy", "moderate", "hard") for s in ("train", "test") for i in range(12)]
        papers = sample_papers(pool, "test", 5, pc["composition"], pc["marks"], random.Random(0))
        for p in papers:
            ids = [q["id"] for q in p["questions"]]
            self.assertEqual(len(ids), len(set(ids)))
            self.assertTrue(all("-test-" in i for i in ids))
            self.assertEqual(total_marks(p), 38)


if __name__ == "__main__":
    unittest.main()
