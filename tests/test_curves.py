"""Budget sweep logic with a fake LLM and tokenizer (planning.md, step 7).

Run:  PYTHONPATH=vendor python -m unittest discover -s tests -v
"""
import sys
import types
import unittest

sys.modules.setdefault("vllm", types.SimpleNamespace(SamplingParams=lambda **kw: kw))

from tok_exam.curves import FORCE, parse_forced, sweep_rows


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        return text.split()

    def decode(self, ids):
        return " ".join(ids)


class FakeLLM:
    """Answers every forced prompt with `reply` and records the prompts."""

    def __init__(self, reply):
        self.reply, self.prompts = reply, []

    def generate(self, prompts, params):
        self.prompts += prompts
        out = types.SimpleNamespace(text=self.reply, token_ids=self.reply.split())
        return [types.SimpleNamespace(outputs=[out]) for _ in prompts]


def trace(thinking_words, closed=True, correct=True):
    thinking = " ".join(f"w{i}" for i in range(thinking_words))
    return {"id": "q", "tier": "easy", "split": "test", "sample_idx": 0, "prompt": "P<think>\n",
            "full_output": thinking + ("</think>\\boxed{18}" if closed else ""),
            "thinking_text": thinking, "answer_tokens": 2 if closed else 0,
            "pred": "18" if closed and correct else None, "gold": "18",
            "correct": closed and correct}


class TestSweep(unittest.TestCase):
    def test_parse_forced(self):
        self.assertEqual(parse_forced("18} and more"), "18")
        self.assertEqual(parse_forced(r"\frac{1}{2}}."), r"\frac{1}{2}")
        self.assertIsNone(parse_forced("18 never closed"))

    def test_short_trace_keeps_answer_long_trace_forced(self):
        llm, tok = FakeLLM("17}"), FakeTokenizer()
        res = {r["budget"]: r for r in sweep_rows(llm, tok, [trace(5)], [0, 3, 5, 8], 64)}
        self.assertTrue(res[0]["forced"] and res[3]["forced"])
        self.assertFalse(res[5]["forced"] or res[8]["forced"])
        self.assertEqual((res[5]["correct"], res[5]["tokens"]), (True, 5 + 2))
        self.assertEqual([res[b]["thinking"] for b in (0, 3, 5, 8)], [0, 3, 5, 5])
        self.assertEqual((res[3]["pred"], res[3]["correct"]), ("17", False))
        force_cost = len(FORCE.split())
        self.assertEqual(res[3]["tokens"], 3 + force_cost + 1)
        self.assertEqual(res[0]["tokens"], 0 + force_cost + 1)
        self.assertEqual(llm.prompts[0], "P<think>\n" + FORCE)
        self.assertEqual(llm.prompts[1], "P<think>\nw0 w1 w2" + FORCE)

    def test_capped_trace_is_always_forced(self):
        llm = FakeLLM("18}")
        res = sweep_rows(llm, FakeTokenizer(), [trace(4, closed=False)], [10], 64)
        self.assertTrue(res[0]["forced"])
        self.assertTrue(res[0]["correct"])
        self.assertEqual(llm.prompts[0], "P<think>\nw0 w1 w2 w3" + FORCE)


if __name__ == "__main__":
    unittest.main()
