"""GPU features for the difficulty predictors (planning.md, step 10 c-d).

self:  the model rates the question 1-5 with its thinking skipped. One greedy
       token; the feature is the expected rating under the probabilities of the
       digit tokens 1-5. Writes <probe_dir>/self.jsonl: id, rating, cost_tokens.
probe: the model starts its normal answer and thinks for at most probe_tokens.
       Features: mean entropy of the top-20 next-token distribution, mean
       negative log-probability of the sampled tokens, and whether it already
       closed its thinking. Writes <probe_dir>/probe.jsonl:
       id, entropy, neg_logprob, closed, cost_tokens.

cost_tokens is the generated tokens the feature costs, charged to the exam budget.

Run:  python -m tok_exam.probe [--only self|probe]
"""
import argparse
import math

from .config import load_config, resolve
from .llm import load_llm
from .prompt import build_prompt
from .records import THINK_CLOSE, read_jsonl, write_jsonl

DIGITS = "12345"


def skip_thinking(prompt):
    """Close the <think> block the chat template opened, so the reply is immediate."""
    return prompt + ("\n</think>\n\n" if prompt.rstrip().endswith("<think>") else "")


def top_entropy(logprobs):
    """Entropy of one position's top-k distribution, renormalised over the top k."""
    ps = [math.exp(lp.logprob) for lp in logprobs.values()]
    z = sum(ps)
    return -sum(p / z * math.log(p / z) for p in ps if p > 0)


def self_ratings(cfg, llm, items):
    from vllm import SamplingParams

    tok = llm.get_tokenizer()
    digit_ids = {d: tok.encode(d, add_special_tokens=False) for d in DIGITS}
    if any(len(v) != 1 for v in digit_ids.values()):
        raise ValueError(f"digits are not single tokens: {digit_ids}")
    digit_ids = {d: v[0] for d, v in digit_ids.items()}
    template = cfg["difficulty"]["self_rating_prompt"]
    prompts = [skip_thinking(tok.apply_chat_template(
        [{"role": "user", "content": template.format(question=it["question"].strip())}],
        tokenize=False, add_generation_prompt=True)) for it in items]
    outs = llm.generate(prompts, SamplingParams(temperature=0, max_tokens=1, logprobs=20))
    rows = []
    for it, out in zip(items, outs):
        top = out.outputs[0].logprobs[0]
        mass = {d: math.exp(top[i].logprob) if i in top else 0.0 for d, i in digit_ids.items()}
        z = sum(mass.values())
        rating = sum(int(d) * m for d, m in mass.items()) / z if z else 3.0
        rows.append({"id": it["id"], "rating": rating, "digit_mass": z, "cost_tokens": 1})
    return rows


def probes(cfg, llm, items):
    from vllm import SamplingParams

    s, n = cfg["sampling"], cfg["difficulty"]["probe_tokens"]
    tok = llm.get_tokenizer()
    params = SamplingParams(temperature=s["temperature"], top_p=s["top_p"], max_tokens=n,
                            seed=s["base_seed"], logprobs=20, skip_special_tokens=False)
    outs = llm.generate([build_prompt(cfg, tok, it["question"]) for it in items], params)
    rows = []
    for it, out in zip(items, outs):
        o = out.outputs[0]
        steps = list(zip(o.token_ids, o.logprobs))
        rows.append({"id": it["id"],
                     "entropy": sum(top_entropy(lp) for _, lp in steps) / len(steps),
                     "neg_logprob": -sum(lp[t].logprob for t, lp in steps) / len(steps),
                     "closed": THINK_CLOSE in o.text, "cost_tokens": len(o.token_ids)})
    return rows


def main():
    cfg = load_config()
    p = argparse.ArgumentParser()
    p.add_argument("--only", choices=["self", "probe"], default=None)
    args = p.parse_args()
    items = read_jsonl(resolve(cfg["data"]["processed_dir"]) / "pool.jsonl")
    out_dir = resolve(cfg["difficulty"]["probe_dir"])
    llm = load_llm(cfg)
    if args.only in (None, "self"):
        rows = self_ratings(cfg, llm, items)
        write_jsonl(out_dir / "self.jsonl", rows)
        low = sum(r["digit_mass"] < 0.5 for r in rows)
        print(f"self: {len(rows)} ratings, {low} with digit mass < 0.5", flush=True)
    if args.only in (None, "probe"):
        rows = probes(cfg, llm, items)
        write_jsonl(out_dir / "probe.jsonl", rows)
        print(f"probe: {len(rows)} probes, {sum(r['closed'] for r in rows)} closed within "
              f"{cfg['difficulty']['probe_tokens']} tokens", flush=True)


if __name__ == "__main__":
    main()
