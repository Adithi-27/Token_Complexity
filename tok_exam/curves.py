"""Budget curves by truncating existing traces (planning.md, step 7).

For each trace and budget B: keep the first B thinking tokens, append
`</think>` and `Final answer: \\boxed{`, generate a short greedy continuation
and grade it. B = 0 means no thinking at all. A trace that closed its thinking
within B tokens keeps its original answer; a trace that hit the generation cap
never answered, so it is always forced. Nothing is regenerated from scratch.

Writes one row per (trace, budget) to <curves.dir>/{tier}.jsonl:
  id, tier, split, sample_idx, budget, forced, thinking, tokens, pred, gold, correct
`thinking` is the thinking tokens kept (what a budget limits); `tokens` adds the
answer tokens (the forced prefix counts as answer tokens).

Run:  python -m tok_exam.curves [--tiers ...] [--in-dir ...]
"""
import argparse
import collections

from .config import TIERS, load_config, resolve
from .grader import is_equiv, last_boxed
from .records import hit_cap, read_jsonl, write_jsonl

FORCE = "\n</think>\n\nFinal answer: \\boxed{"


def forced_prompt(row, thinking_prefix):
    """The original prompt, the truncated thinking, then the forced answer start."""
    return row["prompt"] + thinking_prefix + FORCE


def parse_forced(text):
    """Answer from the continuation of `\\boxed{`, matching nested braces."""
    return last_boxed("\\boxed{" + text)


def sweep_rows(llm, tokenizer, rows, budgets, max_tokens):
    from vllm import SamplingParams

    params = SamplingParams(temperature=0, max_tokens=max_tokens)
    force_cost = len(tokenizer.encode(FORCE, add_special_tokens=False))
    out, todo = [], []
    for r in rows:
        ids = tokenizer.encode(r["thinking_text"], add_special_tokens=False)
        for b in budgets:
            base = {"id": r["id"], "tier": r["tier"], "split": r["split"],
                    "sample_idx": r["sample_idx"], "budget": b}
            if len(ids) <= b and not hit_cap(r):
                out.append({**base, "forced": False, "thinking": len(ids),
                            "tokens": len(ids) + r["answer_tokens"],
                            "pred": r["pred"], "gold": r["gold"], "correct": r["correct"]})
            else:
                kept = min(b, len(ids))
                out.append({**base, "forced": True, "thinking": kept, "tokens": kept + force_cost,
                            "pred": None, "gold": r["gold"], "correct": False})
                todo.append((len(out) - 1, forced_prompt(r, tokenizer.decode(ids[:kept]))))
    gens = llm.generate([p for _, p in todo], params)
    for (i, _), g in zip(todo, gens):
        pred = parse_forced(g.outputs[0].text)
        out[i]["tokens"] += len(g.outputs[0].token_ids)
        out[i]["pred"], out[i]["correct"] = pred, is_equiv(pred, out[i]["gold"])
    return out


def load_curves(cfg, tiers=TIERS):
    """{question id: {budget: (accuracy over samples, mean tokens spent)}}."""
    acc = collections.defaultdict(lambda: collections.defaultdict(list))
    for tier in tiers:
        for r in read_jsonl(resolve(cfg["curves"]["dir"]) / f"{tier}.jsonl"):
            acc[r["id"]][r["budget"]].append((r["correct"], r["tokens"]))
    return {qid: {b: (sum(c for c, _ in v) / len(v), sum(t for _, t in v) / len(v))
                  for b, v in sorted(by_b.items())}
            for qid, by_b in acc.items()}


def main():
    cfg = load_config()
    cc = cfg["curves"]
    p = argparse.ArgumentParser()
    p.add_argument("--tiers", nargs="+", default=list(TIERS), choices=TIERS)
    p.add_argument("--in-dir", default=cfg["output"]["dir"])
    p.add_argument("--out-dir", default=cc["dir"])
    args = p.parse_args()
    in_dir, out_dir = resolve(args.in_dir), resolve(args.out_dir)

    from .llm import load_llm

    # every budget of a trace shares its prefix, so prefix caching saves most prefill
    llm = load_llm(cfg, enable_prefix_caching=True)
    tokenizer = llm.get_tokenizer()
    for tier in args.tiers:
        path = in_dir / f"{tier}.jsonl"
        if not path.exists():
            print(f"{tier}: no generations, skipped", flush=True)
            continue
        res = sweep_rows(llm, tokenizer, read_jsonl(path), cc["budgets"], cc["max_answer_tokens"])
        write_jsonl(out_dir / f"{tier}.jsonl", res)
        for b in cc["budgets"]:
            rs = [x for x in res if x["budget"] == b]
            print(f"{tier} @ {b:5d}: acc {100 * sum(x['correct'] for x in rs) / len(rs):5.1f}%  "
                  f"mean tokens {sum(x['tokens'] for x in rs) / len(rs):7.0f}  "
                  f"({sum(x['forced'] for x in rs)} forced)", flush=True)


if __name__ == "__main__":
    main()
