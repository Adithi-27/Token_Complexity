"""Does truncation stand in for real budget control? (ICML validity checks)

prefix  Regenerate stratified stored traces with the same prompt and seed and
        max_tokens = cap. A hard cap is exactly what truncation replays if the
        regeneration reproduces the stored stream, so report how many leading
        tokens match the stored trace (vLLM per-request seeds are deterministic, but
        batch composition can change floating-point results).
prompt  A budget-aware prompt: "Think for at most B thinking tokens." is added to
        the question and generation is hard-stopped at B thinking tokens (then the
        answer is forced exactly as in tok_exam.curves). The model now *knows* its
        budget and may plan; compare its accuracy and spend with the truncation
        curve of the same question at the same B.

Questions are drawn per tier from the test split with a fixed seed.

Writes <realbudget.dir>/prefix.jsonl, prompt.jsonl and summary.json.

Run:  python -m tok_exam.realbudget [--only prefix|prompt]
"""
import argparse
import json
import random

import numpy as np

from .config import TIERS, load_config, resolve, sample_seed
from .curves import FORCE, parse_forced
from .grader import grade, is_equiv
from .prompt import build_prompt
from .records import THINK_CLOSE, read_jsonl, split_thinking, write_jsonl


def stratified(rows_by_tier, n_per_tier, seed, split="test"):
    """n_per_tier question ids per tier from `split`, fixed by seed."""
    out = {}
    for t in TIERS:
        ids = sorted({r["id"] for r in rows_by_tier[t] if r["split"] == split})
        out[t] = random.Random(f"{seed}-{t}").sample(ids, min(n_per_tier, len(ids)))
    return out


def matched_prefix(a, b):
    """Number of leading tokens two token lists share."""
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def budget_message(cfg, question, B):
    return question.strip() + "\n" + cfg["realbudget"]["budget_line"].format(budget=B)


def compare(prompt_rows, curve_acc):
    """Per (tier, budget): prompted accuracy vs truncation accuracy on the same questions."""
    out = {}
    for t in TIERS + ("all",):
        for B in sorted({r["budget"] for r in prompt_rows}):
            rs = [r for r in prompt_rows if r["budget"] == B and (t == "all" or r["tier"] == t)]
            if not rs:
                continue
            trunc = [curve_acc[r["id"]][B] for r in rs if r["id"] in curve_acc]
            out[f"{t}@{B}"] = {
                "n": len(rs), "prompted_acc": float(np.mean([r["correct"] for r in rs])),
                "truncation_acc": float(np.mean(trunc)) if trunc else None,
                "prompted_thinking": float(np.mean([r["thinking"] for r in rs])),
                "stopped_by_itself": float(np.mean([not r["forced"] for r in rs]))}
    return out


def run_prefix(cfg, llm, tok, rows_by_tier, ids):
    from vllm import SamplingParams

    rc, s = cfg["realbudget"], cfg["sampling"]
    jobs = [r for t in TIERS for r in rows_by_tier[t] if r["id"] in ids[t]
            and r["sample_idx"] == 0][: rc["prefix_traces"]]
    params = [SamplingParams(temperature=s["temperature"], top_p=s["top_p"],
                             max_tokens=rc["prefix_cap"], seed=sample_seed(cfg, r["sample_idx"]),
                             skip_special_tokens=False) for r in jobs]
    outs = llm.generate([r["prompt"] for r in jobs], params)
    res = []
    for r, o in zip(jobs, outs):
        stored = tok.encode(r["full_output"], add_special_tokens=False)[: rc["prefix_cap"]]
        new = tok.encode(o.outputs[0].text, add_special_tokens=False)[: rc["prefix_cap"]]
        res.append({"id": r["id"], "tier": r["tier"], "cap": rc["prefix_cap"],
                    "matched": matched_prefix(stored, new), "compared": min(len(stored), len(new))})
    return res


def run_prompted(cfg, llm, tok, pool, ids):
    from vllm import SamplingParams

    rc, s = cfg["realbudget"], cfg["sampling"]
    jobs = [(pool[q], B, i) for t in TIERS for q in ids[t]
            for B in rc["budgets"] for i in range(rc["n_samples"])]
    prompts = [build_prompt(cfg, tok, budget_message(cfg, it["question"], B)) for it, B, _ in jobs]
    outs = llm.generate(prompts, [SamplingParams(
        temperature=s["temperature"], top_p=s["top_p"], seed=sample_seed(cfg, i),
        max_tokens=B + s["max_answer_tokens"], skip_special_tokens=False) for _, B, i in jobs])
    rows, forced = [], []
    for (it, B, i), pr, o in zip(jobs, prompts, outs):
        text, ids_ = o.outputs[0].text, list(o.outputs[0].token_ids)
        thinking, answer = split_thinking(text)
        n_think = len(tok.encode(thinking, add_special_tokens=False))
        row = {"id": it["id"], "tier": it["tier"], "budget": B, "sample_idx": i}
        if answer is not None and n_think <= B:
            pred, ok = grade(answer, it["gold"])
            rows.append({**row, "forced": False, "thinking": n_think, "pred": pred, "correct": ok})
        else:
            kept = tok.decode(ids_[:B]).split(THINK_CLOSE)[0]
            rows.append({**row, "forced": True, "thinking": min(B, n_think), "pred": None,
                         "correct": False})
            forced.append((len(rows) - 1, pr + kept + FORCE))
    if forced:
        gens = llm.generate([p for _, p in forced], SamplingParams(
            temperature=0, max_tokens=cfg["curves"]["max_answer_tokens"]))
        for (j, _), g in zip(forced, gens):
            pred = parse_forced(g.outputs[0].text)
            rows[j]["pred"], rows[j]["correct"] = pred, is_equiv(pred, pool[rows[j]["id"]]["gold"])
    return rows


def main():
    cfg = load_config()
    rc = cfg["realbudget"]
    p = argparse.ArgumentParser()
    p.add_argument("--only", choices=["prefix", "prompt"], default=None)
    args = p.parse_args()

    pool = {r["id"]: r for r in read_jsonl(resolve(cfg["data"]["processed_dir"]) / "pool.jsonl")}
    rows_by_tier = {t: read_jsonl(resolve(cfg["output"]["dir"]) / f"{t}.jsonl") for t in TIERS}
    ids = stratified(rows_by_tier, rc["n_per_tier"], rc["seed"])
    out_dir = resolve(rc["dir"])

    from .llm import load_llm
    llm = load_llm(cfg)
    tok = llm.get_tokenizer()
    summary = {}
    if args.only in (None, "prefix"):
        res = run_prefix(cfg, llm, tok, rows_by_tier, ids)
        write_jsonl(out_dir / "prefix.jsonl", res)
        full = sum(r["matched"] == r["compared"] for r in res)
        summary["prefix"] = {"traces": len(res), "fully_reproduced": full,
                             "mean_matched_fraction": float(np.mean(
                                 [r["matched"] / max(r["compared"], 1) for r in res]))}
        print(f"prefix: {full}/{len(res)} traces reproduced token for token up to "
              f"{rc['prefix_cap']}", flush=True)
    if args.only in (None, "prompt"):
        rows = run_prompted(cfg, llm, tok, pool, ids)
        write_jsonl(out_dir / "prompt.jsonl", rows)
        curve_acc = {}
        cdir = resolve(cfg["curves"]["dir"])
        for t in TIERS:
            if (cdir / f"{t}.jsonl").exists():
                acc = {}
                for r in read_jsonl(cdir / f"{t}.jsonl"):
                    acc.setdefault((r["id"], r["budget"]), []).append(r["correct"])
                for (q, B), v in acc.items():
                    curve_acc.setdefault(q, {})[B] = float(np.mean(v))
        summary["prompt"] = compare(rows, curve_acc)
        for k, v in summary["prompt"].items():
            tr = "n/a" if v["truncation_acc"] is None else f"{100 * v['truncation_acc']:5.1f}%"
            print(f"{k:16s} prompted {100 * v['prompted_acc']:5.1f}%  truncation {tr}  "
                  f"thinking {v['prompted_thinking']:7.0f}  stopped itself "
                  f"{100 * v['stopped_by_itself']:5.1f}%", flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
