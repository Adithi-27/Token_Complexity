"""Ablations of the paper design and of predictor costs (planning.md, step 14).

Paper variants (config ablations.variants): each redraws `n_papers` test papers with its
own composition, marks or excluded questions (same seed for every variant) and runs
`policies` at the main budgets, scaled by `budget_scale` (so a 40-question paper gets
twice the budget and the budget per question stays the same). Nothing is refitted: the
predictors, the tier ratio and the Table are those of the main run.

  default      10/6/4, marks 1/2/4 (the main design, redrawn)
  equal_marks  every question worth 1 mark
  easy_heavy   14/4/2         hard_heavy  4/6/10
  size_10      5/3/2, half the budget      size_40  20/12/8, twice the budget
  gsm8k_clean  the main design without the 8 GSM8K questions whose gold label is wrong
               or ambiguous (agents/gsm8k_check.md); a sensitivity check, not a relabelling

Predictor-cost settings (config ablations.cost_policies, on the default papers):
  charged      the configured cost: generated tokens per question (probe 128, ICL 1)
  free         every predictor costs nothing
  prompt       ICL also pays its prompt (prefill) tokens per question

Writes <ablations.dir>/summary.json.

Run:  python -m tok_exam.ablations
"""
import json
import random

import numpy as np

from .config import TIERS, load_config, resolve
from .evaluate import load_all_predicted, load_predicted, run_paper
from .papers import sample_papers
from .policies import AT, Context, Table
from .records import read_jsonl


def variant_papers(pool, spec, base, n_papers, seed):
    """Test papers for one variant: spec overrides the base composition and marks."""
    comp = {**base["composition"], **spec.get("composition", {})}
    marks = {**base["marks"], **spec.get("marks", {})}
    drop = set(spec.get("exclude", []))
    rows = [r for r in pool if r["id"] not in drop]
    return sample_papers(rows, "test", n_papers, comp, marks, random.Random(seed))


def icl_prompt_cost(cfg):
    """Mean prompt tokens per question of the main ICL setting, or None if absent."""
    from .difficulty import icl_file

    rows = icl_file(cfg)
    return None if rows is None else float(np.mean([r["prompt_tokens"] for r in rows.values()]))


def with_costs(predictors, setting, prompt_cost=None):
    """Copy of {name: (curves, cost)} with costs for one cost setting."""
    if setting == "charged":
        return dict(predictors)
    if setting == "free":
        return {n: (c, 0.0) for n, (c, _) in predictors.items()}
    if setting == "prompt":
        return {n: (c, cost + prompt_cost if n == "icl" or n.startswith("icl/") or n == "icl+text"
                    else cost) for n, (c, cost) in predictors.items()}
    raise ValueError(setting)


def sweep(ctx, papers, policies, budgets):
    return {name: {B: float(np.mean([run_paper(name, ctx, p, B)["score"] for p in papers]))
                   for B in budgets} for name in policies}


def main():
    cfg = load_config()
    ac, pc = cfg["ablations"], cfg["policies"]
    table = Table(cfg)
    predicted, cost = load_predicted(cfg)
    predictors = load_all_predicted(cfg)
    pool = read_jsonl(resolve(cfg["data"]["processed_dir"]) / "pool.jsonl")
    have = {q for q, _, _ in table.rows}
    pool = [r for r in pool if r["id"] in have]
    budgets = pc["total_budgets"]
    out = {"budgets": budgets, "variants": {}, "costs": {}}

    for name, spec in ac["variants"].items():
        spec = spec or {}
        papers = variant_papers(pool, spec, cfg["papers"], ac["n_papers"], ac["seed"])
        scale = spec.get("budget_scale", 1)
        scaled = [int(B * scale) for B in budgets]
        ctx = Context(cfg, table, predicted=predicted, predictor_cost=cost, predictors=predictors)
        res = sweep(ctx, papers, ac["policies"], scaled)
        out["variants"][name] = {"n_questions": len(papers[0]["questions"]),
                                 "total_marks": sum(q["marks"] for q in papers[0]["questions"]),
                                 "budgets": scaled,
                                 "scores": {p: {str(B): v for B, v in r.items()} for p, r in res.items()}}
        print(f"\n{name} ({len(papers[0]['questions'])} questions, budgets {scaled})")
        for p, r in res.items():
            print(f"  {p:26s} " + " ".join(f"{v:5.1f}" for v in r.values()), flush=True)

    papers = variant_papers(pool, {}, cfg["papers"], ac["n_papers"], ac["seed"])
    prompt = icl_prompt_cost(cfg)
    names = [n for n in ac["cost_policies"] if n.split(AT, 1)[1] in predictors]
    settings = ["charged", "free"] + (["prompt"] if prompt is not None else [])
    print(f"\npredictor cost (ICL prompt tokens per question: {prompt})")
    for setting in settings:
        preds = with_costs(predictors, setting, prompt)
        ctx = Context(cfg, table, predicted=predicted, predictor_cost=cost, predictors=preds)
        res = sweep(ctx, papers, names, budgets)
        out["costs"][setting] = {p: {str(B): v for B, v in r.items()} for p, r in res.items()}
        for p, r in res.items():
            print(f"  {setting:8s} {p:28s} " + " ".join(f"{v:5.1f}" for v in r.values()), flush=True)
    out["icl_prompt_tokens"] = prompt

    d = resolve(ac["dir"])
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "summary.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
