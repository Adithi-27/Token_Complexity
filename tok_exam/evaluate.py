"""Main sweep: every policy on every test paper at every total budget (planning.md, step 12).

Before the sweep, the P2 tier ratio is tuned on the train papers (the ratio with
the best mean score over all total budgets) and kept next to the default 1:2:4.

Writes <policies.dir>/runs.jsonl, one row per (policy, budget, paper), scores
averaged over the samples, and summary.json with, per policy and budget, the
mean and std of the score (% of marks) and the mean thinking and total tokens.
Also "tokens to 90% of oracle": the smallest total budget (log-interpolated)
at which a policy's mean score reaches 90% of the oracle's score at the
largest budget.

The "bandit" policy needs the curves from tok_exam.bandit and is skipped without them.

Run:  python -m tok_exam.evaluate [--split test] [--policies ...]
"""
import argparse
import itertools
import json
import math

import numpy as np

from .config import TIERS, load_config, resolve
from .papers import total_marks
from .policies import AT, ROLL, Context, Table, run_policy, score
from .records import read_jsonl, write_jsonl

BASE_POLICIES = ["P0_equal", "P1_in_order", "P2_ratio", "P2_ratio_tuned", "P3_knapsack",
                 "P4_triage", "P4_triage_closed", "bandit", "P6_icl_paper", "P7_shadow", "oracle",
                 "oracle_hindsight"]
ALL_POLICIES = BASE_POLICIES + [n + ROLL for n in ("P0_equal", "P2_ratio", "P2_ratio_tuned",
                                                  "P3_knapsack", "bandit", "P6_icl_paper",
                                                  "P7_shadow", "oracle")]


def run_paper(name, ctx, paper, B):
    """Mean over samples of one policy on one paper."""
    runs = [score(run_policy(name, ctx, paper, B, s)) for s in range(ctx.table.n_samples)]
    pct = [100 * r["marks"] / total_marks(paper) for r in runs]
    return {"score": float(np.mean(pct)),
            "thinking": float(np.mean([r["thinking"] for r in runs])),
            "tokens": float(np.mean([r["tokens"] for r in runs])),
            "thinking_by_tier": {t: float(np.mean([r["thinking_by_tier"].get(t, 0) for r in runs]))
                                 for t in TIERS}}


def tune_ratio(ctx, papers, budgets, weights):
    """Tier ratio (easy fixed at 1) with the best mean P2 score on `papers`."""
    best, best_score = None, -1
    for wm, wh in itertools.product(weights, weights):
        ratio = {"easy": 1, "moderate": wm, "hard": wh}
        ctx.ratio = ratio
        sc = np.mean([run_paper("P2_ratio", ctx, p, B)["score"] for p in papers for B in budgets])
        if sc > best_score:
            best, best_score = ratio, sc
    return best, float(best_score)


def tokens_to_reach(budgets, scores, target):
    """Smallest budget (interpolated on log2 budget) at which scores reach target."""
    for i, s in enumerate(scores):
        if s >= target:
            if i == 0:
                return float(budgets[0])
            lo, hi = math.log2(budgets[i - 1]), math.log2(budgets[i])
            frac = (target - scores[i - 1]) / (s - scores[i - 1])
            return float(2 ** (lo + frac * (hi - lo)))
    return None


def load_all_predicted(cfg):
    """{predictor name: (curves, cost per question)} for every step-10 predictor."""
    ddir = resolve(cfg["difficulty"]["dir"])
    with open(ddir / "predicted_curves.json") as f:
        pred = json.load(f)
    with open(ddir / "summary.json") as f:
        summ = json.load(f)["predictors"]
    if pred["budgets"] != cfg["curves"]["budgets"]:
        raise ValueError("predicted curves were made on a different budget grid")
    return {n: (c, summ[n]["cost_tokens"]) for n, c in pred["curves"].items()}


def predictor_variants(cfg, available):
    """Policies run once per predictor ("P3_knapsack@icl", ...), for predictors that exist."""
    names = [n for n in cfg["policies"]["compare_predictors"] if n in available]
    return [f"{pol}{AT}{n}" for pol in cfg["policies"]["predictor_policies"] for n in names]


def load_predicted(cfg):
    """Predicted curves and per-question cost of the configured step-10 predictor."""
    ddir = resolve(cfg["difficulty"]["dir"])
    name = cfg["policies"]["predictor"]
    with open(ddir / "predicted_curves.json") as f:
        pred = json.load(f)
    with open(ddir / "summary.json") as f:
        cost = json.load(f)["predictors"][name]["cost_tokens"]
    if pred["budgets"] != cfg["curves"]["budgets"]:
        raise ValueError("predicted curves were made on a different budget grid")
    return pred["curves"][name], cost


def main():
    cfg = load_config()
    pc = cfg["policies"]
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="test", choices=["train", "test"])
    p.add_argument("--policies", nargs="+", default=None,
                   help="default: every policy, plus each predictor variant that exists")
    args = p.parse_args()
    budgets = pc["total_budgets"]

    table = Table(cfg)
    predicted, cost = load_predicted(cfg)
    predictors = load_all_predicted(cfg)
    if args.policies is None:
        args.policies = ALL_POLICIES + predictor_variants(cfg, predictors)
    bandit_file = resolve(cfg["bandit"]["dir"]) / "curves.json"
    learned = None
    if bandit_file.exists():
        with open(bandit_file) as f:
            learned = json.load(f)["curves"]
    elif any(n.startswith("bandit") for n in args.policies):
        print("no bandit curves yet (run tok_exam.bandit): bandit skipped")
        args.policies = [n for n in args.policies if not n.startswith("bandit")]
    m = cfg["icl"]["main"]
    icl_file = resolve(cfg["icl"]["dir"]) / f"paper_{m['selection']}_k{m['k']}_{args.split}.jsonl"
    icl_caps = None
    if icl_file.exists():
        icl_caps = {(r["paper_id"], r["budget"]): r["caps"] for r in read_jsonl(icl_file)}
    elif any(n.startswith("P6") for n in args.policies):
        print("no ICL paper allocations yet (run tok_exam.icl): P6 skipped")
        args.policies = [n for n in args.policies if not n.startswith("P6")]
    ctx = Context(cfg, table, predicted=predicted, predictor_cost=cost, bandit=learned,
                  icl_caps=icl_caps, predictors=predictors)
    papers = read_jsonl(resolve(cfg["papers"]["dir"]) / f"{args.split}.jsonl")

    tuned = None
    if any(n.startswith("P2_ratio_tuned") for n in args.policies):
        train = read_jsonl(resolve(cfg["papers"]["dir"]) / "train.jsonl")
        tuned, tuned_score = tune_ratio(ctx, train, budgets, pc["ratio_search"])
        print(f"tuned tier ratio on train papers: {tuned} (mean score {tuned_score:.1f}%)")

    rows, summary = [], {}
    for name in args.policies:
        tuned_variant = name.startswith("P2_ratio_tuned")
        base = name.replace("P2_ratio_tuned", "P2_ratio")
        ctx.ratio = tuned if tuned_variant else pc["tier_ratio"]
        summary[name] = {}
        for B in budgets:
            res = [run_paper(base, ctx, paper, B) for paper in papers]
            rows += [{"policy": name, "budget": B, "paper_id": paper["paper_id"], **r}
                     for paper, r in zip(papers, res)]
            sc = [r["score"] for r in res]
            marks = float(np.mean([r["score"] * total_marks(p) / 100 for p, r in zip(papers, res)]))
            summary[name][B] = {
                "score_mean": float(np.mean(sc)), "score_std": float(np.std(sc)),
                "tokens_per_mark": float(np.mean([r["thinking"] for r in res])) / marks if marks else None,
                "thinking_mean": float(np.mean([r["thinking"] for r in res])),
                "tokens_mean": float(np.mean([r["tokens"] for r in res])),
                "thinking_by_tier": {t: float(np.mean([r["thinking_by_tier"][t] for r in res]))
                                     for t in TIERS}}
        print(f"{name:15s} " + " ".join(f"{summary[name][B]['score_mean']:6.1f}" for B in budgets),
              flush=True)

    reach = {}
    if "oracle" in summary:
        target = 0.9 * summary["oracle"][budgets[-1]]["score_mean"]
        for name in summary:
            reach[name] = tokens_to_reach(budgets, [summary[name][B]["score_mean"]
                                                    for B in budgets], target)
        print(f"\ntokens to reach 90% of the oracle's best ({target:.1f}%):")
        for name, t in reach.items():
            print(f"  {name:15s} {'not reached' if t is None else f'{t:8.0f}'}")

    out_dir = resolve(pc["dir"])
    write_jsonl(out_dir / f"runs_{args.split}.jsonl", rows)
    with open(out_dir / f"summary_{args.split}.json", "w") as f:
        json.dump({"split": args.split, "budgets": budgets, "predictor": pc["predictor"],
                   "predictor_cost": cost, "tuned_ratio": tuned, "policies": summary,
                   "tokens_to_90pct_oracle": reach}, f, indent=2)


if __name__ == "__main__":
    main()
