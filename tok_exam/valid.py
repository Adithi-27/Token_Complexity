"""Budget accounting for allocation: what each policy's budget buys.

From the step-7 truncation table: for trace (q, s) the *sufficient budget* s* is the
smallest grid budget at which the forced answer is correct and stays correct at every
larger grid budget. It is the target an allocation should hit. s* lives on the budget
grid (resolution = grid step). A trace correct at some budget but not stably to the
top is counted as "unstable".

Scope (10 Nov 2026): this module is an allocation diagnostic only. Waste inside a trace
(post-answer tail, valid-thinking shares per tier, when to stop) is not studied or
reported in this project; it belongs to the author's separate, independent work (see
planning.md, Scope). per_tier.json still lists s* coverage per tier (secured and unstable
shares); its per-trace valid/invalid token shares are a by-product and are not reported.

Per allocation policy and total budget, every token a policy spends on a
question is split into
  useful     spent below s*  (needed to secure the answer)
  overspend  spent past s*   (the answer was already secured)
  hopeless   spent on a trace with no s* (no budget on the grid secures it)
  check      spent on a second attempt used only to check the answer (P4's pass 1)
and `shortfall` counts tokens missing where the cap stopped below s*.
Overspend + hopeless is what a better allocation could move to the shortfall.

Writes <valid.dir>/traces.jsonl, per_tier.json and policies_{split}.json.

Run:  python -m tok_exam.valid [--split test] [--policies ...]
"""
import argparse
import collections
import json

import numpy as np

from .config import TIERS, load_config, resolve
from .policies import Context, Table, run_policy
from .records import read_jsonl, write_jsonl


def sufficient_budget(correct_by_budget, grid):
    """Smallest b with correct at b and at every larger grid budget, or None."""
    s_star = None
    for b in reversed(grid):
        if not correct_by_budget[b]:
            break
        s_star = b
    return s_star


def trace_split(table, q, s):
    """{s_star, total, valid, invalid, unstable} for one trace."""
    grid = table.budgets
    outs = {b: table.outcome(q, b, s) for b in grid}
    s_star = sufficient_budget({b: o.correct for b, o in outs.items()}, grid)
    total = outs[grid[-1]].thinking
    valid = min(s_star, total) if s_star is not None else 0
    unstable = s_star is None and any(o.correct for o in outs.values())
    return {"s_star": s_star, "total": total, "valid": valid, "invalid": total - valid,
            "unstable": unstable}


def spend_split(split, cap, thinking):
    """Split `thinking` tokens spent at `cap` on a trace with split `split`."""
    if split["s_star"] is None:
        return {"useful": 0, "overspend": 0, "hopeless": thinking, "shortfall": 0}
    useful = min(thinking, split["s_star"])
    return {"useful": useful, "overspend": thinking - useful, "hopeless": 0,
            "shortfall": max(split["s_star"] - cap, 0)}


def per_tier(trace_rows):
    out = {}
    for t in TIERS + ("all",):
        rs = [r for r in trace_rows if t == "all" or r["tier"] == t]
        if not rs:
            continue
        tot = sum(r["total"] for r in rs)
        out[t] = {"traces": len(rs),
                  "secured": float(np.mean([r["s_star"] is not None for r in rs])),
                  "unstable": float(np.mean([r["unstable"] for r in rs])),
                  "valid_share_of_tokens": sum(r["valid"] for r in rs) / tot if tot else None,
                  "median_valid_fraction": float(np.median(
                      [r["valid"] / r["total"] for r in rs if r["s_star"] is not None and r["total"]]
                      or [np.nan])),
                  "mean_total": tot / len(rs), "mean_valid": float(np.mean([r["valid"] for r in rs]))}
    return out


def policy_waste(ctx, papers, budgets, policies, splits):
    """{policy: {B: mean per paper of useful/overspend/hopeless/shortfall tokens}}."""
    tier_of = {q["id"]: q["tier"] for p in papers for q in p["questions"]}
    res = {}
    for name in policies:
        res[name] = {}
        for B in budgets:
            acc = collections.Counter()
            n = 0
            for paper in papers:
                for s in range(ctx.table.n_samples):
                    for r in run_policy(name, ctx, paper, B, s):
                        # P4's second pass-1 attempt is another trace: count it as a check
                        check = r.get("check", 0)
                        part = spend_split(splits[r["id"], s], r["cap"], r["thinking"] - check)
                        part["check"] = check
                        acc.update(part)
                        acc.update({f"{k}_{tier_of[r['id']]}": v for k, v in part.items()})
                    n += 1
            res[name][B] = {k: v / n for k, v in sorted(acc.items())}
    return res


def main():
    from .evaluate import BASE_POLICIES, load_predicted

    cfg = load_config()
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="test", choices=["train", "test"])
    p.add_argument("--policies", nargs="+",
                   default=[n for n in BASE_POLICIES
                            if n not in ("bandit", "P2_ratio_tuned", "P6_icl_paper")])
    args = p.parse_args()

    table = Table(cfg)
    pool = {r["id"]: r for r in read_jsonl(resolve(cfg["data"]["processed_dir"]) / "pool.jsonl")}
    qids = sorted({q for q, _, _ in table.rows})
    splits, trace_rows = {}, []
    for q in qids:
        for s in range(table.n_samples):
            sp = trace_split(table, q, s)
            splits[q, s] = sp
            trace_rows.append({"id": q, "tier": pool[q]["tier"], "dataset": pool[q]["dataset"],
                               "split": pool[q]["split"], "sample_idx": s, **sp})

    tiers = per_tier(trace_rows)
    print(f"{'tier':9s} {'traces':>6s} {'secured':>8s} {'unstable':>8s} {'valid tok %':>11s} "
          f"{'med valid frac':>14s} {'mean T':>7s}")
    for t, v in tiers.items():
        print(f"{t:9s} {v['traces']:6d} {100 * v['secured']:7.1f}% {100 * v['unstable']:7.1f}% "
              f"{100 * (v['valid_share_of_tokens'] or 0):10.1f}% {v['median_valid_fraction']:14.2f} "
              f"{v['mean_total']:7.0f}")

    predicted, cost = load_predicted(cfg)
    ctx = Context(cfg, table, predicted=predicted, predictor_cost=cost)
    papers = read_jsonl(resolve(cfg["papers"]["dir"]) / f"{args.split}.jsonl")
    budgets = cfg["policies"]["total_budgets"]
    waste = policy_waste(ctx, papers, budgets, args.policies, splits)
    print(f"\nper paper, B = {budgets}: useful / overspend / hopeless / shortfall tokens")
    for name, by_b in waste.items():
        print(f"{name:17s} " + " | ".join(
            f"{v['useful']:.0f}/{v['overspend']:.0f}/{v['hopeless']:.0f}/{v['shortfall']:.0f}"
            for v in by_b.values()))

    out = resolve(cfg["valid"]["dir"])
    write_jsonl(out / "traces.jsonl", trace_rows)
    with open(out / "per_tier.json", "w") as f:
        json.dump(tiers, f, indent=2)
    with open(out / f"policies_{args.split}.json", "w") as f:
        json.dump({"budgets": budgets, "policies": waste}, f, indent=2)


if __name__ == "__main__":
    main()
