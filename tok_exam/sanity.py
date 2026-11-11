"""Accuracy and length per tier and dataset (planning.md, step 8).

Accuracy should fall from easy to moderate to hard. Easy below about 85% points
to a prompt or grader bug. A row whose thinking hit the cap has no answer, so
`capped` also shows whether max_thinking_tokens is too small.

Run:  python -m tok_exam.sanity [--in-dir outputs/smoke]
"""
import argparse
import collections
import statistics
import sys

from .config import TIERS, load_config, resolve
from .records import hit_cap, read_jsonl

EASY_FLOOR = 0.85


def summary(rows):
    toks = [r["thinking_tokens"] for r in rows]
    return {"n": len(rows), "acc": sum(r["correct"] for r in rows) / len(rows),
            "capped": sum(hit_cap(r) for r in rows), "no_box": sum(r["pred"] is None for r in rows),
            "median_tokens": statistics.median(toks), "max_tokens": max(toks)}


def main():
    cfg = load_config()
    p = argparse.ArgumentParser()
    p.add_argument("--in-dir", default=cfg["output"]["dir"])
    args = p.parse_args()

    groups, problems = {}, []
    for tier in TIERS:
        path = resolve(args.in_dir) / f"{tier}.jsonl"
        if not path.exists():
            continue
        rows = read_jsonl(path)
        groups[tier, "all"] = rows
        by_ds = collections.defaultdict(list)
        for r in rows:
            by_ds[r["dataset"]].append(r)
        if len(by_ds) > 1:
            groups.update({(tier, ds): rs for ds, rs in sorted(by_ds.items())})

    print(f"{'tier':9s} {'dataset':9s} {'n':>5s} {'acc%':>6s} {'capped':>6s} {'no box':>6s} "
          f"{'median tok':>10s} {'max tok':>8s}")
    for (tier, ds), rows in groups.items():
        s = summary(rows)
        print(f"{tier:9s} {ds:9s} {s['n']:5d} {100 * s['acc']:6.1f} {s['capped']:6d} "
              f"{s['no_box']:6d} {s['median_tokens']:10.0f} {s['max_tokens']:8d}")

    accs = [summary(groups[t, "all"])["acc"] for t in TIERS if (t, "all") in groups]
    if ("easy", "all") in groups and accs[0] < EASY_FLOOR:
        problems.append(f"easy accuracy {100 * accs[0]:.1f}% is below {100 * EASY_FLOOR:.0f}%")
    if len(accs) == len(TIERS) and not accs[0] >= accs[1] >= accs[2]:
        problems.append("accuracy does not fall from easy to moderate to hard")
    for msg in problems:
        print("PROBLEM:", msg)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
