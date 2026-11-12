"""Question pool with difficulty tiers and a train/test split (planning.md, step 2).

Writes data/processed/pool.jsonl, one row per question:
  id, dataset, tier, level, subject, split, question, gold

Diagram code ([asy]) is kept: many MATH questions cannot be solved without it.

Run:  python -m tok_exam.data
"""
import collections
import random
import re
import sys

from datasets import load_dataset

from .config import TIERS, load_config, resolve
from .grader import normalise_gsm8k_gold
from .records import write_jsonl


def _load(src):
    return load_dataset(src["hf_path"], src["hf_name"], split=src["split"])


def load_gsm8k(cfg, n, rng):
    ds = _load(cfg["data"]["sources"]["gsm8k"])
    picked = sorted(rng.sample(range(len(ds)), n))
    return [{"id": f"gsm8k-{i}", "dataset": "gsm8k", "level": None, "subject": None,
             "question": ds[i]["question"], "gold": normalise_gsm8k_gold(ds[i]["answer"])}
            for i in picked]


def load_math500(cfg, levels):
    rows = []
    for r in _load(cfg["data"]["sources"]["math500"]):
        if int(r["level"]) not in levels:
            continue
        _, subject, num = r["unique_id"].removesuffix(".json").split("/")
        rows.append({"id": f"math-{subject}-{num}", "dataset": "math500", "level": int(r["level"]),
                     "subject": r["subject"], "question": r["problem"], "gold": r["answer"].strip()})
    return rows


def load_aime2025(cfg):
    rows = []
    for r in _load(cfg["data"]["sources"]["aime2025"]):
        m = re.search(r"(\d{4})_AIME_(I{1,2})_Problems/Problem_(\d+)", r["url"])
        rows.append({"id": f"aime-{m[1]}-{m[2]}-{m[3]}", "dataset": "aime2025", "level": None,
                     "subject": None, "question": r["problem"], "gold": str(int(r["answer"]))})
    return rows


def build_tier(cfg, spec, rng):
    rows = []
    if "gsm8k" in spec:
        rows += load_gsm8k(cfg, spec["gsm8k"], rng)
    if "math500_levels" in spec:
        rows += load_math500(cfg, spec["math500_levels"])
    if spec.get("aime2025"):
        rows += load_aime2025(cfg)
    return rows


def assign_split(rows, test_fraction, rng):
    """Shuffle one tier and mark the first test_fraction of it as test."""
    order = list(range(len(rows)))
    rng.shuffle(order)
    n_test = round(len(rows) * test_fraction)
    for rank, i in enumerate(order):
        rows[i]["split"] = "test" if rank < n_test else "train"


def main():
    cfg = load_config()
    dc = cfg["data"]
    rng = random.Random(dc["seed"])

    pool, problems = [], []
    for tier in TIERS:
        rows = build_tier(cfg, dc["tiers"][tier], rng)
        assign_split(rows, dc["test_fraction"], rng)
        pool += [{"id": r["id"], "dataset": r["dataset"], "tier": tier, "level": r["level"],
                  "subject": r["subject"], "split": r["split"], "question": r["question"],
                  "gold": r["gold"]} for r in rows]

    ids = [r["id"] for r in pool]
    if len(set(ids)) != len(ids):
        problems.append("duplicate ids")
    if any(not r["question"].strip() or not r["gold"] for r in pool):
        problems.append("empty question or gold")
    need = cfg["papers"]["composition"]
    counts = collections.Counter((r["tier"], r["split"]) for r in pool)
    for tier in TIERS:
        for split in ("train", "test"):
            if counts[tier, split] < need[tier]:
                problems.append(f"{tier}/{split}: {counts[tier, split]} questions, "
                                f"a paper needs {need[tier]}")

    write_jsonl(resolve(dc["processed_dir"]) / "pool.jsonl", pool)
    print(f"wrote pool.jsonl ({len(pool)} questions)")
    for tier in TIERS:
        by_ds = collections.Counter(r["dataset"] for r in pool if r["tier"] == tier)
        print(f"  {tier:9s} train {counts[tier, 'train']:3d}  test {counts[tier, 'test']:3d}  "
              f"{dict(by_ds)}")
    for p in problems:
        print("PROBLEM:", p)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
