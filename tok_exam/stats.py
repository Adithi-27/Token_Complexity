"""Two-level paired bootstrap over questions, samples and papers.

The 100 test papers are drawn from ~280 test questions, so they overlap and a
bootstrap over papers alone understates the variance. One replicate:
  1. resample the test questions of each tier with replacement (a question drawn
     twice becomes two distinct virtual questions "id#0", "id#1");
  2. resample the samples behind each virtual question with replacement;
  3. redraw the papers with the configured composition from the resampled pool;
  4. run every policy on these same papers and samples (paired).
Policies run unchanged on a resampled view of the truncation table, and every
predictor's curves are looked up through the virtual ids, so fitted predictors
are not refitted (the train split is fixed; split variance is a separate study).
P6 is left out: its allocations exist only for the real test papers.

Reports, per policy and budget, the mean and a 95% percentile interval; for each
pre-registered comparison (config stats.comparisons), the paired difference, its
interval and a two-sided bootstrap p-value, Holm-corrected over all comparisons
x budgets (plus the area under score vs log2 budget).

Writes <stats.dir>/bootstrap_{split}.json.

Run:  python -m tok_exam.stats [--n-boot 1000] [--workers 8]
"""
import argparse
import json
import math

import numpy as np

from .config import TIERS, load_config, resolve
from .papers import total_marks
from .policies import AT, ROLL, Context, Table, run_policy, score
from .records import read_jsonl


class ResampledTable:
    """The truncation table seen through virtual question ids and resampled samples."""

    def __init__(self, base, origin, samples):
        self.base, self.origin, self.samples = base, origin, samples
        self.budgets, self.n_samples = base.budgets, base.n_samples

    def outcome(self, q, b, s):
        return self.base.outcome(self.origin[q], b, self.samples[q][s])

    def mean_acc(self, q):
        return [float(np.mean([self.outcome(q, b, s).correct for s in range(self.n_samples)]))
                for b in self.budgets]


class ByOrigin(dict):
    """A per-question dict (curves) looked up through virtual ids."""

    def __init__(self, base, origin):
        super().__init__()
        self.base, self.origin = base, origin

    def __missing__(self, q):
        return self.base[self.origin[q]]

    def __contains__(self, q):
        return q in self.origin and self.origin[q] in self.base


def replicate(cfg, table, test_items, rng):
    """(resampled table, origin map, papers) for one bootstrap replicate."""
    pc = cfg["papers"]
    origin, samples, pool = {}, {}, {t: [] for t in TIERS}
    for t in TIERS:
        ids = [it["id"] for it in test_items if it["tier"] == t]
        for k, i in enumerate(rng.integers(len(ids), size=len(ids))):
            v = f"{ids[i]}#{k}"
            origin[v] = ids[i]
            samples[v] = rng.integers(table.n_samples, size=table.n_samples).tolist()
            pool[t].append(v)
    papers = []
    for n in range(pc["n_papers"]):
        qs = [{"id": v, "tier": t, "marks": pc["marks"][t]}
              for t in TIERS for v in rng.choice(pool[t], pc["composition"][t], replace=False)]
        papers.append({"paper_id": f"boot-{n:03d}", "questions": qs})
    return ResampledTable(table, origin, samples), origin, papers


def paper_scores(name, ctx, papers, B):
    """Mean score (%) over papers, each averaged over samples."""
    out = []
    for paper in papers:
        runs = [score(run_policy(name, ctx, paper, B, s))["marks"]
                for s in range(ctx.table.n_samples)]
        out.append(100 * np.mean(runs) / total_marks(paper))
    return float(np.mean(out))


def auc(budgets, scores):
    """Mean score over log2 budget (trapezoid), in % marks."""
    x, y = np.log2(budgets), np.asarray(scores, dtype=float)
    return float(np.sum((y[1:] + y[:-1]) / 2 * np.diff(x)) / (x[-1] - x[0]))


def make_context(cfg, table, base_ctx, origin):
    ctx = Context(cfg, table, predicted=ByOrigin(base_ctx.predicted, origin),
                  predictor_cost=base_ctx.predictor_cost,
                  ratio=base_ctx.ratio,
                  bandit=ByOrigin(base_ctx.bandit, origin) if base_ctx.bandit else None,
                  predictors={n: (ByOrigin(c, origin), cost)
                              for n, (c, cost) in base_ctx.predictors.items()})
    return ctx


_STATE = {}


def init_worker(state):
    """Shared, read-only state for every replicate (sent once per worker, not per job)."""
    _STATE.update(state)


def one_replicate(seed):
    st = _STATE
    cfg, table, base_ctx, tuned = st["cfg"], st["table"], st["base_ctx"], st["tuned"]
    test_items, policies, budgets = st["test_items"], st["policies"], st["budgets"]
    rng = np.random.default_rng(seed)
    rtable, origin, papers = replicate(cfg, table, test_items, rng)
    ctx = make_context(cfg, rtable, base_ctx, origin)
    res = {}
    for name in policies:
        ctx.ratio = tuned if name.startswith("P2_ratio_tuned") and tuned else cfg["policies"]["tier_ratio"]
        run_name = name.replace("P2_ratio_tuned", "P2_ratio")
        res[name] = [paper_scores(run_name, ctx, papers, B) for B in budgets]
    return res


def holm(pvals):
    """Holm-adjusted p-values, same order as given."""
    order = np.argsort(pvals)
    m, adj, running = len(pvals), [0.0] * len(pvals), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        adj[i] = running
    return adj


def summarise(reps, policies, budgets, comparisons):
    """Intervals per policy and budget, and the paired comparison tests."""
    arr = {n: np.array([r[n] for r in reps]) for n in policies}
    out = {"policies": {}, "comparisons": []}
    for n in policies:
        a = arr[n]
        aucs = np.array([auc(budgets, row) for row in a])
        out["policies"][n] = {
            **{str(B): {"boot_mean": float(a[:, j].mean()),
                        "ci95": [float(np.percentile(a[:, j], 2.5)),
                                 float(np.percentile(a[:, j], 97.5))]}
               for j, B in enumerate(budgets)},
            "auc": {"boot_mean": float(aucs.mean()),
                    "ci95": [float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))]}}
    tests = []
    for x, y in comparisons:
        d = arr[x] - arr[y]
        cols = [(str(B), d[:, j]) for j, B in enumerate(budgets)]
        cols.append(("auc", np.array([auc(budgets, rx) - auc(budgets, ry)
                                      for rx, ry in zip(arr[x], arr[y])])))
        for where, diff in cols:
            p = min(1.0, 2 * min(np.mean(diff <= 0), np.mean(diff >= 0)))
            tests.append({"a": x, "b": y, "at": where, "diff_mean": float(diff.mean()),
                          "ci95": [float(np.percentile(diff, 2.5)),
                                   float(np.percentile(diff, 97.5))], "p": float(p)})
    for t, padj in zip(tests, holm([t["p"] for t in tests])):
        t["p_holm"] = padj
    out["comparisons"] = tests
    return out


def main():
    from multiprocessing import Pool

    from .evaluate import load_all_predicted, load_predicted

    cfg = load_config()
    sc = cfg["stats"]
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="test", choices=["train", "test"])
    p.add_argument("--n-boot", type=int, default=sc["n_boot"])
    p.add_argument("--workers", type=int, default=sc["workers"])
    args = p.parse_args()
    budgets = sc["budgets"]

    table = Table(cfg)
    predicted, cost = load_predicted(cfg)
    predictors = load_all_predicted(cfg)
    bandit = None
    bfile = resolve(cfg["bandit"]["dir"]) / "curves.json"
    if bfile.exists():
        with open(bfile) as f:
            bandit = json.load(f)["curves"]
    base_ctx = Context(cfg, table, predicted=predicted, predictor_cost=cost, bandit=bandit,
                       predictors=predictors)
    tuned = None
    sfile = resolve(cfg["policies"]["dir"]) / f"summary_{args.split}.json"
    if sfile.exists():
        with open(sfile) as f:
            tuned = json.load(f).get("tuned_ratio")

    def available(name):
        base, _, pred = name.partition(AT)
        if pred and pred not in predictors:
            return False
        if base.removesuffix(ROLL) == "bandit" and bandit is None:
            return False
        return not base.startswith("P6")

    comparisons = [c for c in sc["comparisons"] if all(available(n) for n in c)]
    for c in sc["comparisons"]:
        if c not in comparisons:
            print(f"skipped comparison {c[0]} vs {c[1]}: a policy or predictor is missing")
    policies = list(dict.fromkeys([n for n in sc["policies"] if available(n)]
                                  + [n for c in comparisons for n in c]))

    pool = read_jsonl(resolve(cfg["data"]["processed_dir"]) / "pool.jsonl")
    have = {q for q, _, _ in table.rows}
    test_items = [it for it in pool if it["split"] == args.split and it["id"] in have]
    seeds = np.random.SeedSequence(sc["seed"]).generate_state(args.n_boot).tolist()
    state = {"cfg": cfg, "table": table, "base_ctx": base_ctx, "tuned": tuned,
             "test_items": test_items, "policies": policies, "budgets": budgets}
    print(f"{args.n_boot} replicates x {len(policies)} policies x {len(budgets)} budgets "
          f"on {args.workers} workers", flush=True)
    if args.workers > 1:
        with Pool(args.workers, initializer=init_worker, initargs=(state,)) as pool_:
            reps = pool_.map(one_replicate, seeds,
                             chunksize=max(1, args.n_boot // (4 * args.workers)))
    else:
        init_worker(state)
        reps = [one_replicate(s) for s in seeds]

    out = summarise(reps, policies, budgets, comparisons)
    out.update({"split": args.split, "n_boot": args.n_boot, "budgets": budgets,
                "tuned_ratio": tuned})
    print(f"\n{'policy':28s} " + " ".join(f"{B:>17d}" for B in budgets) + f" {'AUC':>17s}")
    for n, v in out["policies"].items():
        cells = [v[str(B)] for B in budgets] + [v["auc"]]
        print(f"{n:28s} " + " ".join(f"{c['boot_mean']:5.1f} [{c['ci95'][0]:4.1f},{c['ci95'][1]:5.1f}]"
                                     for c in cells))
    print("\npre-registered comparisons (paired difference, 95% CI, Holm p)")
    for t in out["comparisons"]:
        print(f"  {t['a']} - {t['b']} @ {t['at']:>5s}: {t['diff_mean']:+6.2f} "
              f"[{t['ci95'][0]:+6.2f}, {t['ci95'][1]:+6.2f}]  p_holm={t['p_holm']:.4f}")
    out_dir = resolve(sc["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"bootstrap_{args.split}.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
