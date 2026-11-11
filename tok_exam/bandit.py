"""Difficulty-aware contextual bandit (planning.md, step 13).

Context of a question: [1, tier one-hot, predicted curve from step 10]. Arms:
the budget grid. LinUCB keeps one ridge regression per arm that estimates the
probability of a correct answer at that budget. Training replays the stored
truncation results of train questions (a random sample per pull), so it costs
no GPU time. Each round draws a token price lambda and pulls the arm with the
best upper bound on  marks * p - lambda * budget / 1000,  so every arm gets
explored at some price.

Used two ways:
  lambda sweep  each question takes argmax_a marks * p_a - lambda * a / 1000;
                one point (score, thinking) per lambda, compared with the same
                rule on the step-10 predicted curves and on the true curves.
  "bandit"      learned curves fed to the knapsack in tok_exam.evaluate.

Writes <bandit.dir>/curves.json (learned curve per question) and summary.json.

Run:  python -m tok_exam.bandit
"""
import json

import numpy as np

from .config import TIERS, load_config, resolve
from .evaluate import load_predicted
from .papers import total_marks
from .policies import Table
from .records import read_jsonl


def context(item, predicted):
    return np.r_[1.0, [item["tier"] == t for t in TIERS], predicted[item["id"]]]


class LinUCB:
    def __init__(self, n_arms, dim, alpha, reg):
        self.alpha = alpha
        self.A = np.stack([reg * np.eye(dim)] * n_arms)
        self.b = np.zeros((n_arms, dim))

    def estimate(self, x):
        """(mean, width) of the predicted success probability of every arm."""
        A_inv = np.linalg.inv(self.A)
        theta = np.einsum("aij,aj->ai", A_inv, self.b)
        width = np.sqrt(np.einsum("i,aij,j->a", x, A_inv, x))
        return theta @ x, width

    def update(self, arm, x, reward):
        self.A[arm] += np.outer(x, x)
        self.b[arm] += reward * x

    def curve(self, x):
        """Estimated accuracy per arm, clipped to [0, 1] and non-decreasing."""
        mean, _ = self.estimate(x)
        return np.maximum.accumulate(np.clip(mean, 0, 1))


def train(cfg, table, items, predicted, rng):
    bc = cfg["bandit"]
    grid = np.array(table.budgets)
    marks = cfg["papers"]["marks"]
    model = LinUCB(len(grid), len(context(items[0], predicted)), bc["alpha"], bc["reg"])
    lambdas = bc["lambdas"]
    for _ in range(bc["epochs"]):
        for i in rng.permutation(len(items)):
            it = items[i]
            x = context(it, predicted)
            lam = lambdas[rng.integers(len(lambdas))]
            mean, width = model.estimate(x)
            ucb = marks[it["tier"]] * (mean + width) - lam * grid / 1000
            arm = int(np.argmax(ucb))
            o = table.outcome(it["id"], int(grid[arm]), int(rng.integers(table.n_samples)))
            model.update(arm, x, float(o.correct))
    return model


def lambda_point(table, papers, curves, lam):
    """Mean score (%) and thinking per paper when each question takes
    argmax_a marks * curve_a - lam * a / 1000, run with every sample."""
    grid = np.array(table.budgets)
    scores, thinking = [], []
    for paper in papers:
        for s in range(table.n_samples):
            got = spent = 0
            for q in paper["questions"]:
                arm = int(np.argmax(q["marks"] * np.asarray(curves[q["id"]]) - lam * grid / 1000))
                o = table.outcome(q["id"], int(grid[arm]), s)
                got += q["marks"] * o.correct
                spent += o.thinking
            scores.append(100 * got / total_marks(paper))
            thinking.append(spent)
    return float(np.mean(scores)), float(np.mean(thinking))


def main():
    cfg = load_config()
    bc = cfg["bandit"]
    table = Table(cfg)
    predicted, _ = load_predicted(cfg)
    pool = [it for it in read_jsonl(resolve(cfg["data"]["processed_dir"]) / "pool.jsonl")
            if it["id"] in predicted]
    train_items = [it for it in pool if it["split"] == "train"]
    test_items = [it for it in pool if it["split"] == "test"]

    model = train(cfg, table, train_items, predicted, np.random.default_rng(bc["seed"]))
    learned = {it["id"]: model.curve(context(it, predicted)).round(4).tolist() for it in pool}
    true = {it["id"]: table.mean_acc(it["id"]) for it in pool}

    def brier(curves):
        return float(np.mean([(np.asarray(curves[it["id"]]) - true[it["id"]]) ** 2
                              for it in test_items]))

    print(f"Brier on test questions: predicted {brier(predicted):.4f}  bandit {brier(learned):.4f}")
    papers = read_jsonl(resolve(cfg["papers"]["dir"]) / "test.jsonl")
    sweep = {}
    print(f"\n{'lambda':>7s}  " + "  ".join(f"{n:>17s}" for n in ("predicted", "bandit", "oracle")))
    for lam in bc["lambdas"]:
        sweep[lam] = {n: lambda_point(table, papers, c, lam)
                      for n, c in (("predicted", predicted), ("bandit", learned), ("oracle", true))}
        print(f"{lam:7.3f}  " + "  ".join(f"{s:6.1f}% {t:8.0f} tok" for s, t in sweep[lam].values()))

    out_dir = resolve(bc["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "curves.json", "w") as f:
        json.dump({"budgets": table.budgets, "curves": learned}, f)
    with open(out_dir / "summary.json", "w") as f:
        json.dump({"brier_test": {"predicted": brier(predicted), "bandit": brier(learned)},
                   "lambda_sweep": {str(k): v for k, v in sweep.items()}}, f, indent=2)


if __name__ == "__main__":
    main()
