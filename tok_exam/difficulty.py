"""Empirical difficulty and difficulty predictors (planning.md, steps 9-10).

Needed budget of a question: the smallest grid budget at which at least half
its samples are correct (step 9). Its bin is that budget's index in the grid,
or len(grid) if no budget is enough.

A predictor maps question features to an estimated accuracy curve p_i(b), one
logistic regression per budget, fitted on train questions only. The curve is
what the knapsack policy needs (step 11); the predicted needed budget is read
off it. Feature sets:
  tier   (a) dataset tier one-hot
  text   (b) cheap features of the question text
  self   (c) the model's own 1-5 rating          (from tok_exam.probe, GPU)
  probe  (d) statistics of a 128-token probe     (from tok_exam.probe, GPU)
  embed  (e) MiniLM sentence embedding
  icl    (f) in-context learning: the model's distribution over needed-budget bins
             after k labelled train exemplars (from tok_exam.icl, GPU), as features
  icl_raw    the same distribution read directly as a curve, nothing fitted
  knn        control for ICL: mean true curve of the k most similar train questions
             (the exemplars ICL would see), no LLM
A missing GPU feature file just drops that predictor.

Fitted predictors give test questions the fit on all train questions and train
questions a 5-fold out-of-fold prediction, so no question's curve has seen its own
label (the bandit trains on the train curves).

Writes <difficulty.dir>/needed.jsonl (true needed budget per question),
predicted_curves.json and summary.json.

Run:  python -m tok_exam.difficulty
"""
import json
import re

import numpy as np

from .config import TIERS, load_config, resolve
from .curves import load_curves
from .records import read_jsonl, write_jsonl

_LATEX_CMD = re.compile(r"\\[A-Za-z]+")
_NUMBER = re.compile(r"\d+(?:\.\d+)?")
KEYWORDS = ("how many", "find all", "probability", "sum of", "remainder", "prove", "maximum",
            "minimum", "triangle", "circle", "polynomial", "integer", "real", "complex", "digits")


# ---- step 9: needed budget -------------------------------------------------

def curve_matrix(curves, ids, budgets):
    """[n questions, n budgets] accuracy from load_curves output."""
    return np.array([[curves[q][b][0] for b in budgets] for q in ids])


def needed_bins(acc, threshold):
    """Index of the first budget with accuracy >= threshold, or n_budgets if none."""
    ok = acc >= threshold
    return np.where(ok.any(1), ok.argmax(1), acc.shape[1])


def bin_label(i, budgets):
    return str(budgets[i]) if i < len(budgets) else "never"


# ---- step 10: features -----------------------------------------------------

def tier_features(items):
    return np.array([[it["tier"] == t for t in TIERS] for it in items], dtype=float)


def text_features(items):
    rows = []
    for it in items:
        q = it["question"]
        low = q.lower()
        rows.append([np.log1p(len(q)), np.log1p(len(q.split())), len(_NUMBER.findall(q)),
                     len(_LATEX_CMD.findall(q)), q.count("$") / 2, "[asy]" in q]
                    + [k in low for k in KEYWORDS])
    return np.array(rows, dtype=float)


def embed_features(items, model_name):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device="cpu")
    return model.encode([it["question"] for it in items], batch_size=64, show_progress_bar=False)


def cached_embeddings(cfg, items):
    """Embeddings of `items` from the pool cache that tok_exam.icl writes (computed and
    cached on first use), so GPU jobs with the older transformers need no MiniLM load."""
    from .icl import load_embeddings

    pool = read_jsonl(resolve(cfg["data"]["processed_dir"]) / "pool.jsonl")
    E = load_embeddings(cfg, pool, resolve(cfg["icl"]["dir"]) / "embeddings.npz")
    row = {r["id"]: i for i, r in enumerate(pool)}
    return E[[row[it["id"]] for it in items]]


def probe_features(items, path, keys):
    """Columns `keys` from a tok_exam.probe output file, or None if it does not exist."""
    if not path.exists():
        return None
    by_id = {r["id"]: r for r in read_jsonl(path)}
    return np.array([[by_id[it["id"]][k] for k in keys] for it in items], dtype=float)


def probe_cost(path):
    """Mean generated tokens per question that a probe costs, or 0 if absent."""
    if not path.exists():
        return 0.0
    rows = read_jsonl(path)
    return sum(r["cost_tokens"] for r in rows) / len(rows)


# ---- step 10: curve predictor ----------------------------------------------

class CurvePredictor:
    """One logistic regression per budget on standardised features.

    Each question's accuracy at a budget is a soft label: it enters the fit as
    a positive row weighted acc and a negative row weighted 1 - acc. Predicted
    curves are made non-decreasing in the budget.
    """

    def __init__(self, C=1.0):
        self.C = C

    def fit(self, X, acc):
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        self.scaler = StandardScaler().fit(X)
        Z = self.scaler.transform(X)
        Z2, y = np.vstack([Z, Z]), np.r_[np.ones(len(Z)), np.zeros(len(Z))]
        self.models = []
        for j in range(acc.shape[1]):
            w = np.r_[acc[:, j], 1 - acc[:, j]]
            if w[:len(Z)].sum() == 0 or w[len(Z):].sum() == 0:  # one class only
                self.models.append(float(acc[:, j].mean()))
                continue
            m = LogisticRegression(C=self.C, max_iter=2000)
            self.models.append(m.fit(Z2, y, sample_weight=w + 1e-9))
        return self

    def predict(self, X):
        Z = self.scaler.transform(X)
        cols = [np.full(len(Z), m) if isinstance(m, float) else m.predict_proba(Z)[:, 1]
                for m in self.models]
        return np.maximum.accumulate(np.column_stack(cols), axis=1)


def out_of_fold(X, acc, train, C, folds=5, seed=0):
    """Predictions for the train rows, each from a model fitted without its fold.
    Train curves feed the bandit's context; in-sample fits there would be more
    accurate than the test-time curves the bandit is then used with."""
    idx = np.where(train)[0]
    fold = np.random.default_rng(seed).permutation(len(idx)) % folds
    out = np.zeros((len(idx), acc.shape[1]))
    for f in range(folds):
        fit, held = idx[fold != f], fold == f
        out[held] = CurvePredictor(C=C).fit(X[fit], acc[fit]).predict(X[idx[held]])
    return out


def evaluate(pred, acc, threshold):
    """How well a predicted curve matches the true one on held-out questions."""
    from scipy.stats import spearmanr

    true_bin, pred_bin = needed_bins(acc, threshold), needed_bins(pred, threshold)
    rho = spearmanr(true_bin, pred_bin).statistic
    return {"brier": float(((pred - acc) ** 2).mean()),
            "bin_acc": float((true_bin == pred_bin).mean()),
            "bin_within_1": float((abs(true_bin - pred_bin) <= 1).mean()),
            "spearman": None if np.isnan(rho) else float(rho)}


def knn_neighbours(E, train, k):
    """[n, k] indices of the k most cosine-similar train questions (never the question itself)."""
    Z = E / np.linalg.norm(E, axis=1, keepdims=True)
    tr = np.where(train)[0]
    sims = Z @ Z[tr].T
    sims[tr, np.arange(len(tr))] = -np.inf       # a train question is not its own neighbour
    return tr[np.argsort(-sims, axis=1)[:, :k]]


def knn_curves(E, acc, train, k):
    """Mean accuracy curve of the k most cosine-similar train questions (never itself)."""
    return acc[knn_neighbours(E, train, k)].mean(axis=1)


def knn_majority_bins(E, bins, train, k):
    """Label-level ICL control: the most common needed bin among the k neighbours
    (ties go to the smaller bin)."""
    nn = knn_neighbours(E, train, k)
    n_bins = int(bins.max()) + 1
    return np.array([np.bincount(bins[row], minlength=n_bins).argmax() for row in nn])


def bin_metrics(true_bin, pred_bin):
    """Exact, within-one and Spearman agreement of two needed-bin vectors."""
    from scipy.stats import spearmanr

    rho = spearmanr(true_bin, pred_bin).statistic
    return {"bin_acc": float((true_bin == pred_bin).mean()),
            "bin_within_1": float((abs(true_bin - pred_bin) <= 1).mean()),
            "spearman": None if np.isnan(rho) else float(rho)}


def icl_main(cfg):
    m = cfg["icl"]["main"]
    return f"{m['selection']}_k{m['k']}"


def icl_files(cfg):
    """{setting "retrieved_k8": {id: row}} for every tok_exam.icl question file present."""
    out = {}
    for path in sorted(resolve(cfg["icl"]["dir"]).glob("question_*_k*.jsonl")):
        out[path.stem.removeprefix("question_")] = {r["id"]: r for r in read_jsonl(path)}
    return out


def icl_file(cfg):
    """{id: row} of the main ICL setting, or None if tok_exam.icl has not run."""
    return icl_files(cfg).get(icl_main(cfg))


def icl_cost(cfg, rows):
    """Tokens per question an ICL prediction costs (prompt tokens only if charged)."""
    key = "prompt_tokens" if cfg["icl"]["charge_prompt_tokens"] else "cost_tokens"
    return float(np.mean([r[key] for r in rows.values()]))


def feature_sets(cfg, items):
    """{name: (features, cost in generated tokens per question)}; GPU ones only if present."""
    dc = cfg["difficulty"]
    pdir = resolve(dc["probe_dir"])
    tier, text = tier_features(items), text_features(items)
    sets = {"tier": (tier, 0.0), "text": (text, 0.0), "tier+text": (np.hstack([tier, text]), 0.0)}
    sets["embed"] = (cached_embeddings(cfg, items), 0.0)
    self_f = probe_features(items, pdir / "self.jsonl", ["rating"])
    probe_f = probe_features(items, pdir / "probe.jsonl", ["entropy", "neg_logprob", "closed"])
    if self_f is not None:
        sets["self"] = (np.hstack([tier, self_f]), probe_cost(pdir / "self.jsonl"))
    if probe_f is not None:
        sets["probe"] = (np.hstack([tier, probe_f]), probe_cost(pdir / "probe.jsonl"))
    # one fitted ICL predictor per setting ("icl/random_k4"); "icl" is the main setting
    for setting, rows in icl_files(cfg).items():
        icl_f = np.array([rows[it["id"]]["probs"] + [rows[it["id"]]["expected_bin"]]
                          for it in items])
        sets[f"icl/{setting}"] = (np.hstack([tier, icl_f]), icl_cost(cfg, rows))
        if setting == icl_main(cfg):
            sets["icl"] = sets[f"icl/{setting}"]
            # exploratory (added 11 Nov, after the curves): ICL plus the cheap text features,
            # to fix ICL's calibration at small budgets; not a pre-registered predictor
            sets["icl+text"] = (np.hstack([tier, text, icl_f]), icl_cost(cfg, rows))
    if self_f is not None and probe_f is not None:
        sets["all"] = (np.hstack([tier, text, self_f, probe_f]),
                       probe_cost(pdir / "self.jsonl") + probe_cost(pdir / "probe.jsonl"))
    return sets


def main():
    cfg = load_config()
    dc = cfg["difficulty"]
    budgets, thr = cfg["curves"]["budgets"], dc["threshold"]
    out_dir = resolve(dc["dir"])

    pool = read_jsonl(resolve(cfg["data"]["processed_dir"]) / "pool.jsonl")
    curves = load_curves(cfg)
    items = [it for it in pool if it["id"] in curves]
    missing = len(pool) - len(items)
    acc = curve_matrix(curves, [it["id"] for it in items], budgets)
    train = np.array([it["split"] == "train" for it in items])

    # step 9
    bins = needed_bins(acc, thr)
    write_jsonl(out_dir / "needed.jsonl",
                [{"id": it["id"], "tier": it["tier"], "dataset": it["dataset"],
                  "split": it["split"], "needed_bin": int(b), "needed": bin_label(b, budgets)}
                 for it, b in zip(items, bins)])
    print(f"needed budget (acc >= {thr}) by tier, {len(items)} questions"
          + (f" ({missing} without curves)" if missing else ""))
    labels = [bin_label(i, budgets) for i in range(len(budgets) + 1)]
    print(f"{'tier':9s} " + " ".join(f"{l:>6s}" for l in labels))
    for t in TIERS:
        sel = np.array([it["tier"] == t for it in items])
        counts = np.bincount(bins[sel], minlength=len(labels))
        print(f"{t:9s} " + " ".join(f"{c:6d}" for c in counts))

    # step 10
    summary, predicted = {}, {}
    print(f"\n{'predictor':22s} {'brier':>6s} {'bin acc':>7s} {'±1 bin':>6s} "
          f"{'spearman':>8s} {'cost tok':>8s}")
    feature_sets_cache = {}
    for name, (X, cost) in feature_sets(cfg, items).items():
        feature_sets_cache[name] = X
        model = CurvePredictor(C=dc["C"]).fit(X[train], acc[train])
        pred = model.predict(X)          # test rows: fitted on all train questions
        pred[train] = out_of_fold(X, acc, train, dc["C"], seed=cfg["data"]["seed"])
        summary[name] = {**evaluate(pred[~train], acc[~train], thr), "cost_tokens": cost}
        predicted[name] = {it["id"]: [round(float(p), 4) for p in row]
                           for it, row in zip(items, pred)}
        s = summary[name]
        rho = "n/a" if s["spearman"] is None else f"{s['spearman']:.3f}"
        print(f"{name:22s} {s['brier']:6.3f} {s['bin_acc']:7.3f} {s['bin_within_1']:6.3f} "
              f"{rho:>8s} {cost:8.1f}")

    def add_direct(name, pred, cost):
        summary[name] = {**evaluate(pred[~train], acc[~train], thr), "cost_tokens": cost}
        predicted[name] = {it["id"]: [round(float(p), 4) for p in row]
                           for it, row in zip(items, pred)}
        s = summary[name]
        rho = "n/a" if s["spearman"] is None else f"{s['spearman']:.3f}"
        print(f"{name:22s} {s['brier']:6.3f} {s['bin_acc']:7.3f} {s['bin_within_1']:6.3f} "
              f"{rho:>8s} {cost:8.1f}")

    # ICL control: average the true curves of the same retrieved train neighbours, for
    # every k ICL is run with ("knn/k4"); "knn" uses the main k
    E = feature_sets_cache["embed"]
    for k in sorted(set(cfg["icl"]["shots"]) - {0} | {cfg["icl"]["main"]["k"]}):
        pred = knn_curves(E, acc, train, k)
        add_direct(f"knn/k{k}", pred, 0.0)
        if k == cfg["icl"]["main"]["k"]:
            add_direct("knn", pred, 0.0)

    # the ICL distribution itself, nothing fitted ("icl_raw/retrieved_k8"; main = "icl_raw")
    from .icl import curve_from_probs
    for setting, rows in icl_files(cfg).items():
        pred = np.array([curve_from_probs(np.array(rows[it["id"]]["probs"])) for it in items])
        add_direct(f"icl_raw/{setting}", pred, icl_cost(cfg, rows))
        if setting == icl_main(cfg):
            add_direct("icl_raw", pred, icl_cost(cfg, rows))

    # label-level controls (a bin, not a curve, so not usable by the knapsack): the
    # majority needed bin of the same k neighbours vs the ICL letter the model ranks first
    k = cfg["icl"]["main"]["k"]
    labels = {"knn_majority": knn_majority_bins(E, bins, train, k)}
    main_rows = icl_file(cfg)
    if main_rows is not None:
        labels["icl_argmax"] = np.array([int(np.argmax(main_rows[it["id"]]["probs"])) for it in items])
    label_controls = {n: bin_metrics(bins[~train], b[~train]) for n, b in labels.items()}
    for n, m in label_controls.items():
        rho = "n/a" if m["spearman"] is None else f"{m['spearman']:.3f}"
        print(f"{n:22s} {'':6s} {m['bin_acc']:7.3f} {m['bin_within_1']:6.3f} {rho:>8s}  (label level)")

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "predicted_curves.json", "w") as f:
        json.dump({"budgets": budgets, "curves": predicted}, f)
    with open(out_dir / "summary.json", "w") as f:
        json.dump({"threshold": thr, "n_train": int(train.sum()), "n_test": int((~train).sum()),
                   "predictors": summary, "label_controls": label_controls}, f, indent=2)


if __name__ == "__main__":
    main()
