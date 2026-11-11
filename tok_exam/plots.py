"""Figures for the report (planning.md, step 15). Each figure is skipped if its
inputs do not exist yet.

  accuracy_by_budget.png  mean accuracy vs thinking budget, one line per tier (step 7)
  needed_by_tier.png      share of questions per needed budget, per tier (step 9)
  score_vs_budget.png     paper score vs total budget, one line per policy (step 12)
  score_vs_budget_rollover.png  the same with the rollover variants (dotted)
  tier_spend.png          thinking spent per tier by each policy at one budget (step 12)
  lambda_sweep.png        score vs thinking for the token-price rule (step 13)

Run:  python -m tok_exam.plots [--split test] [--spend-budget 16384]
"""
import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .config import TIERS, load_config, resolve  # noqa: E402
from .records import read_jsonl  # noqa: E402

# reference palette (dataviz skill), light mode, slots in fixed order
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
TEXT, TEXT_2, GRID_C, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
REFERENCE = "#52514e"   # oracle / upper bound: neutral, dashed
# colour follows the entity: each policy and tier keeps its slot in every figure
POLICY_COLOR = {"P0_equal": SERIES[0], "P1_in_order": SERIES[1], "P2_ratio": SERIES[2],
                "P2_ratio_tuned": SERIES[3], "P3_knapsack": SERIES[4], "P4_triage": SERIES[5],
                "P4_triage_closed": "#5c8a00",
                "bandit": SERIES[6], "P5_self": SERIES[7], "P6_icl_paper": "#1f8fa0", "P7_shadow": "#8a5a2b",
                "oracle": REFERENCE,
                "oracle_hindsight": REFERENCE}
POLICY_LABEL = {"P0_equal": "P0 equal share", "P1_in_order": "P1 in order",
                "P2_ratio": "P2 tier ratio 1:2:4", "P2_ratio_tuned": "P2 tier ratio (tuned)",
                "P3_knapsack": "P3 predicted knapsack", "P4_triage": "P4 two-pass triage",
                "P4_triage_closed": "P4 triage, closed answers only",
                "bandit": "Bandit knapsack", "P5_self": "P5 self-allocation", "P6_icl_paper": "P6 ICL paper allocation",
                "P7_shadow": "P7 shadow price (CLEAR-style)",
                "oracle": "Oracle (best plan)",
                "oracle_hindsight": "Hindsight oracle (upper bound)"}
TIER_COLOR = dict(zip(TIERS, SERIES[:3]))
ROLL = "+roll"   # rollover variant: same colour as its plan, dotted line


def policy_style(name):
    name, _, pred = name.partition("@")
    base = name.removesuffix(ROLL)
    label = (POLICY_LABEL[base] + (" + rollover" if name.endswith(ROLL) else "")
             + (f" [{pred}]" if pred else ""))
    dash = (":" if name.endswith(ROLL) else "--" if base == "oracle"
            else "-." if base == "oracle_hindsight" else "-")
    return POLICY_COLOR[base], label, dash


def style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID_C, "axes.labelcolor": TEXT_2, "text.color": TEXT,
        "xtick.color": TEXT_2, "ytick.color": TEXT_2, "axes.grid": True, "grid.color": GRID_C,
        "grid.linewidth": 0.8, "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
        "lines.linewidth": 2, "lines.markersize": 6, "font.size": 10, "legend.frameon": False,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left"})


def budget_axis(ax, budgets, label):
    ax.set_xscale("symlog", base=2, linthresh=128)
    ax.set_xticks(budgets)
    ax.set_xticklabels([f"{b // 1024}k" if b >= 1024 else str(b) for b in budgets])
    ax.set_xlabel(label)
    ax.minorticks_off()


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print("wrote", path)


def accuracy_by_budget(cfg, out):
    from .curves import load_curves
    try:
        curves = load_curves(cfg)
    except FileNotFoundError:
        return
    pool = read_jsonl(resolve(cfg["data"]["processed_dir"]) / "pool.jsonl")
    budgets = cfg["curves"]["budgets"]
    fig, ax = plt.subplots(figsize=(6.4, 4))
    for t in TIERS:
        ids = [q["id"] for q in pool if q["tier"] == t and q["id"] in curves]
        acc = [100 * np.mean([curves[q][b][0] for q in ids]) for b in budgets]
        ax.plot(budgets, acc, marker="o", color=TIER_COLOR[t], label=f"{t} ({len(ids)})")
    budget_axis(ax, budgets, "Thinking budget per question (tokens)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Accuracy rises with budget, at a different rate per tier")
    ax.legend(loc="lower right")
    save(fig, out / "accuracy_by_budget.png")


def needed_by_tier(cfg, out):
    path = resolve(cfg["difficulty"]["dir"]) / "needed.jsonl"
    if not path.exists():
        return
    rows = read_jsonl(path)
    budgets = cfg["curves"]["budgets"]
    labels = [f"{b // 1024}k" if b >= 1024 else str(b) for b in budgets] + ["never"]
    fig, ax = plt.subplots(figsize=(7.2, 4))
    width = 0.27
    x = np.arange(len(labels))
    for k, t in enumerate(TIERS):
        bins = [r["needed_bin"] for r in rows if r["tier"] == t]
        share = 100 * np.bincount(bins, minlength=len(labels)) / len(bins)
        ax.bar(x + (k - 1) * width, share, width, color=TIER_COLOR[t], label=t,
               edgecolor=SURFACE, linewidth=1)
    ax.set_xticks(x, labels)
    ax.set_xlabel(f"Needed budget (first with accuracy ≥ {cfg['difficulty']['threshold']})")
    ax.set_ylabel("Questions in tier (%)")
    ax.set_title("Dataset tiers overlap in how much thinking they need")
    ax.grid(axis="x", visible=False)
    ax.legend()
    save(fig, out / "needed_by_tier.png")


def score_vs_budget(cfg, out, split, show_roll=False, name="score_vs_budget.png"):
    path = resolve(cfg["policies"]["dir"]) / f"summary_{split}.json"
    if not path.exists():
        return
    with open(path) as f:
        summ = json.load(f)
    budgets = summ["budgets"]
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    series = dict(summ["policies"])
    p5 = resolve(cfg["selfalloc"]["dir"]) / f"{split}.jsonl"
    if p5.exists():
        rows = read_jsonl(p5)
        series["P5_self"] = {str(B): {"score_mean": np.mean([r["score"] for r in rows
                                                             if r["budget"] == B])}
                             for B in sorted({r["budget"] for r in rows})}
    for pol, by_b in series.items():
        if "@" in pol:          # predictor variants have their own figure
            continue
        if pol.endswith(ROLL) and not show_roll:
            continue
        # the rollover figure pairs each plan with its rollover twin, nothing else
        if show_roll and not pol.endswith(ROLL) and pol + ROLL not in series:
            continue
        bs = [int(b) for b in by_b]
        ys = [by_b[b]["score_mean"] for b in by_b]
        color, label, dash = policy_style(pol)
        ax.plot(bs, ys, marker="o", color=color, label=label, linestyle=dash)
    budget_axis(ax, budgets, "Total thinking budget for the paper (tokens)")
    ax.set_ylabel("Paper score (% of marks)")
    ax.set_ylim(0, 100)
    ax.set_title("Score vs total budget, by allocation policy")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8.5)
    save(fig, out / name)


def predictor_knapsack(cfg, out, split, policy="P3_knapsack+roll"):
    """Score vs budget of one policy run with each predictor's curves (the ICL comparison)."""
    path = resolve(cfg["policies"]["dir"]) / f"summary_{split}.json"
    if not path.exists():
        return
    with open(path) as f:
        summ = json.load(f)
    runs = {n.split("@", 1)[1]: v for n, v in summ["policies"].items()
            if n.startswith(policy + "@")}
    if not runs:
        return
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    for k, (pred, by_b) in enumerate(sorted(runs.items())):
        bs = [int(b) for b in by_b]
        width = 2.6 if pred.startswith("icl") else 1.4
        ax.plot(bs, [by_b[b]["score_mean"] for b in by_b], marker="o", linewidth=width,
                color=SERIES[k % len(SERIES)], label=pred)
    for ref in ("oracle_hindsight", "oracle" + ROLL):
        if ref in summ["policies"]:
            by_b = summ["policies"][ref]
            ax.plot([int(b) for b in by_b], [by_b[b]["score_mean"] for b in by_b],
                    color=REFERENCE, linestyle=policy_style(ref)[2], label=policy_style(ref)[1])
    budget_axis(ax, summ["budgets"], "Total thinking budget for the paper (tokens)")
    ax.set_ylabel("Paper score (% of marks)")
    ax.set_ylim(0, 100)
    ax.set_title(f"{policy_style(policy)[1]}: which difficulty predictor drives it")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8.5)
    save(fig, out / "predictor_knapsack.png")


def icl_shots(cfg, out, metric="spearman"):
    """Predictor quality vs number of exemplars k: ICL (raw) per selection, and the
    no-LLM kNN control on the same neighbours (the lead ICL figure)."""
    path = resolve(cfg["difficulty"]["dir"]) / "summary.json"
    if not path.exists():
        return
    with open(path) as f:
        summ = json.load(f)["predictors"]
    lines = {}
    for name, v in summ.items():
        if name.startswith("icl_raw/"):
            sel, k = name.split("/", 1)[1].rsplit("_k", 1)
            lines.setdefault(f"ICL ({sel})", {})[int(k)] = v[metric]
        elif name.startswith("knn/k"):
            lines.setdefault("kNN on the same neighbours (no LLM)", {})[int(name[5:])] = v[metric]
    if not any(n.startswith("ICL") for n in lines):
        return
    zero = next((v for (n, v) in lines.items() if n.startswith("ICL") and 0 in v), None)
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    for k, (name, pts) in enumerate(sorted(lines.items())):
        if zero is not None and name.startswith("ICL") and 0 not in pts:
            pts = {0: zero[0], **pts}          # k = 0 is one shared zero-shot run
        ks = sorted(pts)
        ax.plot(ks, [pts[x] for x in ks], marker="o", color=SERIES[k % len(SERIES)],
                linestyle="--" if name.startswith("kNN") else "-", label=name)
    if "tier" in summ:
        ax.axhline(summ["tier"][metric], color=REFERENCE, linestyle=":", linewidth=1,
                   label="dataset tier only")
    ax.set_xlabel("Exemplars k")
    ax.set_ylabel({"spearman": "Spearman with true needed budget",
                   "bin_acc": "Exact needed-budget bin accuracy"}.get(metric, metric))
    ax.set_title("Does the model learn its budgets in context?")
    ax.legend(fontsize=8.5)
    save(fig, out / f"icl_shots_{metric}.png")


def valid_spend(cfg, out, split, B):
    """Where each policy's tokens go at budget B: useful / overspend / hopeless, and
    the shortfall it leaves (tok_exam.valid)."""
    path = resolve(cfg["valid"]["dir"]) / f"policies_{split}.json"
    if not path.exists():
        return
    with open(path) as f:
        pol = json.load(f)["policies"]
    names = [n for n in pol if str(B) in pol[n]]
    if not names:
        return
    parts = [("useful", SERIES[2]), ("overspend", SERIES[3]), ("hopeless", SERIES[7])]
    fig, ax = plt.subplots(figsize=(7.6, 0.45 * len(names) + 1.6))
    left = np.zeros(len(names))
    for key, color in parts:
        vals = np.array([pol[n][str(B)].get(key, 0) for n in names])
        ax.barh(range(len(names)), vals, left=left, color=color, label=key,
                edgecolor=SURFACE, linewidth=2, height=0.6)
        left += vals
    ax.scatter([pol[n][str(B)].get("shortfall", 0) for n in names], range(len(names)),
               marker="|", s=160, color=TEXT, label="shortfall", zorder=3)
    ax.set_yticks(range(len(names)), [policy_style(n)[1] for n in names])
    ax.invert_yaxis()
    ax.axvline(B, color=REFERENCE, linestyle=":", linewidth=1)
    ax.set_xlabel(f"Thinking tokens per paper (budget {B})")
    ax.set_title("What each policy's budget buys", pad=24)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=4, fontsize=9)
    save(fig, out / "valid_spend.png")


def tier_spend(cfg, out, split, B):
    path = resolve(cfg["policies"]["dir"]) / f"summary_{split}.json"
    if not path.exists():
        return
    with open(path) as f:
        summ = json.load(f)["policies"]
    names = [n for n in summ if str(B) in summ[n] and not n.endswith(ROLL) and "@" not in n]
    fig, ax = plt.subplots(figsize=(7.2, 0.45 * len(names) + 1.4))
    left = np.zeros(len(names))
    for t in TIERS:
        vals = np.array([summ[n][str(B)]["thinking_by_tier"][t] for n in names])
        ax.barh(range(len(names)), vals, left=left, color=TIER_COLOR[t], label=t,
                edgecolor=SURFACE, linewidth=2, height=0.6)
        left += vals
    ax.set_yticks(range(len(names)), [policy_style(n)[1] for n in names])
    ax.invert_yaxis()
    ax.axvline(B, color=REFERENCE, linestyle=":", linewidth=1)
    ax.set_xlabel(f"Mean thinking tokens spent per paper (budget {B})")
    ax.set_title("Where each policy spends its tokens", pad=24)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, fontsize=9)
    save(fig, out / "tier_spend.png")


def lambda_sweep(cfg, out):
    path = resolve(cfg["bandit"]["dir"]) / "summary.json"
    if not path.exists():
        return
    with open(path) as f:
        sweep = json.load(f)["lambda_sweep"]
    colors = {"predicted": POLICY_COLOR["P3_knapsack"], "bandit": POLICY_COLOR["bandit"],
              "oracle": REFERENCE}
    labels = {"predicted": "Step-10 predicted curves", "bandit": "Bandit-learned curves",
              "oracle": "True curves (oracle)"}
    fig, ax = plt.subplots(figsize=(6.4, 4))
    for name in colors:
        pts = sorted((v[name][1], v[name][0]) for v in sweep.values())
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", color=colors[name],
                label=labels[name], linestyle="--" if name == "oracle" else "-")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Mean thinking tokens per paper")
    ax.set_ylabel("Paper score (% of marks)")
    ax.set_title("Score per token as the token price λ varies")
    ax.legend(loc="lower right")
    save(fig, out / "lambda_sweep.png")


def main():
    cfg = load_config()
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="test")
    p.add_argument("--spend-budget", type=int, default=16384)
    args = p.parse_args()
    out = resolve(cfg["plots"]["dir"])
    out.mkdir(parents=True, exist_ok=True)
    style()
    accuracy_by_budget(cfg, out)
    needed_by_tier(cfg, out)
    score_vs_budget(cfg, out, args.split)
    score_vs_budget(cfg, out, args.split, show_roll=True, name="score_vs_budget_rollover.png")
    predictor_knapsack(cfg, out, args.split)
    icl_shots(cfg, out)
    icl_shots(cfg, out, "bin_acc")
    valid_spend(cfg, out, args.split, args.spend_budget)
    tier_spend(cfg, out, args.split, args.spend_budget)
    lambda_sweep(cfg, out)


if __name__ == "__main__":
    main()
