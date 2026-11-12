"""Token allocation policies for a whole paper (planning.md, step 11).

Every policy gets a paper and a total budget B and decides a thinking cap per
question from the budget grid of step 7. Running a policy needs no generation:
the outcome of question q at cap b in run s is read from the stored truncation
results of sample s (see Table). A paper is run once per sample and the scores
are averaged.

Two kinds of policy:
  plans       fix every cap before answering; the caps must sum to at most B.
              P0 equal, P2 tier ratio, P3 predicted knapsack, P* oracle.
  sequential  decide as they go and pay the thinking actually spent.
              P1 in order, P4 two-pass triage.
  rollover    a plan whose unused tokens move on to later questions
              ("<plan>+roll", see run_rollover).

B limits thinking tokens. The answer written after </think> is reported in
`tokens` but not charged, as it is short and the same for every policy.
"""
import collections
import random
import zlib

import numpy as np

from .grader import is_equiv
from .records import read_jsonl
from .config import TIERS, resolve


Outcome = collections.namedtuple("Outcome", "correct thinking tokens pred")


class Table:
    """outcome(q, b, s) -> Outcome from the step 7 curve files."""

    def __init__(self, cfg, tiers=TIERS):
        self.budgets = cfg["curves"]["budgets"]
        self.rows = {}
        for tier in tiers:
            for r in read_jsonl(resolve(cfg["curves"]["dir"]) / f"{tier}.jsonl"):
                self.rows[r["id"], r["budget"], r["sample_idx"]] = Outcome(
                    r["correct"], r["thinking"], r["tokens"], r["pred"])
        self.n_samples = 1 + max(s for _, _, s in self.rows)

    def outcome(self, q, b, s):
        return self.rows[q, b, s]

    def mean_acc(self, q):
        """True accuracy curve of q over the grid (the oracle's knowledge)."""
        return [np.mean([self.rows[q, b, s].correct for s in range(self.n_samples)])
                for b in self.budgets]



def floor_to_grid(x, grid):
    """Largest grid budget <= x (grid[0] is 0)."""
    return max(b for b in grid if b <= x)


def fill(targets, grid, B):
    """Round continuous target budgets down to the grid, then spend what is left
    by moving questions up one grid step, largest target first, while it fits."""
    caps = {q: floor_to_grid(t, grid) for q, t in targets.items()}
    left = B - sum(caps.values())
    for q in sorted(targets, key=targets.get, reverse=True):
        i = grid.index(caps[q])
        while i + 1 < len(grid) and grid[i + 1] - caps[q] <= left and grid[i + 1] <= targets[q] * 2:
            left -= grid[i + 1] - caps[q]
            caps[q] = grid[i + 1]
            i += 1
    return caps


def knapsack(values, grid, B, unit):
    """Multiple-choice knapsack: pick one grid budget per question to maximise the
    summed value with summed budget <= B. values[q] lists the value at each grid
    budget. Exact DP over budget in steps of `unit` (every grid budget is a multiple)."""
    if any(b % unit for b in grid):
        raise ValueError(f"grid {grid} is not a multiple of {unit}")
    cap_units = int(B // unit)
    w = [b // unit for b in grid]
    qs = list(values)
    best = np.zeros(cap_units + 1)
    choice = np.zeros((len(qs), cap_units + 1), dtype=int)
    for k, q in enumerate(qs):
        new = np.full(cap_units + 1, -np.inf)
        for j, (wj, vj) in enumerate(zip(w, values[q])):
            if wj > cap_units:
                continue
            cand = np.full(cap_units + 1, -np.inf)
            cand[wj:] = best[:cap_units + 1 - wj] + vj
            better = cand > new
            new[better], choice[k][better] = cand[better], j
        best = new
    caps, c = {}, int(np.argmax(best))
    for k in range(len(qs) - 1, -1, -1):
        j = choice[k][c]
        caps[qs[k]] = grid[j]
        c -= w[j]
    return caps


def result(q, cap, o, extra_thinking=0, extra_tokens=0):
    """One question's line in a run: its cap, correctness and tokens spent.
    `check` is the part of `thinking` spent outside the answering trace (P4's
    second pass-1 attempt), so valid.py can keep it apart."""
    return {"id": q["id"], "tier": q["tier"], "marks": q["marks"], "cap": cap,
            "correct": o.correct, "thinking": o.thinking + extra_thinking,
            "tokens": o.tokens + extra_tokens, "check": extra_thinking}


def run_plan(table, paper, caps, s):
    """Answer every question at its planned cap with sample s."""
    return [result(q, caps[q["id"]], table.outcome(q["id"], caps[q["id"]], s))
            for q in paper["questions"]]


def run_rollover(table, paper, caps, s):
    """Answer in increasing order of planned cap; tokens a question leaves unused
    are banked and added to the next question's cap (rounded down to the grid).
    Spending never exceeds the planned total, since only unused tokens move on."""
    grid, bank, out = table.budgets, 0, []
    for q in sorted(paper["questions"], key=lambda q: caps[q["id"]]):
        planned = caps[q["id"]]
        cap = floor_to_grid(min(planned + bank, grid[-1]), grid)
        o = table.outcome(q["id"], cap, s)
        bank += planned - o.thinking
        out.append(result(q, cap, o))
    return out



def p0_equal(ctx, paper, B):
    """B/n per question, rounded to the grid with `fill` like P2 (so P0 is P2 at 1:1:1).
    Flooring alone would leave most of B unspent (at B = 2k every cap would be 0).
    Ties in `fill` go by printed order, so no tier is favoured."""
    n = len(paper["questions"])
    return fill({q["id"]: B / n for q in paper_order(paper)}, ctx.grid, B)


def p2_ratio(ctx, paper, B, ratio=None):
    ratio = ratio or ctx.ratio
    total = sum(ratio[q["tier"]] for q in paper["questions"])
    return fill({q["id"]: B * ratio[q["tier"]] / total for q in paper["questions"]}, ctx.grid, B)


def knapsack_plan(ctx, paper, B, curves, overhead=0.0):
    """Maximise expected marks under `curves` (q -> accuracy per grid budget).
    `overhead` tokens per question (a predictor's cost) come off the budget first."""
    B_left = B - overhead * len(paper["questions"])
    if B_left < 0:
        return {q["id"]: 0 for q in paper["questions"]}
    values = {q["id"]: [q["marks"] * p for p in curves[q["id"]]] for q in paper["questions"]}
    return knapsack(values, ctx.grid, B_left, ctx.unit)


def shadow_price(values, grid, B, iters=60):
    """CLEAR-style allocation (Wan et al. 2026): one token price lam for the whole paper;
    each question takes argmax_b values[q][b] - lam * b on its own (questions not worth
    their price get 0, i.e. are abandoned). lam is the smallest price whose total
    spend fits in B, found by bisection; leftover budget then buys single grid steps
    in order of value gained per token (to any higher grid budget that fits, since
    curves are often flat over several steps). Unlike the exact knapsack it never couples
    questions, which is what makes it cheap and online-friendly."""
    g = np.asarray(grid, dtype=float)
    V = {q: np.asarray(v, dtype=float) for q, v in values.items()}

    def pick(lam):
        return {q: int(np.argmax(v - lam * g)) for q, v in V.items()}

    def spend(c):
        return sum(grid[j] for j in c.values())

    lo, hi = 0.0, max(float(v.max()) for v in V.values()) / max(grid[1], 1) + 1.0
    if spend(pick(lo)) <= B:
        choice = pick(lo)
    else:
        for _ in range(iters):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if spend(pick(mid)) > B else (lo, mid)
        choice = pick(hi)
    left = B - spend(choice)
    while True:
        best = None
        for q, j in choice.items():
            for j2 in range(j + 1, len(grid)):
                cost = grid[j2] - grid[j]
                if cost > left:
                    break
                gain = (V[q][j2] - V[q][j]) / cost
                if gain > 0 and (best is None or gain > best[0]):
                    best = (gain, q, j2)
        if best is None:
            break
        _, q, j2 = best
        left -= grid[j2] - grid[choice[q]]
        choice[q] = j2
    return {q: grid[j] for q, j in choice.items()}


def p7_shadow(ctx, paper, B):
    """Shadow-price allocation on the predicted curves (predictor cost comes off B)."""
    B_left = B - ctx.predictor_cost * len(paper["questions"])
    if B_left < 0:
        return {q["id"]: 0 for q in paper["questions"]}
    values = {q["id"]: [q["marks"] * p for p in ctx.predicted[q["id"]]] for q in paper["questions"]}
    return shadow_price(values, ctx.grid, B_left)


def p3_knapsack(ctx, paper, B):
    return knapsack_plan(ctx, paper, B, ctx.predicted, ctx.predictor_cost)


def bandit(ctx, paper, B):
    """Knapsack on the curves the step-13 bandit learned."""
    return knapsack_plan(ctx, paper, B, ctx.bandit, ctx.predictor_cost)


def p6_icl_paper(ctx, paper, B):
    """Caps the model wrote for the whole paper after k labelled exemplars (tok_exam.icl)."""
    return ctx.icl_caps[paper["paper_id"], B]


def oracle(ctx, paper, B):
    return knapsack_plan(ctx, paper, B, {q["id"]: ctx.table.mean_acc(q["id"])
                                         for q in paper["questions"]})



def paper_order(paper):
    """The order questions are printed in: a fixed shuffle per paper."""
    qs = list(paper["questions"])
    random.Random(zlib.crc32(paper["paper_id"].encode())).shuffle(qs)
    return qs


def p1_in_order(ctx, paper, B, s):
    """Answer in printed order with as much thinking budget as is left (up to the
    largest grid budget); once it is gone, the rest are answered with no thinking."""
    left, out = B, []
    for q in paper_order(paper):
        cap = floor_to_grid(min(left, ctx.grid[-1]), ctx.grid)
        o = ctx.table.outcome(q["id"], cap, s)
        left -= o.thinking
        out.append(result(q, cap, o))
    return out


def p4_triage(ctx, paper, B, s, require_closed=False):
    """Pass 1: two short attempts (samples s and s+1) at cap b1 per question; if
    their answers agree, keep it. Pass 2: knapsack with the predicted curves over
    the unresolved questions, using what is left. Sample s continues from its own
    pass-1 prefix, so for it a pass-2 cap b costs thinking(b) in total; the second
    attempt's pass-1 tokens are the price of the check.
    require_closed (P4_triage_closed): agreement only counts if both attempts closed
    their thinking by themselves within b1 (thinking < b1). Forced answers of traces
    cut at b1 agree often while wrong, so they always go to pass 2.
    The predictor's per-question cost comes off B first, as for P3."""
    n = len(paper["questions"])
    B = B - ctx.predictor_cost * n
    if B < 0:
        return run_plan(ctx.table, paper, {q["id"]: 0 for q in paper["questions"]}, s)
    s2 = (s + 1) % ctx.table.n_samples
    b1 = floor_to_grid(min(ctx.triage_budget, B / (2 * n)), ctx.grid)
    if b1 == 0:
        return run_plan(ctx.table, paper, knapsack_plan(ctx, paper, B, ctx.predicted), s)
    first, second, unresolved = {}, {}, []
    for q in paper["questions"]:
        first[q["id"]] = ctx.table.outcome(q["id"], b1, s)
        second[q["id"]] = ctx.table.outcome(q["id"], b1, s2)
        p1, p2 = first[q["id"]].pred, second[q["id"]].pred
        cut = require_closed and (first[q["id"]].thinking >= b1 or second[q["id"]].thinking >= b1)
        if p1 is None or cut or not is_equiv(p1, p2):
            unresolved.append(q)
    spent = sum(o.thinking for o in first.values()) + sum(o.thinking for o in second.values())
    caps = {q["id"]: b1 for q in paper["questions"]}
    if unresolved:
        refund = sum(first[q["id"]].thinking for q in unresolved)
        sub = {"paper_id": paper["paper_id"], "questions": unresolved}
        floor = ctx.grid.index(b1)
        curves = {q["id"]: [-1e6 if j < floor else p for j, p in enumerate(ctx.predicted[q["id"]])]
                  for q in unresolved}
        plan = knapsack_plan(ctx, sub, max(B - spent + refund, 0), curves)
        caps.update({q: max(c, b1) for q, c in plan.items()})
    out = []
    for q in paper["questions"]:
        o2 = second[q["id"]]
        out.append(result(q, caps[q["id"]], ctx.table.outcome(q["id"], caps[q["id"]], s),
                          o2.thinking, o2.tokens))
    return out


def oracle_hindsight(ctx, paper, B, s):
    """Upper bound for every policy here, sequential ones included: knows how each
    question turns out at every cap in run s and pays the thinking actually spent
    (rounded up to the knapsack unit, so the plan is always feasible). The plain
    `oracle` is only the best *plan*: it pays its caps, not what it spends."""
    unit, grid, qs = ctx.unit, ctx.grid, paper["questions"]
    cost = lambda q, b: -(-ctx.table.outcome(q["id"], b, s).thinking // unit)
    cap_units = int(B // unit)
    best = np.zeros(cap_units + 1)
    choice = np.zeros((len(qs), cap_units + 1), dtype=int)
    for k, q in enumerate(qs):
        new = np.full(cap_units + 1, -np.inf)
        for j, b in enumerate(grid):
            wj = cost(q, b)
            if wj > cap_units:
                continue
            v = q["marks"] * ctx.table.outcome(q["id"], b, s).correct - 1e-9 * j
            cand = np.full(cap_units + 1, -np.inf)
            cand[wj:] = best[:cap_units + 1 - wj] + v
            better = cand > new
            new[better], choice[k][better] = cand[better], j
        best = new
    caps, c = {}, int(np.argmax(best))
    for k in range(len(qs) - 1, -1, -1):
        j = choice[k][c]
        caps[qs[k]["id"]] = grid[j]
        c -= cost(qs[k], grid[j])
    return run_plan(ctx.table, paper, caps, s)



class Context:
    """What policies may know: the grid, predicted curves, tuned tier ratio.
    Only `oracle` reads the true outcomes through table.mean_acc."""

    def __init__(self, cfg, table, predicted=None, predictor_cost=0.0, ratio=None, bandit=None,
                 icl_caps=None, predictors=None):
        pc = cfg["policies"]
        self.table, self.grid = table, table.budgets
        self.unit = pc["knapsack_unit"]
        self.triage_budget = pc["triage_budget"]
        self.predicted, self.predictor_cost = predicted, predictor_cost
        self.ratio = ratio or pc["tier_ratio"]
        self.bandit = bandit
        self.icl_caps = icl_caps
        self.predictors = predictors or {}


PLANS = {"P0_equal": p0_equal, "P2_ratio": p2_ratio, "P3_knapsack": p3_knapsack,
         "bandit": bandit, "P6_icl_paper": p6_icl_paper, "P7_shadow": p7_shadow,
         "oracle": oracle}
SEQUENTIAL = {"P1_in_order": p1_in_order, "P4_triage": p4_triage,
              "P4_triage_closed": lambda ctx, paper, B, s: p4_triage(ctx, paper, B, s, True),
              "oracle_hindsight": oracle_hindsight}
ROLL = "+roll"


AT = "@"


def with_predictor(ctx, pred):
    """A copy of ctx whose predicted curves and cost are those of predictor `pred`."""
    import copy

    if not ctx.predictors or pred not in ctx.predictors:
        raise KeyError(f"no curves for predictor {pred!r}")
    out = copy.copy(ctx)
    out.predicted, out.predictor_cost = ctx.predictors[pred]
    return out


def run_policy(name, ctx, paper, B, s):
    if AT in name:
        name, pred = name.split(AT, 1)
        ctx = with_predictor(ctx, pred)
    base = name.removesuffix(ROLL)
    if base in PLANS:
        caps = PLANS[base](ctx, paper, B)
        if sum(caps.values()) > B:
            raise AssertionError(f"{base} planned {sum(caps.values())} > {B}")
        run = run_rollover if name.endswith(ROLL) else run_plan
        return run(ctx.table, paper, caps, s)
    return SEQUENTIAL[name](ctx, paper, B, s)


def score(results):
    """Marks scored, thinking and total tokens spent, and thinking per tier, for one run."""
    per_tier = collections.Counter()
    for r in results:
        per_tier[r["tier"]] += r["thinking"]
    return {"marks": sum(r["marks"] * r["correct"] for r in results),
            "thinking": sum(r["thinking"] for r in results),
            "tokens": sum(r["tokens"] for r in results), "thinking_by_tier": dict(per_tier)}
