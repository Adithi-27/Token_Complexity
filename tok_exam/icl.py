"""In-context learning of thinking budgets (the professor's lead experiment).

The model is shown k solved *train* questions, each labelled with the thinking
budget it needed, and asked for the budget of a new question. Thinking is
skipped, so a prediction costs one generated token (the prompt is reported as
prefill). Two uses:

question  per question, the budget options are letters A..J (grid budgets, then
          "never"); the probabilities of those letters give a distribution over
          the needed bin. Its cumulative sum is the predicted curve P(needed <= b),
          used raw ("icl_raw") or as features of a fitted predictor ("icl") in
          tok_exam.difficulty, and then by the knapsack (P3).
paper     the whole paper, its marks and total budget B in one prompt, after k
          exemplars; the model writes "Q<k>: <tokens>" per question. The caps are
          snapped to the grid and shrunk to fit B: policy "P6_icl_paper".

Exemplars come from train questions only and never include the question itself.
Selection: "retrieved" (MiniLM nearest neighbours), "random" (seeded), or
"shuffled" (retrieved exemplars whose labels are permuted among train questions:
the control for whether the model reads the labels at all).
Labels (needed bin per train question): from the step-9 needed.jsonl when the
curves exist, otherwise from the full-budget traces (the second-shortest correct
trace, rounded up to the grid; "never" if fewer than half are correct).

Writes <icl.dir>/question_{sel}_k{k}.jsonl  id, probs, expected_bin, label_source,
                                             prompt_tokens, cost_tokens
       <icl.dir>/paper_{sel}_k{k}_{split}.jsonl  paper_id, budget, raw, caps, prompt_tokens

Run:  python -m tok_exam.icl [--mode question paper] [--k 0 2 4 8 16] [--selection retrieved random]
"""
import argparse
import math
import random
import re

import numpy as np

from .config import TIERS, load_config, resolve
from .policies import fill, floor_to_grid
from .records import read_jsonl, write_jsonl

LETTERS = "ABCDEFGHIJKLMNOP"
_ALLOC = re.compile(r"Q\s*(\d+)\s*[:=\-]\s*\**\s*([\d,]+)")


# ---- labels and exemplars --------------------------------------------------

def bin_names(grid):
    return [str(b) for b in grid] + ["never"]


def labels_from_traces(rows, grid, threshold):
    """{id: needed bin} from full-budget traces: the budget by which a fraction
    `threshold` of samples had finished with a correct answer."""
    by_q = {}
    for r in rows:
        by_q.setdefault(r["id"], []).append(r)
    out = {}
    for q, rs in by_q.items():
        need = math.ceil(threshold * len(rs))
        lens = sorted(r["thinking_tokens"] for r in rs if r["correct"])
        if len(lens) < need:
            out[q] = len(grid)
            continue
        t = lens[need - 1]
        out[q] = next((j for j, b in enumerate(grid) if b >= t), len(grid))
    return out


def load_labels(cfg):
    """({id: needed bin}, source name)."""
    grid = cfg["curves"]["budgets"]
    needed = resolve(cfg["difficulty"]["dir"]) / "needed.jsonl"
    if needed.exists() and cfg["icl"]["label_source"] in ("auto", "curves"):
        return {r["id"]: r["needed_bin"] for r in read_jsonl(needed)}, "curves"
    if cfg["icl"]["label_source"] == "curves":
        raise FileNotFoundError(f"{needed} missing: run tok_exam.difficulty first")
    rows = [r for t in TIERS for r in read_jsonl(resolve(cfg["output"]["dir"]) / f"{t}.jsonl")]
    return labels_from_traces(rows, grid, cfg["difficulty"]["threshold"]), "full_budget"


class Exemplars:
    """Chooses k train exemplars for a question (never the question itself)."""

    def __init__(self, pool, labels, selection, seed, embed=None):
        self.train = [it for it in pool if it["split"] == "train" and it["id"] in labels]
        self.labels, self.selection, self.seed = labels, selection, seed
        if selection == "retrieved":
            idx = {it["id"]: i for i, it in enumerate(pool)}
            self.E = embed / np.linalg.norm(embed, axis=1, keepdims=True)
            self.row = idx
            self.train_rows = np.array([idx[it["id"]] for it in self.train])

    def pick(self, item, k):
        if k == 0:
            return []
        if self.selection == "random":
            rng = random.Random(f"{self.seed}-{item['id']}")
            cands = [it for it in self.train if it["id"] != item["id"]]
            return rng.sample(cands, k)
        sims = self.E[self.train_rows] @ self.E[self.row[item["id"]]]
        order = [i for i in np.argsort(-sims) if self.train[i]["id"] != item["id"]]
        # most similar last, nearest to the question being asked
        return [self.train[i] for i in order[:k]][::-1]


def shuffled_labels(labels, pool, seed):
    """Train labels permuted among train questions (test labels untouched)."""
    ids = sorted(q for q in labels if pool[q]["split"] == "train")
    perm = ids[:]
    random.Random(seed).shuffle(perm)
    return {**labels, **{a: labels[b] for a, b in zip(ids, perm)}}


# ---- prompts ---------------------------------------------------------------

def options_text(grid):
    names = bin_names(grid)
    return "\n".join(f"{LETTERS[j]}) {n}" for j, n in enumerate(names))


def question_message(cfg, item, exemplars, labels, grid):
    names = bin_names(grid)
    parts = [cfg["icl"]["question_header"], "Options:\n" + options_text(grid)]
    for ex in exemplars:
        j = labels[ex["id"]]
        parts.append(f"Problem: {ex['question'].strip()}\nThinking tokens needed: "
                     f"{LETTERS[j]}) {names[j]}")
    parts.append(f"Problem: {item['question'].strip()}\nThinking tokens needed:")
    return "\n\n".join(parts)


def paper_message(cfg, paper, pool, exemplars, labels, grid, B):
    names = bin_names(grid)
    parts = [cfg["icl"]["paper_header"].format(n=len(paper["questions"]), budget=B,
                                               grid=", ".join(map(str, grid)))]
    if exemplars:
        parts.append("Solved examples, with the thinking tokens each one needed:")
        parts += [f"Example ({ex['tier']}): {ex['question'].strip()}\nNeeded: {names[labels[ex['id']]]}"
                  for ex in exemplars]
    parts.append("The exam paper:")
    parts += [f"Q{k} ({q['marks']} mark{'s' if q['marks'] > 1 else ''}). "
              f"{pool[q['id']]['question'].strip()}" for k, q in enumerate(paper["questions"], 1)]
    parts.append(cfg["icl"]["paper_footer"].format(n=len(paper["questions"]), budget=B))
    return "\n\n".join(parts)


def paper_exemplars(ex, paper, pool, k):
    """Up to k exemplars for a paper: the nearest neighbour of each question in turn."""
    out, seen = [], set()
    per_q = [ex.pick(pool[q["id"]], k) for q in paper["questions"]]
    for depth in range(k):
        for lst in per_q:
            if depth < len(lst) and lst[-1 - depth]["id"] not in seen and len(out) < k:
                seen.add(lst[-1 - depth]["id"])
                out.append(lst[-1 - depth])
    return out


# ---- parsing ---------------------------------------------------------------

def letter_probs(top_logprobs, letter_ids):
    """Normalised probabilities of the option letters from one position's top-k.
    letter_ids[j] lists every token id that spells option j ("A" and " A")."""
    mass = np.array([sum(math.exp(top_logprobs[i].logprob) for i in ids if i in top_logprobs)
                     for ids in letter_ids])
    z = mass.sum()
    return (mass / z if z > 0 else np.full(len(letter_ids), 1 / len(letter_ids))), float(z)


def curve_from_probs(probs):
    """P(needed bin <= j) for each grid budget j (the "never" bin is dropped)."""
    return np.cumsum(probs)[:-1]


def parse_allocation(text, n):
    """{k: tokens} from "Q<k>: <tokens>" lines."""
    out = {}
    for m in _ALLOC.finditer(text or ""):
        k = int(m.group(1))
        if 1 <= k <= n and k not in out:
            out[k] = int(m.group(2).replace(",", ""))
    return out


def fit_caps(paper, alloc, grid, B):
    """Grid caps from a model's allocation: missing questions get 0, the rest are
    scaled down if they exceed B, then rounded to the grid without exceeding B."""
    raw = {q["id"]: max(alloc.get(k, 0), 0) for k, q in enumerate(paper["questions"], 1)}
    total = sum(raw.values())
    scale = min(1.0, B / total) if total else 0.0
    targets = {q: min(v * scale, grid[-1]) for q, v in raw.items()}
    caps = fill(targets, grid, B) if total else {q: 0 for q in raw}
    assert sum(caps.values()) <= B
    return caps


def load_embeddings(cfg, pool_rows, path):
    """MiniLM embeddings of the pool, cached: the GPU environment pins an older
    transformers, so run `python -m tok_exam.icl --embed-only` on CPU first."""
    ids = [r["id"] for r in pool_rows]
    if path.exists():
        z = np.load(path, allow_pickle=False)
        if list(z["ids"]) == ids:
            return z["E"]
    from .difficulty import embed_features
    E = embed_features(pool_rows, cfg["difficulty"]["embed_model"])
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, ids=np.array(ids), E=E)
    return E


# ---- GPU stages ------------------------------------------------------------

def chat(tok, message):
    from .probe import skip_thinking

    return skip_thinking(tok.apply_chat_template([{"role": "user", "content": message}],
                                                 tokenize=False, add_generation_prompt=True))


def fitting_prompt(tok, build, exemplars, limit):
    """Chat prompt from build(exemplars), dropping the least similar exemplars (the
    first ones) until it fits in `limit` tokens. Returns (prompt, shots used)."""
    exemplars = list(exemplars)
    while True:
        prompt = chat(tok, build(exemplars))
        if len(tok.encode(prompt, add_special_tokens=False)) <= limit or not exemplars:
            return prompt, len(exemplars)
        exemplars = exemplars[1:]


def run_questions(cfg, llm, pool, ex, labels, k, grid):
    from vllm import SamplingParams

    tok = llm.get_tokenizer()
    n_opt = len(grid) + 1
    letter_ids = []
    for c in LETTERS[:n_opt]:
        ids = {tuple(tok.encode(v, add_special_tokens=False)) for v in (c, " " + c)}
        if any(len(i) != 1 for i in ids):
            raise ValueError(f"option letter {c} is not one token: {ids}")
        letter_ids.append(sorted(i[0] for i in ids))
    limit = cfg["icl"]["max_model_len"] - 16
    built = [fitting_prompt(tok, lambda e, it=it: question_message(cfg, it, e, labels, grid),
                            ex.pick(it, k), limit) for it in pool]
    prompts = [p for p, _ in built]
    outs = llm.generate(prompts, SamplingParams(temperature=0, max_tokens=1, logprobs=20))
    rows = []
    for it, (pr, shots), out in zip(pool, built, outs):
        probs, mass = letter_probs(out.outputs[0].logprobs[0], letter_ids)
        rows.append({"id": it["id"], "probs": [round(float(p), 5) for p in probs],
                     "letter_mass": mass, "expected_bin": float(probs @ np.arange(n_opt)),
                     "shots": shots, "prompt_tokens": len(out.prompt_token_ids),
                     "cost_tokens": 1})
    return rows


def run_papers(cfg, llm, pool, papers, ex, labels, k, grid, budgets):
    from vllm import SamplingParams

    tok = llm.get_tokenizer()
    jobs = [(p, B) for p in papers for B in budgets]
    n_out = cfg["icl"]["paper_answer_tokens"]
    limit = cfg["icl"]["max_model_len"] - n_out
    built = [fitting_prompt(tok, lambda e, p=p, B=B: paper_message(cfg, p, pool, e, labels, grid, B),
                            paper_exemplars(ex, p, pool, k), limit) for p, B in jobs]
    outs = llm.generate([pr for pr, _ in built], SamplingParams(temperature=0, max_tokens=n_out))
    rows = []
    for (p, B), (_, shots), out in zip(jobs, built, outs):
        text = out.outputs[0].text
        alloc = parse_allocation(text, len(p["questions"]))
        rows.append({"paper_id": p["paper_id"], "budget": B, "shots": shots, "raw": text,
                     "parsed": len(alloc),
                     "caps": fit_caps(p, alloc, grid, B),
                     "prompt_tokens": len(out.prompt_token_ids),
                     "cost_tokens": len(out.outputs[0].token_ids)})
    return rows


def main():
    cfg = load_config()
    ic = cfg["icl"]
    p = argparse.ArgumentParser()
    p.add_argument("--mode", nargs="+", default=["question", "paper"], choices=["question", "paper"])
    p.add_argument("--k", nargs="+", type=int, default=ic["shots"])
    p.add_argument("--selection", nargs="+", default=ic["selections"],
                   choices=["retrieved", "random", "shuffled"])
    p.add_argument("--split", default="test", choices=["train", "test"])
    p.add_argument("--embed-only", action="store_true", help="cache embeddings and exit (CPU)")
    args = p.parse_args()

    grid = cfg["curves"]["budgets"]
    if args.embed_only:
        rows = read_jsonl(resolve(cfg["data"]["processed_dir"]) / "pool.jsonl")
        E = load_embeddings(cfg, rows, resolve(ic["dir"]) / "embeddings.npz")
        print(f"cached {E.shape} embeddings")
        return
    pool_rows = read_jsonl(resolve(cfg["data"]["processed_dir"]) / "pool.jsonl")
    pool = {r["id"]: r for r in pool_rows}
    labels, source = load_labels(cfg)
    print(f"labels from {source}: {len(labels)} questions", flush=True)
    embed = None
    out_dir = resolve(ic["dir"])
    if {"retrieved", "shuffled"} & set(args.selection):
        embed = load_embeddings(cfg, pool_rows, out_dir / "embeddings.npz")

    from .llm import load_llm
    llm = load_llm(cfg, max_model_len=ic["max_model_len"], enable_prefix_caching=True)
    papers = read_jsonl(resolve(cfg["papers"]["dir"]) / f"{args.split}.jsonl")[:ic["n_papers"]]
    for sel in args.selection:
        ex = Exemplars(pool_rows, labels, "retrieved" if sel == "shuffled" else sel,
                       ic["seed"], embed)
        shown = shuffled_labels(labels, pool, ic["seed"]) if sel == "shuffled" else labels
        for k in args.k:
            if sel != "retrieved" and k == 0 and "retrieved" in args.selection:
                continue  # k = 0 is the same prompt for every selection
            if "question" in args.mode:
                rows = run_questions(cfg, llm, pool_rows, ex, shown, k, grid)
                for r in rows:
                    r["label_source"] = source
                write_jsonl(out_dir / f"question_{sel}_k{k}.jsonl", rows)
                test = [r for r in rows if pool[r["id"]]["split"] == "test"]
                true = np.array([labels.get(r["id"], -1) for r in test])
                pred = np.array([np.argmax(r["probs"]) for r in test])
                ok = true >= 0
                print(f"question {sel} k={k:2d}: test bin acc {np.mean(pred[ok] == true[ok]):.3f}"
                      f"  ±1 {np.mean(abs(pred[ok] - true[ok]) <= 1):.3f}  "
                      f"mean prompt tokens {np.mean([r['prompt_tokens'] for r in rows]):.0f}",
                      flush=True)
            if "paper" in args.mode:
                rows = run_papers(cfg, llm, pool, papers, ex, shown, k, grid,
                                  cfg["policies"]["total_budgets"])
                write_jsonl(out_dir / f"paper_{sel}_k{k}_{args.split}.jsonl", rows)
                print(f"paper {sel} k={k:2d}: parsed {np.mean([r['parsed'] for r in rows]):.1f}"
                      f"/{len(papers[0]['questions'])} questions per paper", flush=True)


if __name__ == "__main__":
    main()
