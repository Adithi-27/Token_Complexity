"""P5 self-allocation: the model gets the whole paper and the budget (planning.md, step 11).

One prompt per (paper, B) lists every question with its marks and says the
thinking budget is B tokens for the whole paper. The model thinks once and
then writes one answer per question as "Q<k>: \\boxed{...}". If its thinking
runs past B it is cut there, `</think>` is appended, and the answers are
generated greedily. Answers are matched to questions by their "Q<k>" label.

Writes <selfalloc.dir>/{split}.jsonl, one row per (paper, budget, sample):
  paper_id, budget, sample_idx, thinking, cut, marks, total_marks, score,
  per_question: [{id, tier, marks, pred, correct}]
Rows are appended one chunk of papers at a time and runs already in the output file
are skipped, so a killed or timed-out job can be resubmitted and continues.

Run:  python -m tok_exam.selfalloc                              # all papers, one job
      python -m tok_exam.selfalloc --papers 0:50 --out test_part0.jsonl   # one slice
      python -m tok_exam.selfalloc --summarise                  # CPU: merge test_part*.jsonl
          into test.jsonl and compare with the replay policies on the same papers
"""
import argparse
import glob
import json
import re

import numpy as np

from .config import load_config, resolve, sample_seed
from .grader import is_equiv, last_boxed
from .llm import load_llm
from .papers import total_marks
from .records import THINK_CLOSE, append_jsonl, read_jsonl, write_jsonl

_LABEL = re.compile(r"(?:^|\n)\s*(?:\*\*)?Q(\d+)(?:\*\*)?\s*[:.)]", re.M)


def paper_slice(papers, spec):
    """papers[a:b] for a spec "a:b" (either side may be empty); all papers for None."""
    if not spec:
        return papers
    a, b = spec.split(":")
    return papers[int(a) if a else None:int(b) if b else None]


def pending_jobs(papers, budgets, n_samples, done):
    """(paper, B, sample) for every run whose (paper_id, B, sample) is not in `done`."""
    return [(p, B, i) for p in papers for B in budgets for i in range(n_samples)
            if (p["paper_id"], B, i) not in done]


def paired_comparison(p5_rows, run_rows, policies, n_boot=5000, seed=0):
    """P5 against each replay policy on the same papers: mean scores, the paired
    difference with a percentile bootstrap interval over papers, and papers won."""
    budgets = sorted({r["budget"] for r in p5_rows})
    papers = sorted({r["paper_id"] for r in p5_rows})
    p5 = {(p, B): float(np.mean([r["score"] for r in p5_rows if r["paper_id"] == p and r["budget"] == B]))
          for p in papers for B in budgets}
    rng = np.random.default_rng(seed)
    out = {"papers": len(papers), "budgets": budgets,
           "p5": {str(B): float(np.mean([p5[p, B] for p in papers])) for B in budgets},
           "p5_thinking": {str(B): float(np.mean([r["thinking"] for r in p5_rows if r["budget"] == B]))
                           for B in budgets},
           "p5_cut": {str(B): float(np.mean([r["cut"] for r in p5_rows if r["budget"] == B])) for B in budgets},
           "p5_parsed": {str(B): float(np.mean([q["pred"] is not None for r in p5_rows if r["budget"] == B
                                                 for q in r["per_question"]])) for B in budgets},
           "policies": {}}
    for pol in policies:
        other = {(r["paper_id"], r["budget"]): r["score"] for r in run_rows if r["policy"] == pol}
        if not all((p, B) in other for p in papers for B in budgets):
            continue
        res = {}
        for B in budgets:
            d = np.array([p5[p, B] - other[p, B] for p in papers])
            boot = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)]
            res[str(B)] = {"score": float(np.mean([other[p, B] for p in papers])),
                           "diff": float(d.mean()),
                           "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                           "p5_wins": int((d > 0).sum())}
        out["policies"][pol] = res
    return out


def paper_message(cfg, paper, pool, B):
    sc = cfg["selfalloc"]
    lines = [sc["header"].format(n=len(paper["questions"]), budget=B,
                                 marks=total_marks(paper))]
    for k, q in enumerate(paper["questions"], 1):
        lines.append(f"Q{k} ({q['marks']} mark{'s' if q['marks'] > 1 else ''}). "
                     f"{pool[q['id']]['question'].strip()}")
    lines.append(sc["footer"].format(n=len(paper["questions"])))
    return "\n\n".join(lines)


def parse_answers(text, n):
    """{k: boxed answer} for labels Q1..Qn; each label's text runs to the next label."""
    if text is None:
        return {}
    labels = list(_LABEL.finditer(text))
    out = {}
    for m, nxt in zip(labels, labels[1:] + [None]):
        k = int(m.group(1))
        box = last_boxed(text[m.end():nxt.start() if nxt else len(text)])
        if 1 <= k <= n and box is not None:
            out[k] = box
    return out


def score_paper(paper, answers):
    per_q, got = [], 0
    for k, q in enumerate(paper["questions"], 1):
        pred = answers.get(k)
        ok = is_equiv(pred, q["gold"])
        got += q["marks"] * ok
        per_q.append({"id": q["id"], "tier": q["tier"], "marks": q["marks"], "pred": pred,
                      "correct": ok})
    return got, per_q


def run_jobs(cfg, llm, tok, pool, jobs):
    """Generate and grade one batch of (paper, B, sample) runs; returns the rows."""
    from vllm import SamplingParams

    sc, s = cfg["selfalloc"], cfg["sampling"]
    prompts = [tok.apply_chat_template([{"role": "user", "content": paper_message(cfg, paper, pool, B)}],
                                       tokenize=False, add_generation_prompt=True)
               for paper, B, _ in jobs]
    # stage 1: think, at most B tokens (plus room to write the answers if it stops early)
    outs = llm.generate(prompts, [SamplingParams(
        temperature=s["temperature"], top_p=s["top_p"], seed=sample_seed(cfg, i),
        max_tokens=B + sc["answer_tokens"], skip_special_tokens=False) for _, B, i in jobs],
        use_tqdm=False)

    results, forced = [None] * len(jobs), []
    for j, ((paper, B, i), out) in enumerate(zip(jobs, outs)):
        text, ids = out.outputs[0].text, out.outputs[0].token_ids
        if THINK_CLOSE in text and len(tok.encode(text.split(THINK_CLOSE)[0],
                                                  add_special_tokens=False)) <= B:
            thinking_text, answer = text.split(THINK_CLOSE, 1)
            results[j] = (len(tok.encode(thinking_text, add_special_tokens=False)), False, answer)
        else:   # over budget: cut the thinking at B and force the answers
            thinking_text = tok.decode(ids[:B]).split(THINK_CLOSE)[0]
            forced.append((j, prompts[j] + thinking_text + "\n" + THINK_CLOSE + "\n\n"))
            results[j] = (min(B, len(ids)), True, None)
    if forced:
        gens = llm.generate([p for _, p in forced],
                            SamplingParams(temperature=0, max_tokens=sc["answer_tokens"]), use_tqdm=False)
        for (j, _), g in zip(forced, gens):
            results[j] = (results[j][0], True, g.outputs[0].text)

    rows = []
    for (paper, B, i), (thinking, cut, answer) in zip(jobs, results):
        got, per_q = score_paper(paper, parse_answers(answer, len(paper["questions"])))
        rows.append({"paper_id": paper["paper_id"], "budget": B, "sample_idx": i,
                     "thinking": thinking, "cut": cut, "marks": got,
                     "total_marks": total_marks(paper),
                     "score": 100 * got / total_marks(paper), "per_question": per_q})
    return rows


def report(rows, budgets):
    for B in budgets:
        rs = [r for r in rows if r["budget"] == B]
        if rs:
            print(f"P5 @ {B:6d}: score {sum(r['score'] for r in rs) / len(rs):5.1f}%  "
                  f"thinking {sum(r['thinking'] for r in rs) / len(rs):7.0f}  "
                  f"cut {sum(r['cut'] for r in rs)}/{len(rs)}  n {len(rs)}", flush=True)


def summarise(cfg, split):
    """Merge <dir>/<split>_part*.jsonl (if any) into <split>.jsonl, drop duplicate runs,
    and write summary.json with the paired comparison against the replay policies."""
    sc = cfg["selfalloc"]
    d = resolve(sc["dir"])
    parts = sorted(glob.glob(str(d / f"{split}_part*.jsonl")))
    rows = [r for f in parts for r in read_jsonl(f)] if parts else read_jsonl(d / f"{split}.jsonl")
    seen, merged = set(), []
    for r in rows:
        k = (r["paper_id"], r["budget"], r["sample_idx"])
        if k not in seen:
            seen.add(k)
            merged.append(r)
    merged.sort(key=lambda r: (r["paper_id"], r["budget"], r["sample_idx"]))
    if parts:
        write_jsonl(d / f"{split}.jsonl", merged)
    print(f"{len(merged)} runs on {len({r['paper_id'] for r in merged})} papers"
          + (f" (merged from {len(parts)} parts)" if parts else ""))
    report(merged, sc["budgets"])
    runs_file = resolve(cfg["policies"]["dir"]) / f"runs_{split}.jsonl"
    if not runs_file.exists():
        print("no evaluate runs yet: no comparison")
        return
    comp = paired_comparison(merged, read_jsonl(runs_file), sc["compare_policies"])
    with open(d / f"summary_{split}.json", "w") as f:
        json.dump(comp, f, indent=2)
    print(f"\nP5 against the replay policies on the same {comp['papers']} papers "
          f"(difference, 95% CI, papers P5 wins):")
    for pol, by_b in comp["policies"].items():
        print(f"  {pol:20s} " + " | ".join(
            f"{B}: {v['score']:5.1f} {v['diff']:+5.1f} [{v['ci95'][0]:+5.1f},{v['ci95'][1]:+5.1f}] "
            f"{v['p5_wins']}/{comp['papers']}" for B, v in by_b.items()))


def main():
    cfg = load_config()
    sc = cfg["selfalloc"]
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="test", choices=["train", "test"])
    p.add_argument("--n-papers", type=int, default=sc["n_papers"])
    p.add_argument("--papers", default=None, help="slice a:b of the papers (after --n-papers)")
    p.add_argument("--out", default=None, help="output file name inside selfalloc.dir")
    p.add_argument("--chunk-papers", type=int, default=sc.get("chunk_papers", 10),
                   help="papers generated and written per chunk")
    p.add_argument("--summarise", action="store_true", help="CPU only: merge parts and compare")
    args = p.parse_args()
    if args.summarise:
        summarise(cfg, args.split)
        return

    pool = {r["id"]: r for r in read_jsonl(resolve(cfg["data"]["processed_dir"]) / "pool.jsonl")}
    papers = paper_slice(read_jsonl(resolve(cfg["papers"]["dir"]) / f"{args.split}.jsonl")[:args.n_papers],
                         args.papers)
    for paper in papers:
        for q in paper["questions"]:
            q["gold"] = pool[q["id"]]["gold"]
    out_file = resolve(sc["dir"]) / (args.out or f"{args.split}.jsonl")
    done = set()
    if out_file.exists():
        done = {(r["paper_id"], r["budget"], r["sample_idx"]) for r in read_jsonl(out_file)}
    todo = pending_jobs(papers, sc["budgets"], sc["n_samples"], done)
    print(f"{len(papers)} papers, {len(done)} runs already in {out_file.name}, {len(todo)} to run",
          flush=True)
    if not todo:
        return

    llm = load_llm(cfg, max_model_len=sc["max_model_len"])
    tok = llm.get_tokenizer()
    rows = []
    chunk_ids = [p["paper_id"] for p in papers]
    for start in range(0, len(chunk_ids), args.chunk_papers):
        ids = set(chunk_ids[start:start + args.chunk_papers])
        jobs = [j for j in todo if j[0]["paper_id"] in ids]
        if not jobs:
            continue
        new = run_jobs(cfg, llm, tok, pool, jobs)
        for r in new:
            append_jsonl(out_file, r)
        rows += new
        print(f"chunk {start // args.chunk_papers + 1}: {len(rows)}/{len(todo)} runs written", flush=True)
    report(rows, sc["budgets"])


if __name__ == "__main__":
    main()
