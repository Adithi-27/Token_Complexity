"""Wall-clock latency per paper at batch size 1 (planning.md, step 15).

Measured: for the first `n_papers` test papers and each policy in `policies`, at total
budget `budget`, the policy's caps (from sample 0 of the truncation table) are run for
real, one question at a time with batch size 1: thinking is generated up to the cap
(stopping early at </think>), then the answer is forced as in tok_exam.curves. Prefix
caching is off, so no question reuses another's prefill.

Modelled: a least-squares fit  seconds per question = a + b * generated tokens  on the
measured questions turns the mean token counts of tok_exam.evaluate into an estimated
latency per paper for every policy and budget.

Writes <latency.dir>/measured.jsonl (one row per question run) and summary.json.

Run:  python -m tok_exam.latency
"""
import json
import time

import numpy as np

from .config import load_config, resolve, sample_seed
from .curves import FORCE
from .records import THINK_CLOSE, read_jsonl, write_jsonl


def fit_rate(rows):
    """(a, b) of seconds = a + b * tokens over measured question rows."""
    x = np.array([r["generated"] for r in rows], dtype=float)
    y = np.array([r["seconds"] for r in rows], dtype=float)
    A = np.column_stack([np.ones_like(x), x])
    (a, b), *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(a), float(b)


def predicted_seconds(n_questions, tokens, a, b):
    """Estimated seconds for a paper of n questions that generates `tokens` in total."""
    return n_questions * a + b * tokens


def main():
    from vllm import SamplingParams

    from .evaluate import load_all_predicted, load_predicted
    from .llm import load_llm
    from .policies import Context, Table, run_policy
    from .prompt import build_prompt

    cfg = load_config()
    lc, s = cfg["latency"], cfg["sampling"]
    table = Table(cfg)
    predicted, cost = load_predicted(cfg)
    ctx = Context(cfg, table, predicted=predicted, predictor_cost=cost,
                  predictors=load_all_predicted(cfg))
    pool = {r["id"]: r for r in read_jsonl(resolve(cfg["data"]["processed_dir"]) / "pool.jsonl")}
    papers = read_jsonl(resolve(cfg["papers"]["dir"]) / "test.jsonl")[: lc["n_papers"]]
    B = lc["budget"]

    llm = load_llm(cfg, enable_prefix_caching=False)
    tok = llm.get_tokenizer()
    force_params = SamplingParams(temperature=0, max_tokens=cfg["curves"]["max_answer_tokens"])
    llm.generate([build_prompt(cfg, tok, "What is 1 + 1?")],
                 SamplingParams(max_tokens=32), use_tqdm=False)          # warm-up

    rows = []
    for name in lc["policies"]:
        for paper in papers:
            for r in run_policy(name, ctx, paper, B, 0):
                prompt = build_prompt(cfg, tok, pool[r["id"]]["question"])
                t0 = time.perf_counter()
                thinking, n_think = "", 0
                if r["cap"] > 0:
                    o = llm.generate([prompt], SamplingParams(
                        temperature=s["temperature"], top_p=s["top_p"], seed=sample_seed(cfg, 0),
                        max_tokens=r["cap"], stop=[THINK_CLOSE], skip_special_tokens=False),
                        use_tqdm=False)[0].outputs[0]
                    thinking, n_think = o.text.split(THINK_CLOSE)[0], len(o.token_ids)
                a = llm.generate([prompt + thinking + FORCE], force_params, use_tqdm=False)[0]
                dt = time.perf_counter() - t0
                n_ans = len(a.outputs[0].token_ids)
                rows.append({"policy": name, "paper_id": paper["paper_id"], "id": r["id"],
                             "cap": r["cap"], "thinking": n_think, "answer": n_ans,
                             "generated": n_think + n_ans, "seconds": dt})
            last = [x for x in rows if x["policy"] == name and x["paper_id"] == paper["paper_id"]]
            print(f"{name:20s} {paper['paper_id']}: {sum(x['seconds'] for x in last):7.1f} s, "
                  f"{sum(x['generated'] for x in last):6d} tokens", flush=True)

    a, b = fit_rate(rows)
    measured = {}
    for name in lc["policies"]:
        per_paper = [sum(x["seconds"] for x in rows if x["policy"] == name and x["paper_id"] == p["paper_id"])
                     for p in papers]
        measured[name] = {"seconds_mean": float(np.mean(per_paper)), "papers": len(per_paper)}
    print(f"\nfit: seconds per question = {a:.3f} + {b * 1000:.2f} ms x tokens "
          f"({1 / b:.1f} tokens/s at batch size 1)")

    estimated = {}
    sfile = resolve(cfg["policies"]["dir"]) / "summary_test.json"
    if sfile.exists():
        with open(sfile) as f:
            summ = json.load(f)
        n_q = len(papers[0]["questions"])
        for name, by_b in summ["policies"].items():
            estimated[name] = {B_: predicted_seconds(n_q, v["tokens_mean"], a, b)
                               for B_, v in by_b.items()}

    out = resolve(lc["dir"])
    write_jsonl(out / "measured.jsonl", rows)
    with open(out / "summary.json", "w") as f:
        json.dump({"budget": B, "fit": {"a_seconds": a, "b_seconds_per_token": b},
                   "measured": measured, "estimated_seconds_per_paper": estimated}, f, indent=2)


if __name__ == "__main__":
    main()
