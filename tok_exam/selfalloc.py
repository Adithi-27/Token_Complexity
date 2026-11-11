"""P5 self-allocation: the model gets the whole paper and the budget (planning.md, step 11).

One prompt per (paper, B) lists every question with its marks and says the
thinking budget is B tokens for the whole paper. The model thinks once and
then writes one answer per question as "Q<k>: \\boxed{...}". If its thinking
runs past B it is cut there, `</think>` is appended, and the answers are
generated greedily. Answers are matched to questions by their "Q<k>" label.

Writes <selfalloc.dir>/{split}.jsonl, one row per (paper, budget, sample):
  paper_id, budget, sample_idx, thinking, cut, marks, total_marks, score,
  per_question: [{id, tier, marks, pred, correct}]

Run:  python -m tok_exam.selfalloc
"""
import argparse
import re

from .config import load_config, resolve, sample_seed
from .grader import is_equiv, last_boxed
from .llm import load_llm
from .papers import total_marks
from .records import THINK_CLOSE, read_jsonl, write_jsonl

_LABEL = re.compile(r"(?:^|\n)\s*(?:\*\*)?Q(\d+)(?:\*\*)?\s*[:.)]", re.M)


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


def main():
    cfg = load_config()
    sc = cfg["selfalloc"]
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="test", choices=["train", "test"])
    p.add_argument("--n-papers", type=int, default=sc["n_papers"])
    args = p.parse_args()

    from vllm import SamplingParams

    pool = {r["id"]: r for r in read_jsonl(resolve(cfg["data"]["processed_dir"]) / "pool.jsonl")}
    papers = read_jsonl(resolve(cfg["papers"]["dir"]) / f"{args.split}.jsonl")[:args.n_papers]
    for paper in papers:
        for q in paper["questions"]:
            q["gold"] = pool[q["id"]]["gold"]

    llm = load_llm(cfg, max_model_len=sc["max_model_len"])
    tok = llm.get_tokenizer()
    s = cfg["sampling"]
    jobs = [(paper, B, i) for paper in papers for B in sc["budgets"] for i in range(sc["n_samples"])]
    prompts = [tok.apply_chat_template([{"role": "user", "content": paper_message(cfg, paper, pool, B)}],
                                       tokenize=False, add_generation_prompt=True)
               for paper, B, _ in jobs]
    # stage 1: think, at most B tokens (plus room to write the answers if it stops early)
    outs = llm.generate(prompts, [SamplingParams(
        temperature=s["temperature"], top_p=s["top_p"], seed=sample_seed(cfg, i),
        max_tokens=B + sc["answer_tokens"], skip_special_tokens=False) for _, B, i in jobs])

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
                            SamplingParams(temperature=0, max_tokens=sc["answer_tokens"]))
        for (j, _), g in zip(forced, gens):
            results[j] = (results[j][0], True, g.outputs[0].text)

    rows = []
    for (paper, B, i), (thinking, cut, answer) in zip(jobs, results):
        got, per_q = score_paper(paper, parse_answers(answer, len(paper["questions"])))
        rows.append({"paper_id": paper["paper_id"], "budget": B, "sample_idx": i,
                     "thinking": thinking, "cut": cut, "marks": got,
                     "total_marks": total_marks(paper),
                     "score": 100 * got / total_marks(paper), "per_question": per_q})
    write_jsonl(resolve(sc["dir"]) / f"{args.split}.jsonl", rows)
    for B in sc["budgets"]:
        rs = [r for r in rows if r["budget"] == B]
        print(f"P5 @ {B:6d}: score {sum(r['score'] for r in rs) / len(rs):5.1f}%  "
              f"thinking {sum(r['thinking'] for r in rs) / len(rs):7.0f}  "
              f"cut {sum(r['cut'] for r in rs)}/{len(rs)}", flush=True)


if __name__ == "__main__":
    main()
