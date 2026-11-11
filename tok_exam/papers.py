"""Sample exam papers from the pool (planning.md, step 5).

A paper draws `composition[tier]` distinct questions of each tier from one split,
so test papers only ever contain test questions. Papers may share questions.
Writes data/papers/{train,test}.jsonl, one row per paper:
  paper_id, split, questions: [{id, tier, marks}]

Run:  python -m tok_exam.papers
"""
import random

from .config import TIERS, load_config, resolve
from .records import read_jsonl, write_jsonl


def sample_papers(pool, split, n_papers, composition, marks, rng):
    by_tier = {t: sorted(r["id"] for r in pool if r["tier"] == t and r["split"] == split)
               for t in TIERS}
    papers = []
    for k in range(n_papers):
        questions = [{"id": qid, "tier": t, "marks": marks[t]}
                     for t in TIERS for qid in rng.sample(by_tier[t], composition[t])]
        papers.append({"paper_id": f"{split}-{k:03d}", "split": split, "questions": questions})
    return papers


def total_marks(paper):
    return sum(q["marks"] for q in paper["questions"])


def main():
    cfg = load_config()
    pc = cfg["papers"]
    pool = read_jsonl(resolve(cfg["data"]["processed_dir"]) / "pool.jsonl")
    for i, split in enumerate(("train", "test")):
        rng = random.Random(cfg["data"]["seed"] + i)
        papers = sample_papers(pool, split, pc["n_papers"], pc["composition"], pc["marks"], rng)
        write_jsonl(resolve(pc["dir"]) / f"{split}.jsonl", papers)
        used = {q["id"] for p in papers for q in p["questions"]}
        print(f"{split}: {len(papers)} papers, {len(papers[0]['questions'])} questions and "
              f"{total_marks(papers[0])} marks each, {len(used)} distinct questions used")


if __name__ == "__main__":
    main()
