"""Full-budget generation (planning.md, step 6).

Writes one JSONL row per generation to <out-dir>/{tier}.jsonl, for train and
test questions alike. Rows already in the output file are skipped, so a killed
job can be resubmitted.

Run:  python -m tok_exam.generate                    # everything, n_samples each
      python -m tok_exam.generate --out-dir outputs/smoke --limit-questions 2 --limit-samples 2
"""
import argparse

from .config import TIERS, load_config, max_tokens, resolve, sample_seed
from .llm import load_llm
from .prompt import build_prompt
from .records import append_jsonl, make_row, read_jsonl


def pending_requests(cfg, tokenizer, tier, out_file, n_questions, n_samples):
    """(item, sample_idx, prompt) for every generation not yet in out_file."""
    done = set()
    if out_file.exists():
        done = {(r["id"], r["sample_idx"]) for r in read_jsonl(out_file)}
    pool = read_jsonl(resolve(cfg["data"]["processed_dir"]) / "pool.jsonl")
    items = [r for r in pool if r["tier"] == tier][:n_questions]
    return [
        (item, i, build_prompt(cfg, tokenizer, item["question"]))
        for item in items for i in range(n_samples) if (item["id"], i) not in done
    ]


def main():
    cfg = load_config()
    p = argparse.ArgumentParser()
    p.add_argument("--tiers", nargs="+", default=list(TIERS), choices=TIERS)
    p.add_argument("--out-dir", default=cfg["output"]["dir"])
    p.add_argument("--limit-questions", type=int, default=None)
    p.add_argument("--limit-samples", type=int, default=cfg["sampling"]["n_samples"])
    p.add_argument("--chunk", type=int, default=64, help="generations per checkpoint")
    args = p.parse_args()
    out_dir = resolve(args.out_dir)

    from vllm import SamplingParams  # imported late: slow, and needs a GPU

    s = cfg["sampling"]
    llm = load_llm(cfg)
    tokenizer = llm.get_tokenizer()

    for tier in args.tiers:
        out_file = out_dir / f"{tier}.jsonl"
        todo = pending_requests(cfg, tokenizer, tier, out_file,
                                args.limit_questions, args.limit_samples)
        print(f"{tier}: {len(todo)} generations to run -> {out_file}", flush=True)
        for start in range(0, len(todo), args.chunk):
            chunk = todo[start:start + args.chunk]
            params = [SamplingParams(temperature=s["temperature"], top_p=s["top_p"],
                                     max_tokens=max_tokens(cfg), seed=sample_seed(cfg, i),
                                     skip_special_tokens=False)
                      for _, i, _ in chunk]
            outputs = llm.generate([prompt for _, _, prompt in chunk], params)
            for (item, i, prompt), out in zip(chunk, outputs):
                append_jsonl(out_file, make_row(item, i, prompt, out.outputs[0].text, tokenizer))
            print(f"{tier}: {start + len(chunk)}/{len(todo)} done", flush=True)


if __name__ == "__main__":
    main()
