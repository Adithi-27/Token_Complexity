# Token Budgeting for a Whole Exam Paper

Course project (CMInDS, IIT Bombay, Nov 2026). Given one thinking-token budget for a
20-question exam paper of easy, moderate and hard maths questions worth 1, 2 or 4 marks,
how should DeepSeek-R1-Distill-Qwen-32B split the tokens across the questions to score the
most marks, and what does the allocator need to know?

- **Report:** `reports/project_report.pdf` (source `project_report.tex`)
- **Slides:** `reports/slides.pdf`
- **Plan and checklist:** `planning.md`
- **Function-by-function walkthrough of the code:** `agents/codebase_walkthrough.md`

## How it works

Every question is answered 4 times at the full budget (16k thinking tokens). Each trace is
then cut at 9 budgets and an answer is forced, which gives every question an
accuracy-vs-budget curve and a table (question, budget, sample) → outcome. Every allocation
policy is scored by replaying that table, so no policy needs new generation. No model is
trained; "training" means fitting small difficulty predictors and a LinUCB bandit on the
train half of the questions, and "testing" means scoring policies on 100 papers built from
the test half.

## Reproducing

All settings are in `config.yaml`. CPU stages run on the login node, GPU stages on two L40S
through Slurm (`slurm/`). Run in this order:

```bash
export PYTHONPATH=vendor:$PYTHONPATH          # math-verify, matplotlib
python -m unittest discover -s tests          # 62 tests, ~10 s
python -m tok_exam.data && python -m tok_exam.papers
sbatch slurm/generate.sh                      # 2236 traces, ~11 GPU-hours
sbatch slurm/pipeline_tc.sh                   # curves -> difficulty -> icl -> probe -> difficulty
for m in bandit evaluate valid ablations stats plots; do python -m tok_exam.$m; done
bash slurm/submit_tc.sh selfalloc realbudget latency   # P5, truncation check, latency (GPU)
bash reports/build.sh                         # report and slides (tectonic)
```

`slurm/autopilot_tc.sh` watches submitted GPU jobs and resubmits them on failure.
Outputs land in `outputs/`; figures in `reports/figures/`.

## Layout

| Path | Contents |
|---|---|
| `tok_exam/` | the Python package, one module per pipeline stage |
| `tests/` | unit tests on fake tokenizers, LLMs and tables (no GPU) |
| `slurm/` | job scripts for the anandi partition |
| `data/` | question pool and sampled papers |
| `outputs/` | traces, curves, predictions, evaluations, statistics |
| `reports/` | report, slides, figures |
| `agents/` | notes: related work, walkthrough, GSM8K hand check, session findings |
| `vendor/`, `vendor_vllm/` | pinned third-party code (math-verify, matplotlib; transformers 4.48 for vLLM 0.6.6) |
