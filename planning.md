# Token Budgeting for a Whole Exam Paper

**Course project plan: order of work**

Adithi Roy Chowdhury (Member A) and Member B — CMInDS, IIT Bombay

24 October 2026 to about 23 November 2026 (4 weeks). Pre-project report: `reports/preproject_report.tex`.

## The idea in one paragraph

A student with three hours for an exam does not give every question the same time. They finish the easy ones quickly, spend longer on the moderate ones, and decide early which hard ones are worth trying. This project asks the same question of a reasoning LLM: **given one fixed token budget for a whole exam paper of easy, moderate and hard questions, how should the tokens be split across questions to get the highest total score?** We compare simple splits (equal share, fixed difficulty ratios, answering in order until the budget runs out) with splits that predict each question's difficulty, and with a small difficulty-aware bandit (a lightweight form of RL) that learns how many tokens each kind of question needs.

## Revised direction (7 Nov 2026, after meeting with the professor)

1. **In-context learning first** (`tok_exam/icl.py`). The model sees k labelled train questions ("thinking tokens needed: E) 1024") and predicts the budget of a new one. It is used per question (predictors `icl`, `icl_raw` feeding the P3 knapsack) and per paper (policy **P6**). Controls: k = 0, random exemplars, shuffled labels, and a no-LLM kNN over the same neighbours.
2. **Budget accounting** (`tok_exam/valid.py`): each question's sufficient budget s\* (from the truncation table) is compared with the budget each policy gives it: useful / overspend / hopeless tokens and shortfall. This is an allocation diagnostic. Waste *inside* a trace (post-answer tail, when to stop) is not studied here (see Scope below).
3. **Hindsight oracle** added; the old oracle is only the best plan (see `agents/session_findings.md`).
4. Agent reports live in `agents/`.

## Scope, and independence from the Reasoning efficiency project

This project and the first author's Reasoning efficiency project are **independent works, to be published separately** (decided 10 Nov 2026).

| | Token Complexity (this project) | Reasoning efficiency (separate work) |
|---|---|---|
| Main question | How should a *shared* budget be split *across* questions? | How much thinking is wasted *inside* a trace, and can a model learn when to stop? |
| Unit of study | A whole paper of 20 questions | One trace of one question |

- **Claims owned by this project:** allocation across questions (policies, oracles, knapsack, ICL and other budget predictors, the bandit), and the budget accounting of each policy's spend against sufficient budgets.
- **Not claimed here** (they belong to the other work): how much of a trace comes after the answer, overthinking or "valid thinking" per difficulty tier, early exit and within-trace stopping.
- **Nothing shared:** no data, traces, results, text or code. Both may use public benchmarks and standard techniques (e.g. s1-style budget forcing), each with its own implementation.

> **Rule:** never import from, copy from, or write to `../Reasoning efficiency/`, and never use its results here. Each paper may cite the other; neither presents the other's findings as its own.

> **If short on time:** steps 1–11 and 15–16 are the core results. Steps 12–14 and the stretch goals add depth.

## Team and division of work

Two members, one language (English). The split follows the pipeline: **Member A** (Adithi) owns the data, generation and the allocation policies; **Member B** owns everything about predicting difficulty and learning to allocate, plus the GPU experiments that run on top of the stored traces. Both write the report and slides and review each other's code.

| Week | Member A | Member B |
|---|---|---|
| 1 (24–30 Oct) | Steps 1–6: pool, grader, papers, smoke test (done 24 Oct); full generation (job 63023) | Read the code and the stored smoke outputs; set up `outputs/probe`; draft related-work section |
| 2 (31 Oct–6 Nov) | Steps 7–8: curves, sanity check; step 9 needed budget | Step 10: predictors (a)–(e); run the probe stage (`tok_exam.probe`) |
| 3 (7–13 Nov) | Steps 11–12: policies P0–P4, oracle, rollover variant, main sweep | Step 13: bandit; P5 self-allocation (`tok_exam.selfalloc`); step 14 ablations |
| 4 (14–23 Nov) | Step 15 figures and results table; integration | Latency measurement; real-budget-control check (stretch); final report sections |

**Hand-offs:** Member B's predictors need Member A's curve files (end of week 2); Member A's P3/P4 need Member B's `predicted_curves.json` (start of week 3). Until then each side tests on synthetic curves (`TOK_CONFIG` dry-run config).

**GPU slots:** each member's account has its own QOS limit on anandi (2 running, 4 submitted), so Member B's probe and P5 jobs can run alongside Member A's generation and curves.

## Phase 1: Setup (week 1, days 1–3)

1. **Fix the model and inference settings.** DeepSeek-R1-Distill-Qwen-32B, vLLM, temperature 0.6, top-*p* 0.95, max thinking 16384 tokens, fixed seeds, two 48 GB GPUs. All in `config.yaml`. Weights (~62 GB) live in `/janaki/backup/.../hf_home` because home is near its quota.

2. **Build the question pool (559 questions, all with checkable answers).**
   - **Easy (200):** GSM8K test.
   - **Moderate (195):** all MATH-500 questions at levels 2–3.
   - **Hard (164):** all MATH-500 level 5 (134), plus AIME 2025 (30).
   - Difficulty labels come from the dataset. Step 9 checks them against what the model actually finds hard.
   - Split the pool 50/50 into **train** questions (used to fit predictors and the bandit) and **test** questions (used only to build test papers). Nothing in a test paper is ever seen during fitting.

3. **Fix the prompt.** DeepSeek's recommended one: question plus "Please reason step by step, and put your final answer within `\boxed{}`." No system prompt. Same template for every tier.

4. **Write the grader.** Last `\boxed{}`; integer compare for GSM8K and AIME, `math-verify` for MATH (fractions, radicals, intervals). Test on about 15 hand-written outputs.

5. **Define an exam paper.** 20 questions: 10 easy (1 mark each), 6 moderate (2 marks), 4 hard (4 marks), 38 marks total. Sample 100 test papers and 100 train papers with fixed seeds. Report every result as mean ± std over papers.

## Phase 2: Generation and budget curves (week 1 day 4 to week 2 day 2)

6. **Full-budget runs, 4 samples per question.** 559 × 4 = 2236 generations, max 16k thinking tokens. Easy questions finish in a few hundred tokens. Run 2 questions × 2 samples per tier end to end first.

7. **Budget curves by truncation.** Cut every trace at {0, 128, 256, 512, 1k, 2k, 4k, 8k, 16k} thinking tokens, append `</think>` and `Final answer: \boxed{`, and grade a short greedy continuation. Budget 0 means no thinking at all. This gives each question a curve *p_i(b)* = fraction of its 4 samples correct at budget *b*.

8. **Sanity check.** Full-budget accuracy should fall from easy to moderate to hard. If easy questions score below about 85%, suspect a prompt or grader bug.

## Phase 3: Difficulty from the model's point of view (week 2)

9. **Empirical difficulty.** For each question, the *needed budget* is the smallest *b* with *p_i(b) ≥ 0.5* (∞ if never). Plot needed budget by dataset tier. Expected finding: the tiers overlap, so dataset labels alone are a weak guide. This motivates steps 10–11.

10. **Difficulty predictors (fit on train questions only).** Each predicts the needed-budget bin before the question is solved:
    - (a) dataset tier label (the baseline a human would use),
    - (b) cheap question features: length, number of numbers, LaTeX operators, keywords,
    - (c) the model's own estimate: ask it "rate this question 1–5" with thinking turned off,
    - (d) a probe: let it think for 128 tokens and use the mean token entropy of that prefix plus whether it already gave an answer,
    - (e) a small classifier on a sentence embedding (MiniLM, already cached).

    Report accuracy and Spearman correlation with the true needed budget. Count the tokens each predictor costs, because they come out of the exam budget.

## Phase 4: Allocation policies (week 3)

11. **The policies.** Each gets a total budget *B* for the paper and must return a budget *b_i* per question from the grid in step 7. Score a policy by looking up the truncated results, so trying a new policy needs no new generation.

| # | Policy | What it does |
|---|---|---|
| P0 | Equal share | *b_i = B / 20*, rounded to the grid with the same leftover fill as P2 (P2 at 1:1:1) |
| P1 | In order, no plan | Answer questions in printed order with all the budget that is left (up to 16k each) until *B* runs out; the rest are answered with no thinking (the student who runs out of time and guesses) |
| P2 | Fixed tier ratio | Easy : moderate : hard = 1 : 2 : 4 (and a grid search over ratios on train papers) |
| P3 | Predicted knapsack | Use a predictor from step 10 to estimate each *p_i(b)*, then maximise Σ marks_i · p_i(b_i) subject to Σ b_i ≤ B. This is a multiple-choice knapsack, solved exactly by DP over the budget grid |
| P4 | Two-pass triage | Pass 1: 256 tokens on every question. Questions that answer confidently (2 samples agree) stop. Pass 2 shares what is left across the rest, ordered by marks per expected token. Hard questions that look hopeless are skipped |
| P5 | Self-allocation | Give the model the whole paper in one prompt and tell it the budget. Does an LLM spread its own effort sensibly? |
| P* | Oracle | Knapsack with the true curves. Upper bound, not a real policy |

12. **Main sweep.** For *B* ∈ {2k, 4k, 8k, 16k, 32k, 64k}, run every policy on the 100 test papers. Main plot: paper score (%) vs total budget, one line per policy. Main number: tokens each policy needs to reach 90% of the oracle's score.

## Phase 5: Difficulty-aware bandit (week 3, end)

13. **Contextual bandit as lightweight RL.** Context = predictor features from step 10; arms = budgets in the grid; reward = marks × correct − λ × tokens/1000. Train LinUCB (or ε-greedy with a linear model) on the train papers by replaying the stored truncation results, so training costs no GPU time. Use it as a policy: each question's arm, then rescale so the total fits *B*. Sweep λ to trace a score-vs-tokens curve and compare it with P3 and the oracle.

14. **Ablations.**
    - Marks scheme: equal marks vs 1/2/4. Does the best policy change?
    - Paper mix: an easy-heavy paper vs a hard-heavy paper.
    - Predictor cost: P3 with predictor tokens counted vs not counted.

## Phase 6: Results (week 4)

15. **Main table** (template below) and **plots**:
    - Score vs total budget, one line per policy (the headline figure).
    - Needed budget by tier (box plot), showing the overlap.
    - Tokens spent per tier under each policy at *B* = 16k (stacked bars).
    - Latency: wall-clock seconds per paper, measured at batch size 1 for a sample of papers.

16. **Report and slides.** Problem, setup, policies, results, and what a "good exam strategy" looks like for an LLM.

| Policy | Score @ 4k (%) | Score @ 16k (%) | Tokens to 90% of oracle | Tokens / mark | Latency / paper (s) |
|---|---|---|---|---|---|
| P0 Equal share | | | | | |
| P1 In order | | | | | |
| P2 Tier ratio | | | | | |
| P3 Knapsack (best predictor) | | | | | |
| P4 Two-pass triage | | | | | |
| P5 Self-allocation | | | | | |
| Bandit | | | | | |
| Oracle | | | | | |

*Table 1: Main results on 100 test papers. Mean over papers.*

## Stretch goals (only if weeks 1–3 run on time)

- **Coding section.** Add 20 MBPP problems graded by unit tests as one section of the paper.
- **Real budget control instead of truncation.** Regenerate a sample of questions with a budget-aware prompt ("think for at most N tokens") and check that truncation results match.

## Week-by-week

| Week | Dates | Steps | Done when |
|---|---|---|---|
| 1 | 24–30 Oct | 1–6 | Model downloaded, 559 questions split, grader tested, smoke test passed (24 Oct), full generation running |
| 2 | 31 Oct – 6 Nov | 7–10 | Budget curves for all questions, needed-budget plot, predictors scored |
| 3 | 7–13 Nov | 11–14 | All policies and bandit evaluated on test papers, ablations run |
| 4 | 14–23 Nov | 15–16 (+ stretch) | Table, plots, report, slides |

## Planned layout

```
Token Complexity/
  planning.md          this plan
  config.yaml          single source of truth
  tok_exam/            Python package
    data.py            step 2: pool, tiers, train/test split
    papers.py          step 5: sample papers
    grader.py          step 4
    generate.py        step 6
    curves.py          step 7: truncate + force answer
    difficulty.py      steps 9-10: needed budget, predictors
    policies.py        step 11: P0-P5, oracle, knapsack DP
    bandit.py          step 13
    evaluate.py        steps 12, 14
    plots.py           step 15
  tests/
  slurm/
  vendor/              third-party: math-verify (do not edit)
  vendor_vllm/         third-party: transformers 4.48 for vLLM 0.6.6 (do not edit)
  data/  outputs/  reports/  logs/
```

## Checklist

- [x] Config fixed, model downloaded
- [x] Question pool built and split train/test
- [x] Prompt and grader tested
- [x] Papers sampled
- [x] Full-budget generations done (4 samples)
- [x] Budget curves computed
- [x] Needed budget per question; tier overlap plot
- [x] Difficulty predictors scored
- [x] Policies P0–P7 and oracles evaluated
- [x] Bandit trained and evaluated (out-of-fold context, 10 Nov)
- [x] Ablations done (11 Nov, tok_exam/ablations.py: marks, mix, size, predictor cost, GSM8K-cleaned)
- [x] Table, plots and report written (11 Nov); slides drafted (reports/slides.tex, 11 Nov), to finish after the 15 Nov freeze
- [x] Pre-project report written
- [x] Rollover variant of the plan policies
- [x] ICL module (P6, icl predictors, controls) and valid/invalid thinking module written and tested
- [x] P7 shadow price, predictor variants (`P3_knapsack@icl`), two-level bootstrap (`tok_exam/stats.py`), real-budget checks (`tok_exam/realbudget.py`), leakage check, GSM8K hand check (9 Nov)
- [x] GPU pipeline job 65019: curves -> ICL -> probe (done 9 Nov, 4h10m); CPU stages bandit/evaluate/valid/plots/stats rerun on real curves
- [x] P4 fix: P4_triage_closed (10 Nov; recovers 52.8 -> 84.2 @64k, still < P3)
- [x] Review fixes (10 Nov): submit_tc.sh stage args, P0 fill, P4 predictor cost, out-of-fold train curves, valid.py check tokens, realbudget prefix encoding; 57 tests
- [x] P5 selfalloc (65152), realbudget (65153), latency (65154): done 11 Nov. P5 beats every replay policy at 8k-16k by compressing (~400 tokens/question); truncation matches a budget-aware prompt within 1.4 points; 19.9 tok/s at batch 1
- [x] P5 full run, 100 test papers x 4 samples (job 65210, done 12 Nov 02:20; 12h47 / 12h01). P5 20.0 / 46.2 / 64.2 / 76.7 at 2k-16k vs P7+roll 35.2 / 41.5 / 53.0 / 67.8 (+11.1 at 8k, +9.0 at 16k, CIs clear of 0); outputs/selfalloc/summary_test.json; report updated 12 Nov, slides still show the interim 20-paper numbers
- [x] GSM8K label decision: course report keeps raw labels + cleaned redraw (option A); ICML fixes labels (option B), steps in the report
- [x] ICL GPU run (job 65019)
- [x] Valid/invalid analysis on real curves
