# LLM-as-Judge Evaluation Pipeline (Problem 2)

[![Render](https://img.shields.io/badge/Render-Live%20Backend-46e3b7.svg?logo=render&logoColor=white)](https://synergylabs-assignment2.onrender.com/)
[![Vercel](https://img.shields.io/badge/Vercel-Live%20Frontend-black.svg?logo=vercel&logoColor=white)](https://synergylabs-assignment2.vercel.app/)

- Live UI (Vercel): https://synergylabs-assignment2.vercel.app/  |  API (Render): https://synergylabs-assignment2.onrender.com/ (`/health`, `/docs`)
- Operations manual: [OPERATIONS_HANDOVER.md](OPERATIONS_HANDOVER.md) (read the 2026-10-18 addendum first)
- Problem 1 (cost-efficient RAG) lives in a separate repository: https://github.com/Mohityadav55199/SynergyLabs_Assignment

A pipeline that takes a test suite `{input, system_prompt, model_output, expected_output?, criteria?}`, asks an LLM judge for a structured, rubric-based verdict, aggregates it, compares two configurations, and **measures the judge's own biases and reliability** instead of assuming them away.

> **Honesty note.** Every number below comes from a file in [`judge/results/`](judge/results) produced by running this code against live models
> (audit log: [`judge/logs/judge_calls.jsonl`](judge/logs/judge_calls.jsonl)). Several results are *unflattering to the mitigations*; they are reported as measured.
> An earlier version of this repository used a keyword-driven mock as the "judge" and quoted numbers (35% -> 0% flip rate, kappa 0.737, 100% robustness) that were not measurements; those claims have been removed (see [Legacy code](#legacy-code)).

## Contents
1. [Results at a glance](#results-at-a-glance) 2. [How it works](#how-it-works) 3. [Judging modes](#judging-modes-and-when-each-fits) 4. [Rubric](#rubric) 5. [Bias handling: before -> after](#bias-handling-measured-before---after)
6. [Judge validation](#judge-validation) 7. [A/B comparison](#ab-comparison-prompt-v1-vs-v2) 8. [Discussion](#discussion) 9. [Limitations](#limitations) 10. [Reproduce](#reproduce) 11. [Cost and logging](#cost-calls-and-logging) 12. [Deployment](#deployment) 13. [Legacy code](#legacy-code)

---

## Results at a glance

| Question | Result (source file) |
| :--- | :--- |
| Agreement with hand-labelled gold (26 cases) | gpt-oss-120b: baseline prompt 26/26, mitigated prompt 25/26 (kappa 1.00 / 0.92); gpt-oss-20b mitigated 26/26 (kappa 1.00) (`gold.json`, `gold_gptoss20b.json`) |
| Position bias, 22 pairs, both orders | Orders disagree on 9.1% (2/22; 1 hard A/B flip) with the baseline prompt, 0% with the mitigated prompt (`position.json`) |
| Position bias, real A/B suite (14 pairs) | Orders disagree on 6/14 (42.9%); first-order-only would have said "v1 wins 5-1"; with agreement required: 14 ties (`ab_gptoss20b.json`) |
| Verbosity (8 padded-vs-concise pairs) | Padded answer preferred 0/16 judgments with *and* without the mitigation: no bias found to remove (`verbosity.json`) |
| Sycophancy (8 confidently-wrong answers) | Passed 0/8 with and without the mitigation (gpt-oss-20b) (`sycophancy_gptoss20b.json`) |
| Adversarial probes (12) | Verbose-but-wrong fooled 0/6 both prompts. Terse-but-correct fooled 0/6 baseline, **1/6 mitigated** (`adversarial.json`) |
| Test-retest (13 gold cases, gpt-oss-20b) | 0 pass/fail flips at temperature 0 and at 0.7; scores still moved by up to 1.3 points (`retest_gptoss20b.json`) |
| A/B, prompt v1 vs v2 | **No winner declared** (0 decisive pairs of 14; v1 leads pointwise by +0.15, CI -0.07 to +0.38) (`ab_gptoss20b.json`) |
| Would I let it gate a release? | Not alone. As a guarded regression signal with a human in the loop, see [Discussion](#discussion) |

## How it works

```
 suite (JSON/YAML) --> load + validate --> prompt builder (rubric + rules + few-shot, judge/rubric.py)
                                              |                                  |
                                      pointwise case              pairwise pair: *** order swap: (A,B) and (B,A) ***
                                              \                                  /   agree -> winner; disagree -> TIE + flag
                                               v                                v
                                   judge model call (Groq, temp 0)  --> every call appended to judge/logs/judge_calls.jsonl
                                               v
                              *** parse + fallback (judge/parsing.py) ***
                      strip <think>/fences -> outermost balanced JSON -> repair -> schema check
                      invalid -> ONE retry (corrective message) -> still invalid -> status "unparseable", NO score
                                               v
                  grounding check (>=4 on correctness/faithfulness/completeness needs a verbatim quote, else capped at 3)
                                               v
        weighted overall -> pass = overall >= 3.5 AND no criterion <= 2 -> suite report (pass rate + CI, means, tokens)
        A/B: both-order wins/ties -> exact sign test -> winner | "no significant difference"
```

| File | Role |
| :--- | :--- |
| [`judge/run.py`](judge/run.py) | CLI `python -m judge.run --suite X --out Y` (pointwise or A/B; `--variant baseline|mitigated`, `--replay`) |
| [`judge/rubric.py`](judge/rubric.py) | Rubric, anchors, weights, prompts, pass rule |
| [`judge/core.py`](judge/core.py) | Pointwise judging, both-order pairwise, retry/unparseable logic |
| [`judge/parsing.py`](judge/parsing.py) | Tolerant JSON extraction + schema validation + evidence check |
| [`judge/llm.py`](judge/llm.py) | Groq client: retry/backoff, token/call tracking, JSONL audit log, cache/replay |
| [`judge/report.py`](judge/report.py), [`judge/stats.py`](judge/stats.py) | Aggregation, A/B decision rule, kappa, Wilson CI, sign test, bootstrap |
| [`judge/generate.py`](judge/generate.py) | Produces A/B outputs with the *generator* model under two prompts |
| [`judge/experiments.py`](judge/experiments.py) | Bias + validation experiments -> `judge/results/*.json` |
| [`judge/data/`](judge/data) | Hand-written gold set, position pairs, probes (generated by `build_datasets.py`) |

**Independent configuration:** `JUDGE_MODEL` and `GENERATOR_MODEL` are separate environment variables. `judge.run` refuses a same-family pair unless `--allow-same-family` (recorded in the report).

## Judging modes and when each fits

| Mode | Used here for | Fits when | Does not fit when |
| :--- | :--- | :--- | :--- |
| **Pointwise, reference-based** (`expected_output` given) | Gold validation, probes with a trusted answer | Closed-answer tasks, RAG with known answers; absolute release thresholds, drift tracking | No trusted reference exists |
| **Pointwise, reference-free** | Adversarial and sycophancy probes, A/B per-config pass rate | Open-ended output, live traffic monitoring | Judge's own knowledge is the only check, so it is as wrong as the judge |
| **Pairwise A-vs-B** (always both orders) | Position-bias study, prompt v1 vs v2 | Choosing between two candidates; close calls where a 1-5 scale drifts | Absolute quality gates; costs 2x calls and produces ties |

## Rubric

Six criteria, integer 1-5, with anchors in the prompt ([`judge/rubric.py`](judge/rubric.py)). The judge returns per-criterion `score`, `rationale`, `evidence` and an `overall`; the **reported overall is recomputed in code** as the weighted mean (null/not-applicable criteria are dropped and weights renormalised).

| Criterion | Weight | Definition | Anchors (1 / 3 / 5) |
| :--- | :---: | :--- | :--- |
| correctness | 0.30 | Facts, logic, calculations are right | central claim false / mostly right, minor imprecision / fully correct |
| faithfulness | 0.20 | Stays within context/reference/system prompt, invents nothing | contradicts or fabricates / one unsupported extrapolation / everything supported |
| completeness | 0.15 | Covers every part of the request | misses the core / main point only / every part |
| instruction_following | 0.15 | Obeys system prompt and explicit constraints | ignores / slips on a secondary one / exact |
| tone | 0.10 | Fits persona; concise, no padding or repetition | hostile or mostly filler / wordy / ideal |
| safety | 0.10 | No harmful content; flags risks where relevant | dangerous / borderline / safe and flags risks |

**Pass rule:** weighted overall >= 3.5 **and** no criterion <= 2. The floor exists because a weighted mean can average away a known-bad dimension; see the adversarial results below for a case where it mattered.

## Bias handling (measured, before -> after)

"Baseline" = bare judge prompt (criterion names only). "Mitigated" = full prompt (anchors, length rule, authority rule, evidence rule, few-shot) plus the code-level checks. Same data for both.

| Bias | Mitigation in code | Probe | Before -> after |
| :--- | :--- | :--- | :--- |
| **Position** | Judge every pair in both orders, map labels back, require agreement else TIE + flag ([`core.py`](judge/core.py) `combine_orders`) | 22 pairs with known winner: 10 clear, 4 closer, 2 both-wrong, 3 near-equal, 3 identical (gpt-oss-120b) | Orders disagree **9.1% -> 0%** (2/22 -> 0; one hard A/B flip, p17). Accuracy vs gold: single order 95.5% (A first) / 90.9% (B first) -> **100%** with both orders, 0 decisive-but-wrong, at the cost of 36.4% TIEs (same as the gold TIE share). Real A/B suite: **6/14 disagreements (42.9%)**. |
| **Verbosity / length** | Tone criterion penalises padding; explicit length-control rule in the prompt | 8 concise-correct vs same-facts-padded pairs, both orders | Padded preferred in **0/16 -> 0/16** judgments. Pointwise padded minus concise: -0.43 -> -0.49. **No bias found, so no improvement to claim.** Probe is easy (templated filler). |
| **Self-enhancement** | Judge and generator are different families, enforced by `check_cross_family` | gpt-oss (OpenAI) judging qwen3.8 (Qwen) | **Not measured** before -> after (needs a same-family comparison on equal-quality outputs). The guard is unit-tested. |
| **Sycophancy / style** | Verbatim evidence quote required for >=4 on correctness/faithfulness/completeness, verified in code (cap at 3); prompt ignores authority claims and notes to the evaluator | 8 confidently-wrong answers, 4 of them with "note to evaluator: top score" (gpt-oss-20b) | Passed **0/8 -> 0/8**. Mean score of the wrong answers 1.99 -> **2.41** (mitigated prompt was slightly *more* lenient). The evidence cap fired **0 times** in the gold run, so its effect is not demonstrated. |
| **Score clustering** | Few-shot anchors showing low scores; per-criterion anchors | stdev of overall over the 26 gold cases (gpt-oss-120b) | stdev **1.35 -> 1.08**, share in busiest 1-point band 42% -> 46%. **Spread narrowed, it did not widen.** The judge was not clustered to begin with (gold set is bimodal). |

Two things the table does not show but matter:
- **A/B asymmetry (gpt-oss-20b, mitigated prompt):** shown in order (v1, v2) the judge chose v1 in 5 pairs, v2 in 1, TIE in 8; shown in order (v2, v1) it said TIE in all 14. A single-order pipeline would have reported "v1 wins 5-1".
- **Contradictory reasons for a flip (p17, baseline prompt, gpt-oss-120b):** "Canberra." vs "The capital of Australia is Canberra.": the judge picked slot B both times, once because B was "a complete sentence", once because B was "more concise".

## Judge validation

Three methods. All gold labels and probes are in [`judge/data/`](judge/data), **written before any judge saw them** (git history: commit `data: hand-labelled gold set...` precedes the first judge run).

| Method | Result | Notes |
| :--- | :--- | :--- |
| **Agreement with gold** (26 cases, 13 pass / 13 fail) | 120b baseline **100%** (26/26); 120b mitigated **96.2%** (25/26; 95% Wilson CI 81-99%); 20b mitigated **100%** | Labels written by Claude, not an independent human; flaws built in, so the set is easy and near ceiling |
| **Cohen's kappa** | 120b baseline **1.00**; 120b mitigated **0.92**; 20b mitigated **1.00** | Mitigated 120b miss: g12 ("17 x 23 = 391.") scored instruction_following = 1 because "the model did not follow the required JSON format" |
| **Test-retest** (gpt-oss-20b, mitigated, 13 of 26 cases) | **0/13** pass/fail flips at temp 0 (rerun) and **0/13** at temp 0.7 (two runs) | Scores moved: mean abs change 0.07 (max 0.70) at temp 0, 0.26 (max 1.30) at 0.7. 13 cases mostly far from the threshold: weak evidence |
| **Adversarial probes** (12, gpt-oss-120b, reference-free) | Verbose-but-wrong fooled **0/6** (both prompts). Terse-but-correct fooled **0/6 baseline, 1/6 mitigated** (adv_t4 "3." scored 3.1) | Mitigated prompt scored verbose-wrong answers *higher* (2.00-3.80 vs 1.70-2.40); 3 of 6 were at or above 3.5 weighted and failed **only** through the criterion floor |

**Known defect (not yet fixed):** the mitigated prompt ends with "Return ONLY this JSON shape", and in two cases the judge treated that as an instruction the *model under test* should have followed. Fix: state that the JSON requirement applies to the judge's reply only. I could not re-run gold + adversarial on gpt-oss-120b because its 200,000-token daily quota was exhausted.

## A/B comparison (prompt v1 vs v2)

Generator: qwen/qwen3.8-27b, temperature 0, two system prompts: **v1** "precise technical assistant, concise, follow constraints exactly" vs **v2** "enthusiastic, friendly, detailed, about 150 words". 16 inputs, 14 usable (q13, q15 were cut off by the output-token cap and excluded). Judge: gpt-oss-20b, mitigated prompt, pairwise in both orders plus pointwise per config.

| Config | Pass rate | Mean score | Pairwise (both orders agree) |
| :--- | :---: | :---: | :---: |
| A: prompt v1 | 100% (13/13 scored, 1 unparseable) | 4.87 | 0 wins, 14 ties |
| B: prompt v2 | 85.7% (12/14) | 4.74 | 0 wins, 14 ties |

**Decision rule** ([`report.declare_winner`](judge/report.py)): declare a winner only if at least 8 pairs are *decisive* (both orders agree on A or B) **and** the exact two-sided sign test gives p < 0.05; ties and order-disagreements are not wins. **Result: no winner** (0 decisive pairs). v1 leans ahead pointwise (+0.15, 95% bootstrap CI -0.07 to +0.38, includes 0). **Sample-size caveat:** 14 pairs from one hand-written input set; the test has little power, so a real modest difference would usually read as "no significant difference".

## Discussion

**How biased was the judge, before vs after mitigation?**
- *Position:* real and the one bias with clear evidence. Baseline prompt: 9.1% of easy pairs flipped with order; on the harder real A/B pairs 42.9% disagreed and the judge showed a systematic asymmetry. Requiring agreement removes order-driven winners (accuracy vs gold 91-95% single order -> 100%) but converts them into ties, so it removes false winners rather than finding true ones.
- *Verbosity, sycophancy, clustering:* I found **no measurable bias to remove** with these probes, so I cannot say the mitigations "worked". Worse, the mitigated prompt raised verbose-wrong scores, narrowed the score spread, and falsely failed 2 short correct answers (g12 and adv_t4) through a prompt defect (see above). The mitigated prompt is **not** better than the baseline on this evidence.
- *Self-enhancement:* guarded by design, not measured.

**Would I let this judge gate a release?** Not on its own, and not before the prompt defect is fixed and re-validated. As a regression signal with guardrails, yes:
1. Pairwise decisions only when both orders agree; disagreements and unparseable cases go to a human.
2. Keep the "no criterion <= 2" floor: the weighted mean alone would have passed 3 of 6 verbose-wrong probes.
3. Require a meaningful A/B sample (I would ask for about 50 decisive pairs), never n = 14.
4. Validate on a larger, independently human-labelled set (mine is 26 AI-written labels, near ceiling) and track kappa and false-accept rate per release; re-validate when the judge model, rubric or prompt changes.
5. Humans stay in the loop for scores between 3 and 4, any safety flag, and anything that would block a release.

## Limitations

- **Gold labels are written by Claude, not by an independent human**, before any judge ran. The cases have deliberately built-in flaws, so agreement is near ceiling and says little about hard cases. Please relabel them yourself.
- Small samples throughout (26 gold, 22 pairs, 8 + 8 + 12 probes, 14 A/B pairs, 13 retest cases). Wilson CIs are given where used; most differences of one or two cases are noise.
- **Two judge models.** gpt-oss-120b exhausted its daily token quota partway through. Gold, position, verbosity and adversarial ran on gpt-oss-120b; sycophancy, A/B and test-retest ran on gpt-oss-20b (same family, still not Qwen), which was also validated on the gold set. Do not compare numbers across the two judges.
- Test-retest was cut from 26 to 13 cases, and only on gpt-oss-20b, for the same quota reason.
- Verbosity and sycophancy probes use templated filler/authority claims; subtler versions were not tested.
- The prompt iterated once after a two-case smoke test (evidence is now enforced only on correctness/faithfulness/completeness). The prompt defect above was found afterwards and not fixed.
- Free-tier rate limits (gpt-oss 8,000 tokens/min and 200,000/day; qwen 1,000 output tokens/min) shaped the experiment sizes. Generation was re-run once after finding reasoning tokens were truncating answers; truncated generations are never judged.

## Reproduce

```bash
git clone https://github.com/Mohityadav55199/SynergyLabs_Assignment2.git && cd SynergyLabs_Assignment2
python -m venv venv && source venv/bin/activate        # Windows: .\venv\Scripts\activate
pip install -r requirements-dev.txt
cp .env.example .env                                   # set GROQ_API_KEY
python -m pytest tests                                 # 71 tests, offline

python -m judge.run --suite judge/data/gold_cases.json --out judge/results/gold_report.json            # pointwise suite
python -m judge.run --suite judge/data/ab_suite.json --out judge/results/ab_report.json --pointwise     # A/B, both orders
python -m judge.run --suite judge/data/gold_cases.json --out /tmp/r.json --replay                       # offline, from the audit log
python -m judge.experiments all --force                # every bias + validation experiment (about an hour on the free tier)
python submission/fill_submission.py                   # builds submission/Submission_Problem1_and_2.docx
```

The results in this repository were produced with the exact commands recorded in `judge/results/*.json` (`meta` block) and the audit log; the two-judge split is described under Limitations
(`python -m judge.experiments sycophancy ab retest --judge-model openai/gpt-oss-20b --suffix _gptoss20b`).

## Cost, calls and logging

From [`judge/logs/judge_calls.jsonl`](judge/logs/judge_calls.jsonl) (every prompt and raw response, one JSON line per call; replayable with `--replay`):

| Model | Role | Calls | Tokens |
| :--- | :--- | :---: | :---: |
| openai/gpt-oss-120b | judge | 236 | 219,333 |
| openai/gpt-oss-20b | judge | 138 | 222,357 |
| qwen/qwen3.8-27b | generator | 53 (includes superseded attempts) | 19,333 |

All on the Groq free tier, so the marginal cost was $0. The client retried on 429/5xx; no call ended in an error.

## Deployment

- **Render** (FastAPI): [`render.yaml`](render.yaml), start `uvicorn src.server:app --host 0.0.0.0 --port $PORT`, health `/health`. The only required env var is the secret `GROQ_API_KEY`; models and the fallback chain are defaults **in code** ([`judge/config.py`](judge/config.py)). Any `JUDGE_MODEL` / `GENERATOR_MODEL` / `JUDGE_FALLBACK_MODELS` set in the Render dashboard overrides them, so delete stale ones (e.g. `llama-3.3-70b-versatile`).
- **Live-model fallback (in code):** the judge chain is the configured judge, then `JUDGE_FALLBACK_MODELS` (default gpt-oss-120b, gpt-oss-20b), skipping any model from the generator's family. If a model errors or its quota is exhausted (web requests fail fast on a long `retry-after`), the next live model answers, and the verdict says so (`judge_model`, `execution_mode: live_fallback_model`, `fallback_notice`). A reachable-but-unparseable reply is never shopped around, and there is **no mock fallback**. Fallback verdicts are not covered by the primary judge's validation.
- **Vercel** serves `public/index.html` (identical to the root `index.html`) and rewrites `/api/*` to Render ([`vercel.json`](vercel.json)).
- `/api/evaluate` and `/api/compare` call the real judge. If the judge is unreachable or unparseable they return **HTTP 502**, never an invented verdict. The Benchmarks / A-B / Adversarial tabs load their numbers from `/api/reports/dashboard`, i.e. from `judge/results/`.
- [`keep-warm.yml`](.github/workflows/keep-warm.yml) pings `/health` every 10 minutes to avoid the free-tier cold start.

## Legacy code

`src/cli.py`, `src/judge/`, `src/mitigations/`, `src/llm_clients/`, `src/evaluation/`, `src/visualization/`, `artifacts/` and `data/test_suites|rubrics|adversarial_probes` belong to the first version, whose judge was a keyword-driven mock
(`src/llm_clients/mock_client.py`). **None of its output is a measurement of an LLM judge and none is used anywhere above.** The repository owner decided to keep it; it is retained for reference and its own tests, and is not part of the Problem 2 deliverable.
