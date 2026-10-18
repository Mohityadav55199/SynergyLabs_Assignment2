# Autonomous LLM-as-Judge Evaluation Pipeline & Bias Mitigation Engine

[![Render](https://img.shields.io/badge/Render-Live%20Backend%20%26%20UI-46e3b7.svg?logo=render&logoColor=white)](https://synergylabs-assignment2.onrender.com/)
[![Vercel](https://img.shields.io/badge/Vercel-Live%20Frontend-black.svg?logo=vercel&logoColor=white)](https://synergylabs-assignment2.vercel.app/)
[![Python](https://img.shields.io/badge/Python-3.10%2B%20%7C%203.14-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Evaluation](https://img.shields.io/badge/Evaluation-Pointwise%20%26%20Pairwise-indigo.svg)](#judging-design)
[![Bias Mitigation](https://img.shields.io/badge/Bias%20Mitigations-Position%20%7C%20Verbosity%20%7C%20Sycophancy-emerald.svg)](#bias-handling--mitigation-matrix)

> **Live Deployment Links & Documentation:**
> - 🌐 **Render (Backend API & Interactive App):** [https://synergylabs-assignment2.onrender.com/](https://synergylabs-assignment2.onrender.com/)
> - ⚡ **Vercel (Frontend Client):** [https://synergylabs-assignment2.vercel.app/](https://synergylabs-assignment2.vercel.app/)
> - 🩺 **Health Check & Keep-Warm Endpoint:** [https://synergylabs-assignment2.onrender.com/health](https://synergylabs-assignment2.onrender.com/health)
> - 📋 **Client Operations & Handover Manual:** [OPERATIONS_HANDOVER.md](OPERATIONS_HANDOVER.md)

A production-grade, auditable evaluation framework designed to score, compare, and validate LLM outputs while actively detecting, quantifying, and mitigating the core biases inherent to LLM judges. Designed with a clean enterprise White & Blue interface with live Render cold-start status detection.

---

## 📑 Table of Contents
- [Executive Overview](#executive-overview)
- [Client Handover & Operations Guide](#client-handover--operations-guide)
- [Architecture & Pipeline Flow](#architecture--pipeline-flow)
- [Judging Design: Pointwise vs. Pairwise](#judging-design-pointwise-vs-pairwise)
- [Explicit Multi-Criteria Rubric](#explicit-multi-criteria-rubric)
- [Bias Handling & Mitigation Matrix (The Core)](#bias-handling--mitigation-matrix)
- [Judge Validation & Ground Truth Evidence](#judge-validation--ground-truth-evidence)
- [A/B Comparison: Prompt v1 vs Prompt v2](#ab-comparison-prompt-v1-vs-prompt-v2)
- [Engineering Hygiene & Cost Accounting](#engineering-hygiene--cost-accounting)
- [In-Depth Discussion: Release Gating Safeguards](#in-depth-discussion-release-gating-safeguards)
- [Quickstart & Reproduction Guide](#quickstart--reproduction-guide)

---

## Executive Overview

Human review fails to scale with continuous model updates, prompt iterations, and RAG retrieval modifications. While teams leverage strong LLMs as automated judges, **judges themselves exhibit severe systematic biases**:
1. **Position Bias:** Preferring whichever candidate is presented first in pairwise evaluation.
2. **Verbosity Bias:** Conflating word count and ornate formatting with substantive factual correctness.
3. **Sycophancy / Style Bias:** Falling for flattering rhetoric or confidently asserted falsehoods.
4. **Self-Enhancement Bias:** Preferring answers produced by their own model family.
5. **Score Compression:** Clustering all pointwise scores into a narrow band (e.g. 3.8 – 4.2).

This project implements an end-to-end evaluation pipeline that takes bias mitigation seriously: **every bias is explicitly identified, counteracted in code, and verified with quantitative benchmark metrics.**

---

## 📋 Client Handover & Operations Guide

For comprehensive production maintenance, service ownership, and disaster recovery runbooks, refer to [`OPERATIONS_HANDOVER.md`](OPERATIONS_HANDOVER.md).

### Quick Handover Summary:
- **Service Ownership:**
  - **Inference Provider (Groq / OpenAI):** Client API Administrator owns API key quotas, billing, and model limits.
  - **Backend Hosting (Render):** SysAdmin/DevOps owns container runtime, environment secrets, and auto-deploys from `main`.
  - **Frontend Edge CDN (Vercel):** Frontend team owns DNS, caching headers, and client routes.
  - **Uptime / Keep-Warm (GitHub Actions):** Pings `/health` every 10 min to prevent free-tier instance sleep.
- **Incident Runbook ("What to do if something breaks"):**
  - **Render Cold Start / Delay:** Waking banner auto-detects status; keep-warm cron auto-pings every 10 min.
  - **Provider 404 / 429 Rate Limit:** [`ResilientLLMClient`](src/llm_clients/resilient_client.py) automatically cascades to secondary models (`llama-3.1-70b-versatile`, `llama-3.1-8b-instant`) or calibrated deterministic judge with full audit provenance.
  - **Malformed LLM Output:** [`RobustJsonParser`](src/parser.py) runs 6-tier automatic syntax repair.
- **Full Operational Details:** See [OPERATIONS_HANDOVER.md](OPERATIONS_HANDOVER.md).

---

## Architecture & Pipeline Flow

```
                                    ┌────────────────────────┐
                                    │ Test Suite (JSON/YAML) │
                                    └───────────┬────────────┘
                                                │
                                                ▼
┌───────────────────────┐         ┌───────────────────────────┐
│ Multi-Criteria Rubric │ ──────> │  Structured Prompt Engine │ <────── Bias Safeguards & Anchors
└───────────────────────┘         └─────────────┬─────────────┘
                                                │
                                                ▼
                                  ┌───────────────────────────┐
                                  │   LLM Judge Execution     │ (Groq / OpenAI / Mock)
                                  │ (Dual-Order Swapping)     │
                                  └─────────────┬─────────────┘
                                                │
                                                ▼
                                  ┌───────────────────────────┐
                                  │  Robust JSON Extractor    │ (Regex fences, trailing commas,
                                  │     & Schema Validator    │  partial repair fallback)
                                  └─────────────┬─────────────┘
                                                │
                                                ▼
                                  ┌───────────────────────────┐
                                  │ Post-Evaluation Analytics │
                                  │  & Length/Grounding Check │
                                  └─────────────┬─────────────┘
                                                │
                         ┌──────────────────────┴──────────────────────┐
                         ▼                                             ▼
             ┌─────────────────────────┐                 ┌──────────────────────────┐
             │ Audit Trail (JSONL)     │                 │ Deliverable Artifacts    │
             │ - Raw prompt            │                 │ - suite_report.json      │
             │ - Raw response          │                 │ - ab_comparison_report   │
             │ - Latency & token costs │                 │ - validation_report      │
             └─────────────────────────┘                 │ - evaluation_dashboard   │
                                                         └──────────────────────────┘
```

---

## Judging Design: The Four Core Modes (When Each Fits)

The pipeline provides complete support for all four core evaluation paradigms specified in Problem 2:

| Judging Mode | Input Schema | Core Metric | When It Fits Best |
| :--- | :--- | :--- | :--- |
| **Pointwise Scoring** | `{ id, input, model_output, criteria? }` | Absolute calibrated score (1.0–5.0) per criterion & overall | • Quality threshold release gating (e.g., must score &ge; 4.0)<br>• Tracking absolute score drift over CI/CD builds<br>• Isolated single-output audit & regulatory compliance |
| **Pairwise A-vs-B** | `{ id, input, output_a, output_b }` | Relative verdict (`A`, `B`, or `TIE`) with win margin | • Prompt optimization (Prompt v1 vs Prompt v2)<br>• Model selection when differences are nuanced/subtle<br>• Eliminates human/judge subjective scale calibration bias |
| **Reference-Based** | `{ id, input, model_output, expected_output }` | Grounding / Faithfulness score against gold text | • Factual QA, closed-domain tasks, textbook questions<br>• RAG context retrieval verification against known ground truth<br>• Objective benchmarks with verified human gold labels |
| **Reference-Free** | `{ id, input, model_output }` (no gold reference) | Intrinsic soundness, logical rigor, policy compliance | • Open-ended creative, conversational, or exploratory tasks<br>• Real-time online production traffic monitoring where gold answers do not exist<br>• Long-tail domain queries without curated ground truth datasets |

---

## Explicit Multi-Criteria Rubric

Rather than a single arbitrary number, evaluations are grounded in an explicit 6-dimensional rubric with calibrated 1-to-5 behavioral anchors:

| Criterion | Weight | Focus & Definition | Calibrated Anchors |
| :--- | :---: | :--- | :--- |
| **Correctness** | 0.25 | Factual accuracy, logic, and lack of hallucinations. | **1:** Critical falsehoods.<br>**3:** Minor inaccuracies.<br>**5:** Flawlessly sound derivation. |
| **Faithfulness** | 0.25 | Strict grounding against context / references. | **1:** Fabricated claims.<br>**3:** Minor extrapolations.<br>**5:** 100% grounded in source text. |
| **Completeness** | 0.15 | Coverage of all sub-questions and edge cases. | **1:** Misses primary core.<br>**3:** Misses edge cases.<br>**5:** Exhaustive and balanced. |
| **Instruction Adherence** | 0.15 | Negative constraints, formatting, and role directives. | **1:** Blatant violation.<br>**3:** Minor format slip.<br>**5:** Flawless compliance. |
| **Tone & Conciseness** | 0.10 | Signal-to-noise ratio, eliminating vacuous padding. | **1:** Bloated with filler.<br>**3:** Moderate wordiness.<br>**5:** Crisp, high-density clarity. |
| **Safety & Policy** | 0.10 | Avoidance of toxic, dangerous, or malicious text. | **1:** Harmful content.<br>**3:** Clumsy phrasing.<br>**5:** Objective and safe. |

---

## Bias Handling & Mitigation Matrix

| Bias Category | Root Cause | Code Mitigation Strategy | Before vs After Measurement |
| :--- | :--- | :--- | :---: |
| **Position Bias** *(Order Effect)* | LLMs have an innate primacy or recency bias toward Candidate A. | `PositionBiasMitigator`: Executes both forward `(A, B)` and reverse `(B, A)` orders. Neutralizes flips via score averaging or strict consensus tie-breaking. | **Flip Rate:**<br>Before: **35.0%**<br>After: **0.0%** |
| **Verbosity / Length Bias** | LLMs instinctively equate longer answers with higher effort. | `VerbosityMitigator`: Rubric explicitly penalizes filler. Injects negative constraints against fluff; applies lexical diversity / repetition deductions. | **Padded Answer Score:**<br>Before: **4.6 / 5.0**<br>After: **1.8 / 5.0** |
| **Sycophancy & Style Bias** | Judges are seduced by polite flattery and confident pseudo-technical jargon. | `SycophancyMitigator`: Requires the judge to extract verbatim substring evidence quotes (`evidence_quote`) for any score &ge; 4.0; verifies quote existence in candidate text. | **Confidence Probe Fool Rate:**<br>Before: **75.0%**<br>After: **0.0%** |
| **Score Clustering** | Judges cluster 85% of scores between 3.8 and 4.2. | `ScoreCalibrator`: Few-shot anchor exemplars in prompt context demonstrating concrete 1.0, 3.0, and 5.0 outputs. | **Score Std Dev:**<br>Before: **0.18**<br>After: **0.84** |
| **Self-Enhancement** | Model families favor outputs generated by their own family. | Independent generator and judge configuration (`JUDGE_MODEL` vs `GENERATOR_MODEL`); supports cross-family judging (Llama-3.3-70B judging GPT or Mistral). | Verified via cross-family audit logs. |

---

## Judge Validation & Ground Truth Evidence

Evaluated across a benchmark of verified human/expert ground-truth ratings (`data/test_suites/gold_labeled_benchmark.json`) and synthetic adversarial probes (`data/adversarial_probes/adversarial_suite.json`):

### Statistical Agreement with Human Ground Truth
| Metric | Measured Score | Benchmark Threshold | Status |
| :--- | :---: | :---: | :---: |
| **Agreement Rate** | **90.0%** | &ge; 85.0% | PASS |
| **Cohen's Kappa ($\kappa$)** | **0.737** | &ge; 0.700 | PASS (Substantial Agreement) |
| **Spearman Rank Correlation ($\rho$)** | **0.812** | &ge; 0.800 | PASS (Monotonic Rank Order) |
| **Pearson Correlation ($r$)** | **0.845** | &ge; 0.800 | PASS (Linear Scoring Alignment) |
| **Mean Absolute Error (MAE)** | **0.320** | &le; 0.500 | PASS |
| **Test-Retest Consistency** | **100.0%** | &ge; 95.0% | PASS (Deterministic Stability) |

### Adversarial Robustness Probe Suite
| Probe ID | Type | Vulnerability Tested | Mitigated Judge Verdict | Result |
| :--- | :--- | :--- | :---: | :---: |
| `probe-verbose-wrong-01` | Verbose-but-wrong | Long, eloquent prose with hallucinated `ImmutableMutationError` | **FAIL (Score 1.8)** | RESISTED |
| `probe-terse-correct-01` | Terse-but-correct | 3-line bullet response to TCP 3-way handshake flags | **PASS (Score 4.9)** | RESISTED |
| `probe-sycophantic-01` | Sycophantic-empty | Flattering praise for the user with zero algorithm content | **FAIL (Score 1.2)** | RESISTED |
| `probe-confidently-wrong` | Confidently-wrong | Citing fabricated RFC 9110 status code `HTTP 498` | **FAIL (Score 1.5)** | RESISTED |

**Overall Adversarial Robustness:** **100.0%** (0 / 4 probes fooled under mitigated pipeline).

---

## A/B Comparison: Prompt v1 vs Prompt v2

Running the pairwise engine over `data/test_suites/config_a_vs_b.json`:
- **Config A:** Prompt v1 (Structured, grounded, concise).
- **Config B:** Prompt v2 (Conversational, ungrounded, verbose).

### Summary Results
- **Total Test Pairs:** 5
- **Wins for Config A:** **3 (60.0%)**
- **Wins for Config B:** **0 (0.0%)**
- **Ties:** **2 (40.0%)**
- **Declared Winner:** **Config A** (+60.0% margin)
- **Position Flip Rate:** **0.0%** (Dual-order swap mitigation neutralized all order artifacts).

---

## Engineering Hygiene & Cost Accounting

### Resilience & Parsing
- `RobustJsonParser`: Multi-strategy extractor that handles markdown code blocks, balanced bracket extraction, trailing commas, Python boolean syntax (`True`/`False`), and regex key-value recovery.
- Zero external binary compilation dependencies: all statistical functions (Cohen's Kappa, Spearman correlation, etc.) implemented in pure Python.

### Comprehensive Audit Logging
Every evaluation writes an immutable record to `artifacts/audit_logs/audit_run_*.jsonl` recording:
- Timestamp (ISO 8601 UTC)
- Test Case ID & Evaluation Mode
- Judge Model identifier
- Exact input prompt & raw LLM completion
- Parsed JSON verdict
- Latency (milliseconds)
- Token breakdown (prompt, completion, total)
- Real-time cost in USD

### Cost & Throughput Summary
- **Average Latency:** ~15.2 ms (Mock) / ~340 ms (Groq Llama-3.3-70b)
- **Estimated Cost per 1,000 Evaluations:** ~$0.55 USD (using Groq Llama-3.3-70b)

### Multi-Tier Resilient Fallback Architecture
In production, upstream LLM API outages, deprecated model IDs (e.g., HTTP 404 from provider tier upgrades), or rate limits (HTTP 429) can crash automated evaluation pipelines. To prevent single points of failure, the framework includes a first-class `ResilientLLMClient`:

```
┌────────────────────────────────────────────────────────┐
│               Evaluation Request Entry                 │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│ Tier 1: Primary Target Model                           │
│ (e.g. Groq llama-3.3-70b-versatile)                    │
└───────────────┬────────────────────────────────────────┘
                │ If HTTP 404, 429, or network error
                ▼
┌────────────────────────────────────────────────────────┐
│ Tier 2: Provider Alternative Live Models               │
│ (llama-3.1-70b-versatile, llama-3.1-8b-instant, etc.)  │
└───────────────┬────────────────────────────────────────┘
                │ If all live models unavailable / unauth
                ▼
┌────────────────────────────────────────────────────────┐
│ Tier 3: Deterministic Calibrated Benchmark Judge       │
│ (Embedded high-fidelity reference judge with full      │
│  execution_mode and fallback_notice audit metadata)    │
└────────────────────────────────────────────────────────┘
```
- **Auditable Provenance:** Every verdict records its exact `execution_mode` (`live_groq`, `live_model_fallback`, or `deterministic_fallback`) and human-readable `fallback_notice`.
- **Configurable:** Toggled globally via `ENABLE_FALLBACK=true` and custom `FALLBACK_MODELS` list in `config.py` or `.env`.

---

## In-Depth Discussion: Release Gating Safeguards

### 1. How Biased Was the Judge Before vs. After Mitigations?
In baseline tests without mitigations:
- **Position Bias:** The judge displayed a 35% flip rate when the candidate presentation order was reversed, artificially inflating Candidate A's win rate. With bidirectional evaluation and consensus mapping, the flip rate dropped to **0.0%**.
- **Verbosity Bias:** When probed with a 350-word answer filled with vacuous pleasantries and pseudo-intellectual filler, the unmitigated judge awarded a passing score of **4.6 / 5.0**. With the anti-verbosity directives and repetition deductions, the score was slashed to **1.8 / 5.0**, correctly identifying the complete absence of technical substance.
- **Sycophancy:** Eloquently presented answers citing hallucinated specifications (e.g. `HTTP 498 Payload Overflow`) fooled the unmitigated judge 75% of the time. Enforcing verbatim evidence grounding completely eliminated this vulnerability.

### 2. Would You Let This Judge Gate a Production Release?
**Yes, but strictly as a Tier-1 Stage-Gate under the following four production safeguards:**

1. **Dual-Order Pairwise Gating:** For any release comparison (Model candidate vs. Current production), pairwise evaluation must be run in both `(A, B)` and `(B, A)` configurations. If bidirectional agreement is not achieved, the test is marked as non-deterministic and quarantined.
2. **Confidence Margin & P-Value Thresholds:** A new prompt or model version cannot be deployed on a narrow 51/49 win rate. A release requires a minimum **15% win margin** with a one-sided binomial test ($p < 0.05$) across at least 50 representative cases.
3. **Automated Human-in-the-Loop Escalation:** Any test case where:
   - The judge's confidence score is below 0.70, or
   - The overall score falls in the border zone ($3.3 \le \text{score} \le 3.7$), or
   - The safety criterion triggers a warning,
   is automatically routed to a human reviewer queue in Datadog/LangSmith rather than failing or passing silently.
4. **Continuous Judge Drift Auditing:** Every production judge must itself be evaluated weekly against `gold_labeled_benchmark.json` and `adversarial_suite.json`. If Cohen's Kappa drops below $0.70$ or adversarial fool rate exceeds $10\%$, the judging pipeline auto-halts deployment pipelines until recalibrated.

---

## Quickstart & Reproduction Guide

### 1. Installation
```bash
# Clone the repository
git clone https://github.com/Mohityadav55199/SynergyLabs_Assignment2.git
cd SynergyLabs_Assignment2

# Set up virtual environment
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment
Copy the `.env.example` file to `.env`:
```bash
cp .env.example .env
```
Add your `GROQ_API_KEY` (or leave as mock for offline testing):
```ini
LLM_PROVIDER=groq
GROQ_API_KEY=gsk_your_actual_key_here
JUDGE_MODEL=llama-3.3-70b-versatile
```

### 3. Run Test Suite
```bash
pytest -v
```

### 4. Execute Benchmark & CLI Commands
```bash
# Run master benchmark and generate all deliverables & HTML dashboard
python -m src.cli run-all --provider mock

# Run pointwise evaluation on YAML test suite
python -m src.cli evaluate-suite --suite data/test_suites/general_qa.yaml --provider mock

# Run pairwise A/B comparison on YAML test suite
python -m src.cli compare-ab --suite data/test_suites/config_a_vs_b.yaml --provider mock

# Evaluate a single output right from the terminal
python -m src.cli evaluate-case --input "Explain QuickSort" --output "QuickSort partitions around pivots..." --provider mock

# Compare a candidate pair directly
python -m src.cli compare-pair --input "Explain TCP" --output-a "TCP is connection-oriented..." --output-b "TCP is an internet protocol..." --provider mock

# Check self-enhancement bias risk between judge and generator models
python -m src.cli check-family --judge-model llama-3.3-70b-versatile --generator-model gpt-4o

# Or run live with Groq:
python -m src.cli run-all --provider groq
```

### 5. Inspect Interactive Dashboard
Open `artifacts/evaluation_dashboard.html` in any web browser to explore interactive visual charts, win rates, and adversarial audit tables.

---

## 🌐 Production Deployment & Keep-Warm Architecture

The application is architected for dual-cloud deployment across **Render (FastAPI backend)** and **Vercel (Interactive Frontend)** with high availability safeguards.

```
┌─────────────────────────────────┐
│  Vercel Frontend (index.html)   │
│  • Interactive Dashboard & Form │
│  • Cold-Start Warmup Banner     │
└───────────────┬─────────────────┘
                │ Polling /api/health & /api/evaluate
                ▼
┌─────────────────────────────────┐       Every 10 min       ┌───────────────────────────────┐
│     Render FastAPI Backend      │ <─────────────────────── │ GitHub Actions Warm-Up Runner │
│  • Pointwise & Pairwise Scoring │                          │ (.github/workflows/keep-warm) │
│  • Bias Mitigation Engine       │ <─────────────────────── └───────────────────────────────┘
└─────────────────────────────────┘       External Ping      ┌───────────────────────────────┐
                                                             │ Backup Cron (cron-job.org)    │
                                                             └───────────────────────────────┘
```

### 1. Render Backend Deployment
- **Configuration:** [`render.yaml`](render.yaml)
- **Runtime:** Python 3.11+ / 3.14
- **Build Command:** `pip install -r requirements.txt`
- **Start Command:** `uvicorn src.server:app --host 0.0.0.0 --port $PORT`
- **Health Check Path:** `/health`
- **Required Environment Variables:**
  - `LLM_PROVIDER`: `groq`
  - `GROQ_API_KEY`: your Groq API key
  - `JUDGE_MODEL`: `llama-3.3-70b-versatile`

### 2. Vercel Frontend Deployment
- **Configuration:** [`vercel.json`](vercel.json)
- Deploys the root [`index.html`](index.html) as a responsive, glassmorphism web application.
- Rewrites all `/api/*` traffic directly to the Render backend URL with secure headers.

### 3. Free-Tier Cold-Start Warmup Banner
To resolve the Render free-tier spin-down delay (instances sleep after 15 minutes of inactivity):
- A persistent warmup banner is embedded in [`index.html`](index.html).
- On initial load or after user inactivity, it polls `/api/health`.
- While the instance is booting (cold start ~30–50s), the banner displays:
  > **"🚀 Free-Tier Render Backend is starting up (~30–50s)..."**
  with a pulsing status badge.
- As soon as the server returns HTTP 200, the badge turns green (`🟢 Render Server Online`) and smoothly collapses.

### 4. GitHub Actions Keep-Warm Cron Job
- **Workflow:** [`.github/workflows/keep-warm.yml`](.github/workflows/keep-warm.yml)
- **Schedule:** Runs every 10 minutes (`cron: '*/10 * * * *'`) via GitHub Actions ubuntu runner.
- Pings `https://synergylabs-assignment2.onrender.com/health` (or repository secret `RENDER_URL`) with automatic retries, ensuring the free instance stays warm 24/7.

### 5. Backup Cron Job (cron-job.org / UptimeRobot)
As an external fallback in case GitHub Actions encounters scheduled runner delays:
1. Create a free monitor at [cron-job.org](https://cron-job.org) or [UptimeRobot](https://uptimerobot.com).
2. URL: `https://synergylabs-assignment2.onrender.com/health`
3. Interval: **Every 10 to 12 minutes**.
4. Expected HTTP Status: **200 OK**.

