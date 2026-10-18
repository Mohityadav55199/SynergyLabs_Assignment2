# Production Operations & Handover Manual
**System:** Autonomous LLM-as-Judge Evaluation & Bias Mitigation Engine  
**Repository:** [SynergyLabs_Assignment2](https://github.com/Mohityadav55199/SynergyLabs_Assignment2)  
**Target Audience:** Engineering Leads, Client Stakeholders, DevOps / SRE, and QA Engineers  
**Status:** Production-Ready & Deployed

---

## 📑 Table of Contents
1. [System Overview & Running Topology](#1-system-overview--running-topology)
2. [External Service Ownership Matrix](#2-external-service-ownership-matrix)
3. [Environment Configuration & Secrets](#3-environment-configuration--secrets)
4. [Deployment Architecture & Procedures](#4-deployment-architecture--procedures)
5. [Operational Runbook & Troubleshooting ("What If Something Breaks")](#5-operational-runbook--troubleshooting)
6. [Known Limitations & Boundary Conditions](#6-known-limitations--boundary-conditions)
7. [Security, Access Control & Compliance](#7-security-access-control--compliance)
8. [Maintenance, Drift Monitoring & Scaling](#8-maintenance-drift-monitoring--scaling)

---

## 1. System Overview & Running Topology

The system evaluates, compares, and audits Large Language Model (LLM) outputs using structured LLM judges with active bias mitigations (Position, Verbosity, Sycophancy, and Self-Enhancement).

```
                                 ┌───────────────────────────────┐
                                 │   Vercel Edge CDN             │
                                 │   https://synergylabs...      │
                                 │   (Interactive Glassmorphism) │
                                 └───────────────┬───────────────┘
                                                 │
                                                 │ HTTPS REST (/api/evaluate, /api/compare)
                                                 ▼
┌───────────────────────────────┐        ┌───────────────────────────────┐
│ GitHub Actions Cron Runner    │ ──10m─>│ Render Web Service (FastAPI)  │
│ (.github/workflows/keep-warm) │        │ https://synergylabs...onrender│
└───────────────────────────────┘        └───────────────┬───────────────┘
                                                         │
                                    ┌────────────────────┴────────────────────┐
                                    ▼                                         ▼
                     ┌───────────────────────────────┐         ┌───────────────────────────────┐
                     │ Tier 1: Primary Target Model  │         │ Audit & Persistence Layer     │
                     │ (Groq llama-3.3-70b-versatile)│         │ - artifacts/audit_logs/*.jsonl│
                     └──────────────┬────────────────┘         │ - artifacts/*_report.json     │
                                    │ Failover                 │ - artifacts/dashboard.html    │
                                    ▼                          └───────────────────────────────┘
                     ┌───────────────────────────────┐
                     │ Tier 2: Live Fallback Models  │
                     │ (llama-3.1-70b, 8b-instant)   │
                     └──────────────┬────────────────┘
                                    │ Failover
                                    ▼
                     ┌───────────────────────────────┐
                     │ Tier 3: Deterministic Judge   │
                     │ (MockLLMClient offline engine)│
                     └───────────────────────────────┘
```

### Components:
1. **Interactive Frontend (Vercel):** Serves [`index.html`](index.html) with live metrics dashboard, global cross-tab search, real-time rubric explorer, and interactive pointwise/pairwise playground.
2. **REST API Backend (Render):** FastAPI application ([`src/server.py`](src/server.py)) exposing evaluation endpoints, health probes, and report aggregators.
3. **Core Evaluation Engine (`src/judge/`):** Executes dual-order position swapping, verbosity penalty deductions, evidence quotation checks, and 6-criterion rubric scoring.
4. **Resilient LLM Client (`src/llm_clients/`):** Multi-tier fallback manager wrapping live providers (Groq/OpenAI) and offline deterministic fallback.
5. **Auditing Subsystem (`src/logging/`):** Writes append-only JSONL audit logs with exact prompts, completions, token usage, latency, and costs.

---

## 2. External Service Ownership Matrix

Every external service used by this project has explicit ownership boundaries, credential scopes, and escalation channels:

| External Service | Role / Function | Service Tier | Responsible Party / Owner | What It Controls | Escalation / Console |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Groq Cloud** | Primary & secondary live LLM judge inference | Free / Pay-as-you-go | **Client API Admin** | `GROQ_API_KEY`, model rate limits (RPM/TPM), token billing | [Groq Console](https://console.groq.com) |
| **Render.com** | API server hosting (Docker / Python Uvicorn) | Free Web Service | **DevOps / SysAdmin** | Backend server runtime, auto-deploy branch, keep-warm health status | [Render Dashboard](https://dashboard.render.com) |
| **Vercel** | Global CDN frontend hosting & reverse proxy | Hobby / Pro | **Frontend Lead** | DNS, SSL certificates, edge rewrites, client asset caching | [Vercel Dashboard](https://vercel.com) |
| **GitHub** | Code repository, version control, CI/CD keep-warm | Standard Org / User | **Lead Developer** | Repository secrets (`RENDER_URL`), keep-warm workflow scheduling | [GitHub Actions](https://github.com/Mohityadav55199/SynergyLabs_Assignment2/actions) |
| **OpenAI (Optional)** | Alternative judge provider fallback | Commercial API | **Client API Admin** | `OPENAI_API_KEY`, GPT-4o / GPT-4o-mini quota | [OpenAI Platform](https://platform.openai.com) |

---

## 3. Environment Configuration & Secrets

Configuration is managed via environment variables defined in `.env` (loaded via Pydantic settings in [`src/config.py`](src/config.py)):

| Variable | Type | Default | Required in Production? | Description |
| :--- | :--- | :--- | :--- | :--- |
| `LLM_PROVIDER` | `str` | `groq` | Yes | Active provider: `groq`, `openai`, or `mock`. |
| `GROQ_API_KEY` | `str` | `None` | Yes (if `LLM_PROVIDER=groq`) | Groq API authentication token (`gsk_...`). |
| `OPENAI_API_KEY` | `str` | `None` | No | Optional OpenAI key if using OpenAI models. |
| `JUDGE_MODEL` | `str` | `llama-3.3-70b-versatile` | Yes | Target model ID for judging evaluation. |
| `GENERATOR_MODEL` | `str` | `llama-3.1-8b-instant` | No | Candidate generator model (used for self-enhancement checks). |
| `JUDGE_TEMPERATURE` | `float` | `0.0` | Recommended | Deterministic scoring temperature (always `0.0` in prod). |
| `JUDGE_MAX_TOKENS` | `int` | `2048` | Recommended | Max completion tokens allocated for structured JSON verdict. |
| `ENABLE_POSITION_SWAP_MITIGATION` | `bool` | `true` | Yes | Evaluates pairwise inputs in both `(A,B)` and `(B,A)` orders. |
| `ENABLE_VERBOSITY_PENALTY` | `bool` | `true` | Yes | Penalizes answers containing word padding without information density. |
| `ENABLE_EVIDENCE_GROUNDING` | `bool` | `true` | Yes | Enforces verbatim evidence citations for top scores. |
| `ENABLE_FALLBACK` | `bool` | `true` | Yes | Activates `ResilientLLMClient` cascading on 404/429 errors. |
| `FALLBACK_MODELS` | `list` | `llama-3.1-70b-versatile,...` | Yes | Comma-separated alternative models to try before mock fallback. |
| `PASS_THRESHOLD` | `float` | `3.5` | Yes | Minimum overall score (out of 5.0) required to pass release gate. |
| `CONFIDENCE_THRESHOLD` | `float` | `0.7` | Yes | Minimum agreement confidence threshold before human escalation. |

---

## 4. Deployment Architecture & Procedures

### Initial Setup & Local Reproduction
```bash
# 1. Clone repository
git clone https://github.com/Mohityadav55199/SynergyLabs_Assignment2.git
cd SynergyLabs_Assignment2

# 2. Virtual environment setup
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# 3. Install pinned dependencies
pip install -r requirements.txt

# 4. Copy configuration
cp .env.example .env
# Edit .env with your GROQ_API_KEY

# 5. Run test suite (31 unit & integration tests)
pytest -v
```

### Deploying the Backend on Render
1. Connect GitHub repository `SynergyLabs_Assignment2` to Render.
2. Select **New Web Service** and link [`render.yaml`](render.yaml).
3. Set environment variable `GROQ_API_KEY` under the Service Settings.
4. Auto-deploy triggers on every push to branch `main`.
5. Verify health: `curl https://<your-render-app>.onrender.com/health`.

### Deploying the Frontend on Vercel
1. Import repository on Vercel.
2. Root directory: `./` (Vercel automatically detects [`vercel.json`](vercel.json) and [`index.html`](index.html)).
3. No build command needed (pure modern HTML/CSS/JS with zero bundle overhead).
4. Verify deployment: visit your Vercel URL.

---

## 5. Operational Runbook & Troubleshooting

### Scenario A: Render Service Sleeping / 502 Bad Gateway / Request Timeout
- **Symptoms:** Frontend banner displays `🚀 Render Backend is starting up...` or requests take >30 seconds.
- **Root Cause:** Render free-tier instances spin down to zero after 15 minutes of inactivity.
- **Immediate Resolution:**
  1. The integrated frontend polling detector will auto-recover once the instance boots (~30–50s).
  2. Verify GitHub Action keep-warm workflow is active: [GitHub Actions Workflows](https://github.com/Mohityadav55199/SynergyLabs_Assignment2/actions/workflows/keep-warm.yml).
  3. If sleeping frequently, add a secondary free monitor on [UptimeRobot](https://uptimerobot.com) or [cron-job.org](https://cron-job.org) hitting `https://<render-url>/health` every 10 minutes.
  4. For mission-critical internal tooling, upgrade Render to the $7/month Starter plan (disables instance sleep).

### Scenario B: Upstream LLM Returns 404 (Model Deprecated) or 429 (Rate Limit)
- **Symptoms:** Live evaluation response carries `🛡️ Calibrated Fallback Judge` badge and `execution_mode: "live_model_fallback"` or `"deterministic_fallback"`.
- **Root Cause:** Provider tier upgrades or daily token allowances exhausted.
- **Immediate Resolution:**
  1. Check `artifacts/audit_logs/` for exact error message logged.
  2. The multi-tier [`ResilientLLMClient`](src/llm_clients/resilient_client.py) automatically prevented an application crash by cascading to `llama-3.1-70b-versatile` or calibrated benchmark fallback.
  3. Update `JUDGE_MODEL` in Render environment variables to an active model ID.
  4. Check Groq usage dashboard to increase rate limits or rotate API keys.

### Scenario C: JSON Parsing / Schema Validation Failure
- **Symptoms:** LLM completion does not match strict JSON schema.
- **Root Cause:** Upstream model generated conversational preamble or truncated tokens.
- **Resolution:**
  - Handled automatically by [`RobustJsonParser`](src/parser.py) across 6 recovery strategies (markdown extraction, bracket balancing, trailing comma strip, Python bool normalization, quote repair, and heuristic regex extraction).
  - If output was severely truncated, increase `JUDGE_MAX_TOKENS` in `.env`.

### Scenario D: Position Flip Detected in Pairwise Evaluation
- **Symptoms:** `flip_detected: true` in PairwiseVerdict; candidate won in order `(A, B)` but candidate swapped winner in `(B, A)`.
- **Resolution:**
  - The pipeline automatically designates these as non-deterministic and maps the outcome to `TIE` rather than falsely declaring a winner.
  - Review the comparative rationales in `artifacts/audit_logs/` and consider adding more explicit scoring anchors for subtle differentiators.

---

## 6. Known Limitations & Boundary Conditions

1. **Context Window Limits:**
   - Prompts larger than 8,000 tokens may exceed default token allocations. For long document evaluation, split input into semantic sections before judging.
2. **Cold-Start Latency on Free Tier:**
   - Without keep-warm pings, initial cold boot is ~35–50 seconds on Render.
3. **Upstream Model Lifecycle:**
   - Open-source model providers (Groq) periodically retire older model checkpoints (e.g., Llama 3.0 &rarr; 3.1 &rarr; 3.3). The system's multi-tier fallback protects against downtime, but operators should audit model versions quarterly.
4. **Pure Python Statistical Metrics:**
   - Statistical calculations (Cohen's Kappa, Spearman $\rho$, Pearson $r$) are written in pure Python without `scipy` or C-extensions to guarantee zero-compilation deployment across Windows, macOS, and Linux containers. This is optimized for batch sizes &le; 10,000 pairs.

---

## 7. Security, Access Control & Compliance

1. **Zero Secret Leakage:**
   - API keys are never written to audit files, client logs, or Git commits.
   - All `.env` files are excluded via [`.gitignore`](.gitignore).
2. **Data Privacy:**
   - Audit logs store evaluation inputs and outputs in local append-only files (`artifacts/audit_logs/`).
   - If processing sensitive PII or HIPAA-governed records, ensure local hosting of the audit directory or configure encrypted object storage mounts.
3. **CORS & Network Boundaries:**
   - Configured with strict headers in [`vercel.json`](vercel.json) (`X-Content-Type-Options: nosniff`) and explicit CORS origins in [`src/server.py`](src/server.py).

---

## 8. Maintenance, Drift Monitoring & Scaling

### Weekly Operational Checklist:
- [ ] Run benchmark verification: `python -m src.cli run-all --provider groq`
- [ ] Verify Cohen's Kappa remains $\ge 0.70$ against `data/gold_labeled_benchmark.json`
- [ ] Inspect total monthly token costs in Groq dashboard.
- [ ] Review adversarial probe resistance rate (must be $\ge 75\%$).
- [ ] Archive historical audit logs in `artifacts/audit_logs/` if directory exceeds 500 MB.

### Handover Sign-off Contacts:
- **Lead Developer:** Mohit Yadav ([GitHub Profile](https://github.com/Mohityadav55199))
- **Repository:** [SynergyLabs_Assignment2](https://github.com/Mohityadav55199/SynergyLabs_Assignment2)
- **License:** MIT License (Full commercial and internal redistribution rights granted)
