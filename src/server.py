"""FastAPI wrapper around the `judge` package (live LLM judge). No verdict is ever fabricated: if the judge model is unreachable
or returns something unparseable, the endpoint answers HTTP 502 with the reason."""
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from judge.config import JudgeConfig
from judge.core import Judge
from judge.llm import LLMClient

logger = logging.getLogger(__name__)
BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "judge" / "results"
WEB_LOG = Path(os.getenv("JUDGE_WEB_LOG", str(BASE_DIR / "judge" / "logs" / "web_calls.jsonl")))


class EvaluationRequest(BaseModel):
    id: Optional[str] = None
    input: str
    model_output: str
    system_prompt: Optional[str] = None
    expected_output: Optional[str] = None
    criteria: Optional[List[str]] = None
    api_key: Optional[str] = None


class PairwiseEvaluationRequest(BaseModel):
    id: Optional[str] = None
    input: str
    output_a: str
    output_b: str
    system_prompt: Optional[str] = None
    api_key: Optional[str] = None


app = FastAPI(title="LLM-as-Judge Evaluation API", version="2.0.0",
              description="Pointwise and both-order pairwise judging with an explicit rubric. Results of the bias/validation experiments are in /api/reports/dashboard.")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=False, allow_methods=["*"], allow_headers=["*"])


def _judge(api_key: Optional[str]) -> Judge:
    cfg = JudgeConfig()
    if api_key:
        cfg.api_key = api_key.strip()   # per-request key is used for this call only and is never logged
    return Judge(LLMClient(cfg, log_path=WEB_LOG, fail_fast_wait=20.0), cfg, "mitigated")   # fail fast on quota 429s so the fallback model answers within the request


@app.get("/health")
@app.get("/api/health")
def health_check():
    cfg = JudgeConfig()
    return {"status": "healthy", "service": "llm-judge-pipeline", "timestamp": datetime.now(timezone.utc).isoformat(), "version": app.version,
            "provider": "groq", "judge_model": cfg.judge_model, "judge_chain": cfg.judge_chain, "generator_model": cfg.generator_model, "api_key_configured": bool(cfg.api_key)}


@app.get("/", response_class=HTMLResponse)
def serve_ui():
    index_file = BASE_DIR / "index.html"
    return HTMLResponse(index_file.read_text(encoding="utf-8")) if index_file.exists() else HTMLResponse("<h1>LLM-as-Judge API</h1><p>See /docs</p>")


def _notice(judge: Judge, v: dict) -> Optional[str]:
    """Honest provenance: say when a different LIVE model than the configured judge answered. There is no mock fallback."""
    if v.get("fallback_used"):
        return (f"Primary judge '{judge.config.judge_model}' was unavailable (quota or error); this verdict was produced by live fallback model "
                f"'{v['judge_model_used']}'. Fallback verdicts are not covered by the validation of the primary judge.")
    return None


def _strip(obj: Any) -> Any:
    """Drop bulky per-case arrays so the dashboard payload stays small."""
    if isinstance(obj, dict):
        return {k: _strip(v) for k, v in obj.items() if k not in ("verdicts", "pairs", "cases", "rows", "pointwise_concise", "pointwise_padded", "pointwise_a", "pointwise_b", "rubric", "usage_this_process")}
    return obj


@app.get("/api/reports/dashboard", summary="Measured results of the bias / validation experiments (from judge/results/*.json)")
def get_dashboard_data():
    out: Dict[str, Any] = {}
    if RESULTS_DIR.exists():
        for f in sorted(RESULTS_DIR.glob("*.json")):
            try:
                out[f.stem] = _strip(json.loads(f.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError):
                continue
    return {"results": out, "timestamp": datetime.now(timezone.utc).isoformat()}


def _require_text(**fields):
    for name, value in fields.items():
        if not value or not value.strip():
            raise HTTPException(status_code=400, detail=f"Field '{name}' cannot be empty or whitespace.")


@app.post("/api/evaluate", summary="Pointwise evaluation (reference-based when expected_output is given, else reference-free)")
def evaluate_case(request: EvaluationRequest):
    _require_text(input=request.input, model_output=request.model_output)
    judge = _judge(request.api_key)
    case = {"id": request.id or f"live-{int(time.time())}", "input": request.input.strip(), "system_prompt": request.system_prompt,
            "model_output": request.model_output.strip(), "expected_output": (request.expected_output or "").strip() or None, "criteria": request.criteria}
    t0 = time.perf_counter()
    v = judge.judge_case(case, run_tag=str(time.time_ns()))   # unique tag: live requests must never be answered from the cache
    if v["status"] != "ok":
        raise HTTPException(status_code=502, detail=f"Judge produced no usable verdict (status={v['status']}: {v['error']}). No score was invented.")
    return {"test_case_id": case["id"], "overall_score": v["overall"], "passed": v["passed"], "mode": v["mode"],
            "criterion_scores": {k: c["score"] for k, c in v["criteria"].items()},
            "criterion_verdicts": [{"criterion_id": k, "score": c["score"], "rationale": c["rationale"], "evidence_quote": c["evidence"], "grounded": c["grounded"]} for k, c in v["criteria"].items()],
            "overall_rationale": v["overall_rationale"], "judge_model": v["judge_model_used"], "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
            "total_tokens": v["tokens"], "execution_mode": "live_fallback_model" if v["fallback_used"] else "live", "fallback_notice": _notice(judge, v), "ungrounded_capped": v["ungrounded_capped"]}


@app.post("/api/compare", summary="Pairwise A-vs-B in BOTH orders (position-bias check)")
def compare_pair(request: PairwiseEvaluationRequest):
    _require_text(input=request.input, output_a=request.output_a, output_b=request.output_b)
    judge = _judge(request.api_key)
    case = {"id": request.id or f"live-pair-{int(time.time())}", "input": request.input.strip(), "system_prompt": request.system_prompt,
            "output_a": request.output_a.strip(), "output_b": request.output_b.strip()}
    r = judge.judge_pair(case, run_tag=str(time.time_ns()))
    if r["status"] != "ok":
        errs = "; ".join(f"{o}: {r['orders'][o].get('error')}" for o in ("AB", "BA") if r["orders"][o]["status"] != "ok")
        raise HTTPException(status_code=502, detail=f"Judge produced no usable verdict ({errs}). No winner was invented.")
    ab, ba = r["orders"]["AB"], r["orders"]["BA"]
    return {"test_case_id": case["id"], "final_winner": r["final"], "flip_detected": not r["agree"], "hard_flip": r["hard_flip"], "single_order_winner": r["single_order"],
            "forward_evaluation": {"winner": ab["winner"], "judge_model": judge.config.judge_model, "comparative_rationale": ab.get("rationale")},
            "reverse_evaluation": {"winner": ba["winner"], "judge_model": judge.config.judge_model, "comparative_rationale": ba.get("rationale")},
            "overall_rationale": ab.get("rationale"), "total_tokens": r["tokens"], "judge_model": ab.get("judge_model_used"),
            "execution_mode": "live_fallback_model" if (ab.get("fallback_used") or ba.get("fallback_used")) else "live", "fallback_notice": _notice(judge, ab if ab.get("fallback_used") else ba)}
