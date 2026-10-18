import json
from pathlib import Path
from datetime import datetime, timezone
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from typing import Dict, Any, Optional

from .config import config, BASE_DIR
import logging
from .schema import TestCase, PairwiseTestCase
from .judge.pointwise import PointwiseJudge
from .judge.pairwise import PairwiseJudge
from .llm_clients.factory import get_llm_client
from .llm_clients.groq_client import GroqClient
from .llm_clients.mock_client import MockLLMClient

logger = logging.getLogger(__name__)

class EvaluationRequest(BaseModel):
    id: Optional[str] = None
    domain: Optional[str] = "General"
    input: str
    model_output: str
    expected_output: Optional[str] = None
    criteria: Optional[list] = None
    reference_mode: Optional[str] = "auto"
    api_key: Optional[str] = None
    provider: Optional[str] = None

class PairwiseEvaluationRequest(BaseModel):
    id: Optional[str] = None
    domain: Optional[str] = "General"
    input: str
    output_a: str
    output_b: str
    expected_output: Optional[str] = None
    api_key: Optional[str] = None
    provider: Optional[str] = None

app = FastAPI(
    title="LLM-as-Judge Evaluation & Bias Mitigation API",
    description="Production-grade API for automated LLM output quality scoring, pairwise A/B testing, and bias detection.",
    version="1.0.0"
)

# Enable CORS for Vercel and local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ARTIFACTS_DIR = config.artifacts_dir

@app.get("/health", summary="Health Check")
@app.get("/api/health", summary="Health Check for Keep-Warm and UI Banner")
def health_check():
    """
    Used by Render health checks, GitHub Actions keep-warm cron jobs,
    and the Vercel frontend to detect and recover from cold starts.
    """
    return {
        "status": "healthy",
        "service": "llm-judge-pipeline",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": "1.0.0",
        "provider": config.llm_provider,
        "judge_model": config.judge_model
    }

@app.get("/", response_class=HTMLResponse, summary="Serve Visual Dashboard")
def serve_dashboard():
    index_file = BASE_DIR / "index.html"
    if index_file.exists():
        with open(index_file, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    dashboard_file = ARTIFACTS_DIR / "evaluation_dashboard.html"
    if dashboard_file.exists():
        with open(dashboard_file, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>Dashboard generating... Please visit /api/health</h1>")

@app.get("/api/reports/dashboard", summary="Get Full Evaluation Data")
def get_dashboard_data():
    """Returns aggregated reports across pointwise, pairwise A/B, and validation suites."""
    suite_file = ARTIFACTS_DIR / "suite_report.json"
    ab_file = ARTIFACTS_DIR / "ab_comparison_report.json"
    val_file = ARTIFACTS_DIR / "judge_validation_report.json"
    adv_file = ARTIFACTS_DIR / "adversarial_report.json"

    def read_json_safe(path: Path) -> Dict[str, Any]:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    return {
        "suite_report": read_json_safe(suite_file),
        "ab_report": read_json_safe(ab_file),
        "validation_report": read_json_safe(val_file),
        "adversarial_report": read_json_safe(adv_file),
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

@app.post("/api/evaluate", summary="Run Pointwise Evaluation")
def evaluate_case(request: EvaluationRequest):
    test_case = TestCase(
        id=request.id or f"live-test-{int(datetime.now(timezone.utc).timestamp())}",
        domain=request.domain or "General",
        input=request.input,
        model_output=request.model_output,
        expected_output=request.expected_output,
        criteria=request.criteria,
        reference_mode=request.reference_mode or "auto"
    )
    try:
        client = get_llm_client(
            provider=request.provider,
            api_key=request.api_key,
            enable_fallback=True
        )
        judge = PointwiseJudge(llm_client=client)
        verdict = judge.evaluate_case(test_case)
        return verdict.model_dump()
    except Exception as exc:
        logger.error(f"Evaluation failed: {exc}")
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(exc)}")

@app.post("/api/compare", summary="Run Pairwise Comparison with Position Bias Check")
def compare_pair(request: PairwiseEvaluationRequest):
    test_case = PairwiseTestCase(
        id=request.id or f"live-pair-{int(datetime.now(timezone.utc).timestamp())}",
        domain=request.domain or "General",
        input=request.input,
        output_a=request.output_a,
        output_b=request.output_b,
        expected_output=request.expected_output,
    )
    try:
        client = get_llm_client(
            provider=request.provider,
            api_key=request.api_key,
            enable_fallback=True
        )
        judge = PairwiseJudge(llm_client=client, enable_position_swap=True)
        verdict = judge.evaluate_pair(test_case)
        return verdict.model_dump()
    except Exception as exc:
        logger.error(f"Comparison failed: {exc}")
        raise HTTPException(status_code=500, detail=f"Comparison failed: {str(exc)}")

