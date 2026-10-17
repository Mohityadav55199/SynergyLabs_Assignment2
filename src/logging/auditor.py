import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional
from ..config import config

logger = logging.getLogger(__name__)

class AuditLogger:
    """
    Structured audit trail logger for LLM judges.
    Records every raw prompt, raw response, parsed verdict, token usage, latency, and cost.
    Ensures 100% auditable and replayable judge runs.
    """
    def __init__(self, log_dir: Optional[Path] = None):
        self.log_dir = log_dir or config.audit_logs_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        session_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"audit_run_{session_id}.jsonl"

    def log_judge_call(
        self,
        test_case_id: str,
        evaluation_mode: str,
        prompt: str,
        raw_response: str,
        parsed_verdict: Dict[str, Any],
        judge_model: str,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
        latency_ms: float,
        cost_usd: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "test_case_id": test_case_id,
            "evaluation_mode": evaluation_mode,
            "judge_model": judge_model,
            "latency_ms": latency_ms,
            "tokens": {
                "prompt": prompt_tokens,
                "completion": completion_tokens,
                "total": total_tokens,
            },
            "cost_usd": cost_usd,
            "prompt": prompt,
            "raw_response": raw_response,
            "parsed_verdict": parsed_verdict,
            "metadata": metadata or {},
        }

        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.error(f"Failed to write to audit log: {e}")

auditor = AuditLogger()
