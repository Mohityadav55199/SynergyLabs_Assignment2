import json
import logging
from pathlib import Path
from typing import Dict, Any, List
from ..schema import AdversarialProbeCase, AdversarialProbeResult, AdversarialReport, TestCase
from ..judge.pointwise import PointwiseJudge
from ..llm_clients.mock_client import MockLLMClient
from ..config import config
from ..utils import load_suite_data

logger = logging.getLogger(__name__)

class AdversarialProbeRunner:
    """
    Executes adversarial probe suites (verbose-but-wrong, terse-but-correct,
    sycophantic-empty, confidently-wrong) against the judge.
    Reports fool rate and robustness metrics before and after mitigations.
    """

    def __init__(self, judge: PointwiseJudge):
        self.judge = judge

    def run_probes(self, probes_path: Path) -> AdversarialReport:
        raw_cases = load_suite_data(probes_path)
        probe_cases = [AdversarialProbeCase.model_validate(c) for c in raw_cases]
        results: List[AdversarialProbeResult] = []
        by_type: Dict[str, Dict[str, Any]] = {}

        fooled_count = 0

        for probe in probe_cases:
            tc = TestCase(
                id=probe.id,
                domain="Adversarial Probe",
                input=probe.input,
                model_output=probe.model_output
            )
            v = self.judge.evaluate_case(tc)

            actual_verdict = "PASS" if v.passed else "FAIL"
            fooled = (actual_verdict != probe.expected_verdict)
            if fooled:
                fooled_count += 1

            p_result = AdversarialProbeResult(
                probe_id=probe.id,
                probe_type=probe.probe_type,
                judge_score=v.overall_score,
                judge_passed=v.passed,
                expected_verdict=probe.expected_verdict,
                fooled_judge=fooled,
                rationale=v.overall_rationale
            )
            results.append(p_result)

            if probe.probe_type not in by_type:
                by_type[probe.probe_type] = {"total": 0, "fooled": 0, "robustness": 0.0}
            by_type[probe.probe_type]["total"] += 1
            if fooled:
                by_type[probe.probe_type]["fooled"] += 1

        total_probes = len(results)
        robustness_rate = round((total_probes - fooled_count) / max(total_probes, 1), 4)

        for pt, data in by_type.items():
            data["robustness"] = round((data["total"] - data["fooled"]) / max(data["total"], 1), 4)

        return AdversarialReport(
            total_probes=total_probes,
            total_fooled=fooled_count,
            robustness_rate=robustness_rate,
            by_type=by_type,
            results=results
        )
