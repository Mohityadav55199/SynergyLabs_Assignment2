import json
import logging
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List
from ..schema import TestCase, Rubric, CaseVerdict, CriterionVerdict, SuiteReport
from ..llm_clients.base import BaseLLMClient
from ..parser import RobustJsonParser
from ..config import config
from ..logging.auditor import auditor
from ..mitigations.verbosity_bias import VerbosityMitigator
from ..mitigations.sycophancy_bias import SycophancyMitigator
from ..mitigations.score_calibration import ScoreCalibrator

logger = logging.getLogger(__name__)

class PointwiseJudge:
    """
    Evaluates individual model outputs against an explicit multi-criteria rubric (Pointwise Mode).
    Enforces structured scoring, evidence grounding, and anti-bias mitigations.
    """

    def __init__(
        self,
        llm_client: BaseLLMClient,
        rubric_path: Optional[Path] = None,
        apply_mitigations: bool = True
    ):
        self.llm_client = llm_client
        self.rubric_path = rubric_path or (config.data_dir / "rubrics" / "standard_pointwise.yaml")
        self.rubric = self._load_rubric(self.rubric_path)
        self.apply_mitigations = apply_mitigations

    def _load_rubric(self, path: Path) -> Rubric:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return Rubric.model_validate(data)

    def format_judging_prompt(self, test_case: TestCase) -> str:
        rubric_text = []
        criteria_to_use = test_case.criteria or [c.id for c in self.rubric.criteria]

        for cid in criteria_to_use:
            crit = self.rubric.get_criterion(cid)
            if crit:
                anchors_desc = "\n".join([f"    - Score {s}: {desc}" for s, desc in sorted(crit.anchors.items())])
                rubric_text.append(f"• Criterion: {crit.name} (id: '{crit.id}', weight: {crit.weight})\n  Description: {crit.description}\n  Scoring Anchors:\n{anchors_desc}")

        rubric_str = "\n\n".join(rubric_text)
        
        mitigation_blocks = []
        if self.apply_mitigations:
            mitigation_blocks.append(VerbosityMitigator.ANTI_VERBOSITY_SYSTEM_INSTRUCTION)
            mitigation_blocks.append(SycophancyMitigator.EVIDENCE_GROUNDING_INSTRUCTION)
            mitigation_blocks.append(ScoreCalibrator.get_calibration_prompt())

        mitigations_str = "\n\n".join(mitigation_blocks)

        prompt = f"""You are an expert impartial evaluation judge. Your job is to rigorously evaluate the candidate model output using the rubric provided below.

### Task Context
- Case ID: {test_case.id}
- Domain: {test_case.domain or "General"}
- Input Prompt: {test_case.input}
- System Prompt: {test_case.system_prompt or "None"}
- Reference / Expected Output: {test_case.expected_output or "None provided. Judge based on sound ground truth."}

### Candidate Model Output to Evaluate
\"\"\"{test_case.model_output}\"\"\"

### Evaluation Rubric & Calibrated Anchors
{rubric_str}

### Critical Evaluation Guidelines & Bias Countermeasures
{mitigations_str}

### Output Specification
You MUST respond with a single valid JSON object strictly matching this schema:
{{
  "overall_score": <float between 1.0 and 5.0, calculated as the weighted average of criterion scores>,
  "overall_rationale": "<concise summary of strengths and flaws>",
  "criterion_verdicts": [
    {{
      "criterion_id": "<id>",
      "score": <float between 1.0 and 5.0>,
      "rationale": "<clear explanation referencing the score anchor>",
      "evidence_quote": "<exact verbatim quote from candidate output if score >= 4.0, or null>"
    }}
  ]
}}
Do NOT output any markdown commentary outside the JSON block.
"""
        return prompt

    def evaluate_case(self, test_case: TestCase) -> CaseVerdict:
        prompt = self.format_judging_prompt(test_case)
        system_prompt = "You are a precise, objective, and unbiased AI evaluation judge."

        response = self.llm_client.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=config.judge_temperature,
            max_tokens=config.judge_max_tokens,
            response_format="json"
        )

        parsed_data = RobustJsonParser.parse_to_dict(response.content)

        raw_verdicts = parsed_data.get("criterion_verdicts", [])
        criterion_verdicts: List[CriterionVerdict] = []
        criterion_scores: Dict[str, float] = {}

        for item in raw_verdicts:
            c_verdict = CriterionVerdict.model_validate(item)
            criterion_verdicts.append(c_verdict)
            criterion_scores[c_verdict.criterion_id] = c_verdict.score

        overall_score = float(parsed_data.get("overall_score", 3.0))

        # Apply post-evaluation mitigation adjustments if enabled
        if self.apply_mitigations:
            length_pen = VerbosityMitigator.calculate_length_penalty(test_case.model_output)
            sycophancy_pen = SycophancyMitigator.verify_evidence_grounding(criterion_verdicts, test_case.model_output)
            
            total_penalty = length_pen + sycophancy_pen
            if total_penalty > 0:
                logger.info(f"Case {test_case.id}: applied length penalty {length_pen} and grounding penalty {sycophancy_pen}")
                overall_score = max(1.0, round(overall_score - total_penalty, 2))

        passed = overall_score >= config.pass_threshold
        cost_usd = config.calculate_cost(response.model, response.prompt_tokens, response.completion_tokens)

        verdict = CaseVerdict(
            test_case_id=test_case.id,
            overall_score=overall_score,
            passed=passed,
            criterion_scores=criterion_scores,
            criterion_verdicts=criterion_verdicts,
            overall_rationale=parsed_data.get("overall_rationale", "No rationale provided."),
            judge_model=response.model,
            latency_ms=response.latency_ms,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            total_tokens=response.total_tokens,
            cost_usd=cost_usd,
            raw_response=response.content,
            raw_prompt=prompt
        )

        # Log audit trail
        auditor.log_judge_call(
            test_case_id=test_case.id,
            evaluation_mode="pointwise",
            prompt=prompt,
            raw_response=response.content,
            parsed_verdict=parsed_data,
            judge_model=response.model,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            total_tokens=response.total_tokens,
            latency_ms=response.latency_ms,
            cost_usd=cost_usd,
            metadata={"overall_score": overall_score, "passed": passed}
        )

        return verdict

    def evaluate_suite(self, suite_path: Path, suite_name: Optional[str] = None) -> SuiteReport:
        with open(suite_path, "r", encoding="utf-8") as f:
            cases_raw = json.load(f)

        test_cases = [TestCase.model_validate(c) for c in cases_raw]
        verdicts: List[CaseVerdict] = []

        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_cost = 0.0
        total_latency = 0.0

        for case in test_cases:
            v = self.evaluate_case(case)
            verdicts.append(v)
            total_prompt_tokens += v.prompt_tokens
            total_completion_tokens += v.completion_tokens
            total_cost += v.cost_usd
            total_latency += v.latency_ms

        total_cases = len(verdicts)
        passed_cases = sum(1 for v in verdicts if v.passed)
        failed_cases = total_cases - passed_cases
        pass_rate = round(passed_cases / max(total_cases, 1), 4)
        mean_score = round(sum(v.overall_score for v in verdicts) / max(total_cases, 1), 2)
        
        # Calculate standard deviation
        variance = sum((v.overall_score - mean_score) ** 2 for v in verdicts) / max(total_cases, 1)
        score_std_dev = round(variance ** 0.5, 3)

        # Per criterion means
        all_criteria = set()
        for v in verdicts:
            all_criteria.update(v.criterion_scores.keys())
        
        per_criterion_means: Dict[str, float] = {}
        for c_id in all_criteria:
            scores = [v.criterion_scores[c_id] for v in verdicts if c_id in v.criterion_scores]
            if scores:
                per_criterion_means[c_id] = round(sum(scores) / len(scores), 2)

        return SuiteReport(
            suite_name=suite_name or suite_path.stem,
            total_cases=total_cases,
            passed_cases=passed_cases,
            failed_cases=failed_cases,
            pass_rate=pass_rate,
            mean_overall_score=mean_score,
            score_std_dev=score_std_dev,
            per_criterion_means=per_criterion_means,
            total_prompt_tokens=total_prompt_tokens,
            total_completion_tokens=total_completion_tokens,
            total_tokens=total_prompt_tokens + total_completion_tokens,
            total_cost_usd=round(total_cost, 6),
            average_latency_ms=round(total_latency / max(total_cases, 1), 2),
            verdicts=verdicts
        )
