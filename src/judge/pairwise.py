import json
import logging
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from ..schema import PairwiseTestCase, PairwiseVerdict, SinglePairEvaluation, ABComparisonReport
from ..llm_clients.base import BaseLLMClient
from ..parser import RobustJsonParser
from ..config import config
from ..logging.auditor import auditor
from ..mitigations.position_bias import PositionBiasMitigator
from ..mitigations.verbosity_bias import VerbosityMitigator
from ..mitigations.self_enhancement_bias import SelfEnhancementMitigator
from ..utils import load_suite_data

logger = logging.getLogger(__name__)

class PairwiseJudge:
    """
    Evaluates candidate pairs (Model A vs Model B, or Prompt v1 vs Prompt v2).
    Implements dual-order evaluation (A vs B, B vs A) to measure and eliminate position bias,
    controls for verbosity bias, and protects against self-enhancement bias.
    """

    def __init__(
        self,
        llm_client: BaseLLMClient,
        rubric_path: Optional[Path] = None,
        enable_position_swap: bool = True
    ):
        self.llm_client = llm_client
        self.rubric_path = rubric_path or (config.data_dir / "rubrics" / "pairwise_comparison.yaml")
        self.enable_position_swap = enable_position_swap

    def format_pairwise_prompt(self, test_case: PairwiseTestCase, cand_a: str, cand_b: str) -> str:
        prompt = f"""You are an expert impartial evaluation judge. Your job is to compare two candidate model outputs (Candidate A and Candidate B) in response to the user query.

### User Query
{test_case.input}

### System Prompt
{test_case.system_prompt or "None"}

### Model Output A
\"\"\"{cand_a}\"\"\"

### Model Output B
\"\"\"{cand_b}\"\"\"

### Evaluation Instructions & Bias Safeguards
- Compare the two candidates strictly on technical accuracy, instruction adherence, grounding, and usefulness.
- ANTI-VERBOSITY RULE: Do NOT favor a candidate simply because it is longer or uses excessive conversational pleasantries. Penalize fluff and superficial padding.
- CROSS-FAMILY SAFEGUARD: Do NOT favor a candidate based on stylistic similarity to your own model family or training data.
- If Candidate A is superior, select "A".
- If Candidate B is superior, select "B".
- If both are of indistinguishable quality or equally flawed, select "TIE".
- Rate each candidate on a 1.0 to 5.0 scale.

### Output Specification
You MUST respond with a single valid JSON object strictly matching this schema:
{{
  "winner": "<A, B, or TIE>",
  "score_a": <float between 1.0 and 5.0>,
  "score_b": <float between 1.0 and 5.0>,
  "rationale": "<thorough comparative rationale detailing why the winner was chosen>"
}}
Do NOT output any conversational text outside the JSON block.
"""
        return prompt

    def evaluate_single_order(
        self, 
        test_case: PairwiseTestCase, 
        cand_a: str, 
        cand_b: str, 
        order_tag: str
    ) -> SinglePairEvaluation:
        prompt = self.format_pairwise_prompt(test_case, cand_a, cand_b)
        system_prompt = "You are an objective, unbiased AI pairwise comparison judge."

        response = self.llm_client.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=config.judge_temperature,
            max_tokens=config.judge_max_tokens,
            response_format="json"
        )

        parsed_data = RobustJsonParser.parse_to_dict(response.content)

        winner = parsed_data.get("winner", "TIE").upper()
        if winner not in ["A", "B", "TIE"]:
            winner = "TIE"

        score_a = float(parsed_data.get("score_a", 3.0))
        score_b = float(parsed_data.get("score_b", 3.0))
        rationale = parsed_data.get("rationale", "No rationale provided.")

        cost_usd = config.calculate_cost(response.model, response.prompt_tokens, response.completion_tokens)

        # Log audit trail
        auditor.log_judge_call(
            test_case_id=f"{test_case.id}_{order_tag}",
            evaluation_mode="pairwise",
            prompt=prompt,
            raw_response=response.content,
            parsed_verdict=parsed_data,
            judge_model=response.model,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            total_tokens=response.total_tokens,
            latency_ms=response.latency_ms,
            cost_usd=cost_usd,
            metadata={"winner": winner, "order_tag": order_tag}
        )

        return SinglePairEvaluation(
            winner=winner,
            score_a=score_a,
            score_b=score_b,
            rationale=rationale,
            latency_ms=response.latency_ms,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            total_tokens=response.total_tokens,
            cost_usd=cost_usd
        )

    def evaluate_pair(self, test_case: PairwiseTestCase) -> PairwiseVerdict:
        # Pass 1: Forward order (A, B)
        forward_eval = self.evaluate_single_order(test_case, test_case.output_a, test_case.output_b, "forward")

        if not self.enable_position_swap:
            return PairwiseVerdict(
                test_case_id=test_case.id,
                final_winner=forward_eval.winner,
                forward_eval=forward_eval,
                reverse_eval=None,
                position_agreement=True,
                flip_detected=False,
                average_score_a=forward_eval.score_a,
                average_score_b=forward_eval.score_b,
                synthesis_rationale=forward_eval.rationale,
                total_tokens=forward_eval.total_tokens,
                total_cost_usd=forward_eval.cost_usd
            )

        # Pass 2: Reverse order (B, A)
        reverse_eval = self.evaluate_single_order(test_case, test_case.output_b, test_case.output_a, "reverse")

        # Resolve position bias via PositionBiasMitigator
        final_winner, agreement, flip_detected, avg_a, avg_b, syn_rationale = (
            PositionBiasMitigator.resolve_verdict(forward_eval, reverse_eval)
        )

        total_tokens = forward_eval.total_tokens + reverse_eval.total_tokens
        total_cost = round(forward_eval.cost_usd + reverse_eval.cost_usd, 6)

        return PairwiseVerdict(
            test_case_id=test_case.id,
            final_winner=final_winner,
            forward_eval=forward_eval,
            reverse_eval=reverse_eval,
            position_agreement=agreement,
            flip_detected=flip_detected,
            average_score_a=avg_a,
            average_score_b=avg_b,
            synthesis_rationale=syn_rationale,
            total_tokens=total_tokens,
            total_cost_usd=total_cost
        )

    def evaluate_ab_suite(
        self,
        suite_path: Path,
        config_a_name: str = "Config A (Prompt v1)",
        config_b_name: str = "Config B (Prompt v2)"
    ) -> ABComparisonReport:
        raw_cases = load_suite_data(suite_path)
        test_cases = [PairwiseTestCase.model_validate(c) for c in raw_cases]
        verdicts: List[PairwiseVerdict] = []

        wins_a = 0
        wins_b = 0
        ties = 0
        total_cost = 0.0
        flips = 0

        for case in test_cases:
            v = self.evaluate_pair(case)
            verdicts.append(v)
            total_cost += v.total_cost_usd

            if v.final_winner == "A":
                wins_a += 1
            elif v.final_winner == "B":
                wins_b += 1
            else:
                ties += 1

            if v.flip_detected:
                flips += 1

        total_pairs = len(verdicts)
        win_rate_a = round(wins_a / max(total_pairs, 1), 4)
        win_rate_b = round(wins_b / max(total_pairs, 1), 4)
        tie_rate = round(ties / max(total_pairs, 1), 4)
        position_flip_rate = round(flips / max(total_pairs, 1), 4)
        swap_consistency_rate = round(1.0 - position_flip_rate, 4)

        if wins_a > wins_b:
            declared_winner = "A"
            margin = round(win_rate_a - win_rate_b, 4)
        elif wins_b > wins_a:
            declared_winner = "B"
            margin = round(win_rate_b - win_rate_a, 4)
        else:
            declared_winner = "TIE"
            margin = 0.0

        return ABComparisonReport(
            config_a_name=config_a_name,
            config_b_name=config_b_name,
            total_pairs=total_pairs,
            wins_a=wins_a,
            wins_b=wins_b,
            ties=ties,
            win_rate_a=win_rate_a,
            win_rate_b=win_rate_b,
            tie_rate=tie_rate,
            declared_winner=declared_winner,
            margin=margin,
            swap_consistency_rate=swap_consistency_rate,
            position_flip_rate=position_flip_rate,
            total_cost_usd=round(total_cost, 6),
            verdicts=verdicts
        )
