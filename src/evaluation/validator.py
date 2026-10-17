import json
import logging
from pathlib import Path
from typing import Dict, Any, List
from ..schema import GoldTestCase, ValidationMetrics, TestCase
from ..judge.pointwise import PointwiseJudge
from .metrics import StatisticalMetrics
from ..config import config

logger = logging.getLogger(__name__)

class JudgeValidator:
    """
    Validates judge quality against human/gold ground truth labels.
    Calculates Agreement Rate, Cohen's Kappa, Pearson & Spearman correlations, and test-retest consistency.
    """

    def __init__(self, judge: PointwiseJudge):
        self.judge = judge

    def validate_against_gold(self, benchmark_path: Path) -> Dict[str, Any]:
        with open(benchmark_path, "r", encoding="utf-8") as f:
            raw_cases = json.load(f)

        gold_cases = [GoldTestCase.model_validate(c) for c in raw_cases]

        judge_scores_run1: List[float] = []
        judge_passes_run1: List[bool] = []
        gold_scores: List[float] = []
        gold_passes: List[bool] = []

        case_results: List[Dict[str, Any]] = []

        # Run 1
        for gc in gold_cases:
            tc = TestCase(
                id=gc.id,
                domain=gc.domain,
                input=gc.input,
                model_output=gc.model_output
            )
            v = self.judge.evaluate_case(tc)
            judge_scores_run1.append(v.overall_score)
            judge_passes_run1.append(v.passed)
            gold_scores.append(gc.gold_score)
            gold_passes.append(gc.gold_pass)

            case_results.append({
                "id": gc.id,
                "domain": gc.domain,
                "gold_score": gc.gold_score,
                "judge_score": v.overall_score,
                "gold_pass": gc.gold_pass,
                "judge_pass": v.passed,
                "matched": (gc.gold_pass == v.passed),
                "rationale": v.overall_rationale
            })

        # Run 2 for Test-Retest Consistency
        judge_passes_run2: List[bool] = []
        for gc in gold_cases:
            tc = TestCase(
                id=gc.id,
                domain=gc.domain,
                input=gc.input,
                model_output=gc.model_output
            )
            v2 = self.judge.evaluate_case(tc)
            judge_passes_run2.append(v2.passed)

        metrics = ValidationMetrics(
            agreement_rate=StatisticalMetrics.agreement_rate(judge_passes_run1, gold_passes),
            cohens_kappa=StatisticalMetrics.cohens_kappa(judge_passes_run1, gold_passes),
            pearson_correlation=StatisticalMetrics.pearson_correlation(judge_scores_run1, gold_scores),
            spearman_correlation=StatisticalMetrics.spearman_rank_correlation(judge_scores_run1, gold_scores),
            mean_absolute_error=StatisticalMetrics.mean_absolute_error(judge_scores_run1, gold_scores),
            root_mean_squared_error=StatisticalMetrics.root_mean_squared_error(judge_scores_run1, gold_scores),
            test_retest_consistency=StatisticalMetrics.test_retest_consistency(judge_passes_run1, judge_passes_run2)
        )

        return {
            "metrics": metrics.model_dump(),
            "sample_size": len(gold_cases),
            "case_results": case_results
        }
