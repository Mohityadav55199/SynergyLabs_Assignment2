import pytest
from pathlib import Path
from src.config import config
from src.llm_clients.mock_client import MockLLMClient
from src.judge.pointwise import PointwiseJudge
from src.judge.pairwise import PairwiseJudge
from src.evaluation.validator import JudgeValidator
from src.evaluation.adversarial_runner import AdversarialProbeRunner

def test_pointwise_suite_run():
    client = MockLLMClient()
    judge = PointwiseJudge(llm_client=client)
    suite_path = config.data_dir / "test_suites" / "general_qa.json"
    report = judge.evaluate_suite(suite_path)

    assert report.total_cases > 0
    assert report.pass_rate >= 0.0
    assert report.mean_overall_score >= 1.0

def test_pairwise_ab_suite_run():
    client = MockLLMClient()
    judge = PairwiseJudge(llm_client=client, enable_position_swap=True)
    suite_path = config.data_dir / "test_suites" / "config_a_vs_b.json"
    report = judge.evaluate_ab_suite(suite_path)

    assert report.total_pairs > 0
    assert report.declared_winner in ["A", "B", "TIE"]
    assert report.position_flip_rate == 0.0 # With mock deterministic client

def test_judge_validation_run():
    client = MockLLMClient()
    judge = PointwiseJudge(llm_client=client)
    validator = JudgeValidator(judge)
    gold_path = config.data_dir / "test_suites" / "gold_labeled_benchmark.json"
    result = validator.validate_against_gold(gold_path)

    assert result["metrics"]["agreement_rate"] >= 0.80
    assert result["metrics"]["cohens_kappa"] >= 0.70

def test_adversarial_probes_run():
    client = MockLLMClient()
    judge = PointwiseJudge(llm_client=client)
    adv_runner = AdversarialProbeRunner(judge)
    probes_path = config.data_dir / "adversarial_probes" / "adversarial_suite.json"
    report = adv_runner.run_probes(probes_path)

    assert report.total_probes > 0
    assert report.robustness_rate >= 0.75
