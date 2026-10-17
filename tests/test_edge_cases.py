import pytest
from pathlib import Path
from src.utils import load_suite_data
from src.parser import RobustJsonParser
from src.mitigations.self_enhancement_bias import SelfEnhancementMitigator
from src.evaluation.metrics import StatisticalMetrics
from src.judge.pointwise import PointwiseJudge
from src.llm_clients.mock_client import MockLLMClient
from src.schema import TestCase

def test_load_yaml_and_json_suites():
    yaml_path = Path("data/test_suites/general_qa.yaml")
    json_path = Path("data/test_suites/general_qa.json")

    yaml_data = load_suite_data(yaml_path)
    json_data = load_suite_data(json_path)

    assert isinstance(yaml_data, list)
    assert len(yaml_data) >= 3
    assert yaml_data[0]["id"].startswith("yaml-qa")

    assert isinstance(json_data, list)
    assert len(json_data) >= 5

def test_parser_truncated_json():
    # LLM output truncated mid-string
    truncated = '{"overall_score": 4.2, "overall_rationale": "Strong answer with good points'
    result = RobustJsonParser.parse_to_dict(truncated)
    assert result["overall_score"] == 4.2

def test_parser_single_quotes():
    raw_single = "{'overall_score': 3.5, 'winner': 'A', 'overall_rationale': 'Adequate'}"
    result = RobustJsonParser.parse_to_dict(raw_single)
    assert result["overall_score"] == 3.5
    assert result["winner"] == "A"

def test_self_enhancement_family_detection():
    assert SelfEnhancementMitigator.detect_model_family("llama-3.3-70b-versatile") == "llama"
    assert SelfEnhancementMitigator.detect_model_family("gpt-4o") == "gpt"
    assert SelfEnhancementMitigator.detect_model_family("claude-3-5-sonnet") == "claude"
    assert SelfEnhancementMitigator.detect_model_family("gemini-1.5-pro") == "gemini"
    assert SelfEnhancementMitigator.detect_model_family("mixtral-8x7b-32768") == "mistral"

    # Same family check
    res_same = SelfEnhancementMitigator.check_cross_family_compliance("llama-3.3-70b", "llama-3.1-8b")
    assert res_same["is_cross_family"] is False
    assert res_same["self_enhancement_risk"] == "HIGH"

    # Cross family check
    res_cross = SelfEnhancementMitigator.check_cross_family_compliance("llama-3.3-70b", "gpt-4o")
    assert res_cross["is_cross_family"] is True
    assert res_cross["self_enhancement_risk"] == "LOW"

def test_reference_mode_handling():
    client = MockLLMClient()
    judge = PointwiseJudge(llm_client=client)

    # Reference based
    tc_ref = TestCase(
        id="test-ref-01",
        input="What is the speed of light?",
        model_output="299,792,458 m/s",
        expected_output="299,792,458 m/s",
        reference_mode="reference_based"
    )
    prompt_ref = judge.format_judging_prompt(tc_ref)
    assert "REFERENCE-BASED" in prompt_ref

    # Reference free
    tc_free = TestCase(
        id="test-free-01",
        input="Write a creative poem about sorting algorithms.",
        model_output="Quickly split and conquer all...",
        reference_mode="reference_free"
    )
    prompt_free = judge.format_judging_prompt(tc_free)
    assert "REFERENCE-FREE" in prompt_free

def test_metrics_edge_cases():
    # Identical predictions / constant
    preds = [True, True, True]
    golds = [True, True, True]
    assert StatisticalMetrics.agreement_rate(preds, golds) == 1.0
    assert StatisticalMetrics.cohens_kappa(preds, golds) == 1.0

    # Constant lists in pearson correlation
    assert StatisticalMetrics.pearson_correlation([5.0, 5.0, 5.0], [5.0, 5.0, 5.0]) == 0.0

    # Single element lists
    assert StatisticalMetrics.pearson_correlation([1.0], [1.0]) == 0.0
    assert StatisticalMetrics.spearman_rank_correlation([1.0], [1.0]) == 0.0

    # Empty lists
    assert StatisticalMetrics.agreement_rate([], []) == 0.0
    assert StatisticalMetrics.mean_absolute_error([], []) == 0.0
