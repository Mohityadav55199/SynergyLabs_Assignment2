import pytest
from src.llm_clients.base import BaseLLMClient, LLMResponse
from src.llm_clients.resilient_client import ResilientLLMClient
from src.llm_clients.mock_client import MockLLMClient
from src.judge.pointwise import PointwiseJudge
from src.schema import TestCase

class FailingClient(BaseLLMClient):
    def __init__(self, error_message: str = "404 Not Found: Model deprecated"):
        self.error_message = error_message
        self.default_model = "deprecated-model-70b"

    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        raise RuntimeError(self.error_message)

class SuccessfulClient(BaseLLMClient):
    def __init__(self, model_name: str = "primary-model"):
        self.default_model = model_name

    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        return LLMResponse(
            content='{"overall_score": 4.5, "passed": true, "criterion_verdicts": [], "overall_rationale": "Great"}',
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
            model=self.default_model,
            latency_ms=120.0
        )

def test_resilient_client_success_primary():
    primary = SuccessfulClient()
    client = ResilientLLMClient(primary_client=primary, enable_fallback=True)
    
    response = client.generate("test prompt")
    assert response.execution_mode != "deterministic_fallback"
    assert response.fallback_notice is None
    assert "overall_score" in response.content

def test_resilient_client_fallback_to_mock_when_primary_fails():
    primary = FailingClient("Model llama-3.3-70b-versatile not found")
    client = ResilientLLMClient(
        primary_client=primary,
        api_key=None,  # No API key for live fallbacks, jumps straight to deterministic benchmark
        enable_fallback=True
    )
    
    response = client.generate("Evaluate this output")
    assert response.execution_mode == "deterministic_fallback"
    assert response.fallback_notice is not None
    assert "calibrated benchmark judge" in response.fallback_notice
    assert "overall_score" in response.content

def test_resilient_client_raises_when_fallback_disabled():
    primary = FailingClient("API key expired")
    client = ResilientLLMClient(
        primary_client=primary,
        enable_fallback=False
    )
    
    with pytest.raises(RuntimeError) as exc_info:
        client.generate("Evaluate")
    assert "Primary judge failed and fallback is disabled" in str(exc_info.value)

def test_pointwise_judge_with_resilient_fallback():
    primary = FailingClient("404 Model llama-3.3-70b-versatile does not exist")
    resilient_client = ResilientLLMClient(primary_client=primary, enable_fallback=True)
    
    judge = PointwiseJudge(llm_client=resilient_client)
    test_case = TestCase(
        id="test-resilience-001",
        domain="General",
        input="Explain backpropagation",
        model_output="Backpropagation computes gradients using the chain rule.",
        expected_output="Backpropagation uses chain rule for gradient descent."
    )
    
    verdict = judge.evaluate_case(test_case)
    assert verdict.test_case_id == "test-resilience-001"
    assert verdict.execution_mode == "deterministic_fallback"
    assert verdict.fallback_notice is not None
    assert verdict.overall_score >= 1.0
