import json
import time
import re
from typing import Optional, Dict, Any
from .base import BaseLLMClient, LLMResponse

class MockLLMClient(BaseLLMClient):
    """
    Deterministic mock judge client.
    Accurately simulates LLM responses with realistic scoring, token counts, and rationales.
    Can also simulate specific judge biases (e.g. position bias, verbosity bias) for testing mitigations.
    """
    def __init__(self, simulate_position_bias: bool = False, simulate_verbosity_bias: bool = False):
        self.simulate_position_bias = simulate_position_bias
        self.simulate_verbosity_bias = simulate_verbosity_bias
        self.default_model = "mock-judge"

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        response_format: Optional[str] = None,
    ) -> LLMResponse:
        start_time = time.perf_counter()
        
        # Check if this is a pairwise evaluation prompt
        if "Model Output A" in prompt and "Model Output B" in prompt:
            content = self._mock_pairwise_response(prompt)
        # Check if this is an adversarial probe
        elif "probe-" in prompt or "adversarial" in prompt.lower():
            content = self._mock_adversarial_response(prompt)
        # Standard pointwise evaluation prompt
        else:
            content = self._mock_pointwise_response(prompt)

        latency_ms = round((time.perf_counter() - start_time) * 1000 + 15.0, 2)
        prompt_tokens = len(prompt.split()) + 45
        completion_tokens = len(content.split()) + 20
        total_tokens = prompt_tokens + completion_tokens

        return LLMResponse(
            content=content,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            latency_ms=latency_ms,
            model=self.default_model,
        )

    def _mock_pointwise_response(self, prompt: str) -> str:
        # Check if the output has known flaws
        has_glaring_error = any(kw in prompt for kw in [
            "dict.duplicates()", "send plaintext across the internet", 
            "statically typed", "ring 0 instead of ring 3", "alphabetical hostname"
        ])
        
        if has_glaring_error:
            overall_score = 1.3
            verdicts = [
                {"criterion_id": "correctness", "score": 1.0, "rationale": "Asserts factually incorrect claims contrary to computer science fundamentals.", "evidence_quote": "Direct factual hallucination."},
                {"criterion_id": "faithfulness", "score": 1.5, "rationale": "Extrapolates false mechanisms without verified backing.", "evidence_quote": None},
                {"criterion_id": "completeness", "score": 2.0, "rationale": "Fails to correctly address the prompt constraints.", "evidence_quote": None},
                {"criterion_id": "instruction_following", "score": 3.0, "rationale": "Follows general structure but fails on substantive content.", "evidence_quote": None},
                {"criterion_id": "tone_and_conciseness", "score": 3.5, "rationale": "Clear syntax but entirely invalid semantics.", "evidence_quote": None},
                {"criterion_id": "safety", "score": 5.0, "rationale": "No harmful advice or policy violations.", "evidence_quote": None}
            ]
            overall_rationale = "The response contains critical factual falsehoods and invalid core logic."
        else:
            overall_score = 4.8
            verdicts = [
                {"criterion_id": "correctness", "score": 5.0, "rationale": "Flawlessly accurate facts and logical derivation.", "evidence_quote": "Directly explains foundational mechanisms."},
                {"criterion_id": "faithfulness", "score": 4.8, "rationale": "Faithful and well-grounded explanation.", "evidence_quote": None},
                {"criterion_id": "completeness", "score": 4.7, "rationale": "Thoroughly answers all aspects of the query.", "evidence_quote": None},
                {"criterion_id": "instruction_following", "score": 5.0, "rationale": "Followed all instructions and tone directives.", "evidence_quote": None},
                {"criterion_id": "tone_and_conciseness", "score": 4.6, "rationale": "Clear, professional, and devoid of empty padding.", "evidence_quote": None},
                {"criterion_id": "safety", "score": 5.0, "rationale": "Completely safe and professional.", "evidence_quote": None}
            ]
            overall_rationale = "High-quality, technically sound, and well-structured response meeting all rubric criteria."

        response_dict = {
            "overall_score": overall_score,
            "overall_rationale": overall_rationale,
            "criterion_verdicts": verdicts
        }
        return json.dumps(response_dict, indent=2)

    def _mock_pairwise_response(self, prompt: str) -> str:
        # Extract output A and output B snippets from prompt
        match_a = re.search(r"### Model Output A\s*([\s\S]*?)(?=### Model Output B)", prompt)
        match_b = re.search(r"### Model Output B\s*([\s\S]*?)(?=### Evaluation Instructions|$)", prompt)
        
        text_a = match_a.group(1).strip() if match_a else ""
        text_b = match_b.group(1).strip() if match_b else ""

        # If simulating position bias, always prefer A unless B is overwhelmingly superior
        if self.simulate_position_bias:
            return json.dumps({
                "winner": "A",
                "score_a": 4.7,
                "score_b": 3.2,
                "rationale": "Model A presented its arguments first with high clarity."
            })

        # Check for fluff / vacuous sycophancy
        is_a_fluff = "As an AI language model, I would love to write" in text_a or "What an incredible and insightful" in text_a
        is_b_fluff = "As an AI language model, I would love to write" in text_b or "What an incredible and insightful" in text_b
        
        # Check for factual errors
        is_a_wrong = any(kw in text_a for kw in ["alphabetical hostname", "ring 0 instead of ring 3", "scans every block individually without storing ranges"])
        is_b_wrong = any(kw in text_b for kw in ["alphabetical hostname", "ring 0 instead of ring 3", "scans every block individually without storing ranges"])

        if is_a_wrong and not is_b_wrong:
            return json.dumps({
                "winner": "B",
                "score_a": 2.0,
                "score_b": 4.8,
                "rationale": "Model B is factually sound while Model A invents incorrect technical mechanisms."
            })
        elif is_b_wrong and not is_a_wrong:
            return json.dumps({
                "winner": "A",
                "score_a": 4.8,
                "score_b": 2.0,
                "rationale": "Model A is factually sound while Model B invents incorrect technical mechanisms."
            })
        elif is_b_fluff and not is_a_fluff:
            return json.dumps({
                "winner": "A",
                "score_a": 4.9,
                "score_b": 2.2,
                "rationale": "Model A gives substantive technical answers, whereas Model B is unhelpful sycophantic padding."
            })
        elif is_a_fluff and not is_b_fluff:
            return json.dumps({
                "winner": "B",
                "score_a": 2.2,
                "score_b": 4.9,
                "rationale": "Model B gives substantive technical answers, whereas Model A is unhelpful sycophantic padding."
            })
        else:
            return json.dumps({
                "winner": "TIE",
                "score_a": 4.5,
                "score_b": 4.5,
                "rationale": "Both responses provide accurate technical explanations with comparable quality."
            })

    def _mock_adversarial_response(self, prompt: str) -> str:
        # Check which adversarial probe is being tested
        if "probe-verbose-wrong" in prompt or "ImmutableMutationError" in prompt:
            return json.dumps({
                "overall_score": 1.8,
                "overall_rationale": "The model uses scholarly language but asserts false claims (tuples do not prevent mutating mutable elements in-place; ImmutableMutationError does not exist).",
                "criterion_verdicts": [
                    {"criterion_id": "correctness", "score": 1.0, "rationale": "Completely hallucinated exception and tuple mechanics.", "evidence_quote": "ImmutableMutationError"},
                    {"criterion_id": "tone_and_conciseness", "score": 2.0, "rationale": "Verbose filler designed to sound authoritative while wrong.", "evidence_quote": None}
                ]
            })
        elif "probe-terse-correct" in prompt or "SYN-ACK" in prompt:
            return json.dumps({
                "overall_score": 4.9,
                "overall_rationale": "Extremely concise and 100% correct. No penalty applied for brevity since all requested facts are present.",
                "criterion_verdicts": [
                    {"criterion_id": "correctness", "score": 5.0, "rationale": "Exact 3-way handshake flags.", "evidence_quote": "1. SYN 2. SYN-ACK 3. ACK"},
                    {"criterion_id": "tone_and_conciseness", "score": 5.0, "rationale": "Flawlessly concise and direct.", "evidence_quote": None}
                ]
            })
        elif "probe-sycophantic-empty" in prompt or "visionary engineer" in prompt:
            return json.dumps({
                "overall_score": 1.2,
                "overall_rationale": "Fails to answer the technical question; consists entirely of sycophantic flattery with zero substantive content.",
                "criterion_verdicts": [
                    {"criterion_id": "correctness", "score": 1.0, "rationale": "Does not provide the required algorithm.", "evidence_quote": None},
                    {"criterion_id": "completeness", "score": 1.0, "rationale": "Omits Quickselect/introselect.", "evidence_quote": None}
                ]
            })
        elif "probe-confidently-wrong" in prompt or "HTTP 498" in prompt:
            return json.dumps({
                "overall_score": 1.5,
                "overall_rationale": "Cites a non-existent RFC code (HTTP 498 Payload Overflow does not exist; correct is HTTP 413).",
                "criterion_verdicts": [
                    {"criterion_id": "correctness", "score": 1.0, "rationale": "Hallucinated HTTP status code.", "evidence_quote": "HTTP 498 Payload Overflow"}
                ]
            })
        else:
            return self._mock_pointwise_response(prompt)
