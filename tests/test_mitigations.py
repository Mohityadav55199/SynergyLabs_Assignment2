import pytest
from src.mitigations.position_bias import PositionBiasMitigator
from src.mitigations.verbosity_bias import VerbosityMitigator
from src.mitigations.sycophancy_bias import SycophancyMitigator
from src.schema import SinglePairEvaluation, CriterionVerdict

def test_position_bias_flip_detection():
    # Forward: chose A (score 4.5 vs 3.0)
    eval_fwd = SinglePairEvaluation(
        winner="A", score_a=4.5, score_b=3.0, rationale="A was better",
        latency_ms=10.0, prompt_tokens=10, completion_tokens=10, total_tokens=20, cost_usd=0.0
    )
    # Reverse: chose A in reverse order (which corresponds to candidate B!)
    eval_rev = SinglePairEvaluation(
        winner="A", score_a=4.5, score_b=3.0, rationale="A was better",
        latency_ms=10.0, prompt_tokens=10, completion_tokens=10, total_tokens=20, cost_usd=0.0
    )
    
    winner, agreement, flip_detected, avg_a, avg_b, rat = PositionBiasMitigator.resolve_verdict(
        eval_fwd, eval_rev, strategy="strict_agreement"
    )
    assert flip_detected is True
    assert agreement is False
    assert winner == "TIE" # Strict agreement neutralizes flip

def test_verbosity_mitigator_penalty():
    fluffy_text = "As an AI language model, I would love to delve into the fascinating and truly remarkable aspects of this problem."
    penalty = VerbosityMitigator.calculate_length_penalty(fluffy_text)
    assert penalty > 0.0

def test_sycophancy_grounding_penalty():
    verdicts = [
        CriterionVerdict(criterion_id="correctness", score=4.5, rationale="Great", evidence_quote="completely fabricated quote that does not exist in text")
    ]
    candidate = "The time complexity is O(n log n)."
    penalty = SycophancyMitigator.verify_evidence_grounding(verdicts, candidate)
    assert penalty > 0.0
