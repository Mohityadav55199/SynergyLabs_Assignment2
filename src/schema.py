from typing import List, Dict, Optional, Any, Literal
from pydantic import BaseModel, Field, field_validator

class TestCase(BaseModel):
    id: str
    domain: Optional[str] = "General"
    system_prompt: Optional[str] = None
    input: str
    model_output: str
    expected_output: Optional[str] = None
    criteria: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)

class RubricCriterion(BaseModel):
    id: str
    name: str
    description: str
    weight: float = Field(default=1.0, ge=0.0)
    anchors: Dict[int, str] = Field(default_factory=dict)

class Rubric(BaseModel):
    version: str = "1.0"
    mode: Literal["pointwise", "pairwise"] = "pointwise"
    name: str
    description: str
    criteria: List[RubricCriterion] = Field(default_factory=list)

    def get_criterion(self, criterion_id: str) -> Optional[RubricCriterion]:
        for c in self.criteria:
            if c.id == criterion_id:
                return c
        return None

class CriterionVerdict(BaseModel):
    criterion_id: str
    score: float = Field(..., ge=1.0, le=5.0)
    rationale: str
    evidence_quote: Optional[str] = Field(
        default=None, 
        description="Direct quote or substring from model output verifying this evaluation."
    )

class CaseVerdict(BaseModel):
    test_case_id: str
    overall_score: float = Field(..., ge=1.0, le=5.0)
    passed: bool
    criterion_scores: Dict[str, float]
    criterion_verdicts: List[CriterionVerdict]
    overall_rationale: str
    judge_model: str
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float
    raw_response: Optional[str] = None
    raw_prompt: Optional[str] = None

class SuiteReport(BaseModel):
    suite_name: str
    total_cases: int
    passed_cases: int
    failed_cases: int
    pass_rate: float
    mean_overall_score: float
    score_std_dev: float
    per_criterion_means: Dict[str, float]
    total_prompt_tokens: int
    total_completion_tokens: int
    total_tokens: int
    total_cost_usd: float
    average_latency_ms: float
    verdicts: List[CaseVerdict]

# Pairwise & A/B Comparison Models
class PairwiseTestCase(BaseModel):
    id: str
    prompt_name: Optional[str] = None
    input: str
    system_prompt: Optional[str] = None
    output_a: str
    output_b: str
    expected_winner: Optional[Literal["A", "B", "TIE"]] = None
    rationale: Optional[str] = None

class SinglePairEvaluation(BaseModel):
    winner: Literal["A", "B", "TIE"]
    score_a: float = Field(ge=1.0, le=5.0)
    score_b: float = Field(ge=1.0, le=5.0)
    rationale: str
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float

class PairwiseVerdict(BaseModel):
    test_case_id: str
    final_winner: Literal["A", "B", "TIE"]
    forward_eval: SinglePairEvaluation
    reverse_eval: Optional[SinglePairEvaluation] = None
    position_agreement: bool = True
    flip_detected: bool = False
    average_score_a: float
    average_score_b: float
    synthesis_rationale: str
    total_tokens: int
    total_cost_usd: float

class ABComparisonReport(BaseModel):
    config_a_name: str
    config_b_name: str
    total_pairs: int
    wins_a: int
    wins_b: int
    ties: int
    win_rate_a: float
    win_rate_b: float
    tie_rate: float
    declared_winner: Literal["A", "B", "TIE"]
    margin: float
    swap_consistency_rate: float
    position_flip_rate: float
    total_cost_usd: float
    verdicts: List[PairwiseVerdict]

# Judge Validation Models
class GoldTestCase(BaseModel):
    id: str
    input: str
    model_output: str
    gold_score: float = Field(ge=1.0, le=5.0)
    gold_pass: bool
    domain: Optional[str] = "General"

class ValidationMetrics(BaseModel):
    agreement_rate: float
    cohens_kappa: float
    pearson_correlation: float
    spearman_correlation: float
    mean_absolute_error: float
    root_mean_squared_error: float
    test_retest_consistency: float

class AdversarialProbeCase(BaseModel):
    id: str
    probe_type: Literal["verbose_but_wrong", "terse_but_correct", "sycophantic_empty", "confidently_wrong"]
    description: str
    input: str
    model_output: str
    expected_verdict: Literal["PASS", "FAIL"]
    expected_score_range: List[float]
    flaw: str

class AdversarialProbeResult(BaseModel):
    probe_id: str
    probe_type: str
    judge_score: float
    judge_passed: bool
    expected_verdict: str
    fooled_judge: bool
    rationale: str

class AdversarialReport(BaseModel):
    total_probes: int
    total_fooled: int
    robustness_rate: float
    by_type: Dict[str, Dict[str, Any]]
    results: List[AdversarialProbeResult]
