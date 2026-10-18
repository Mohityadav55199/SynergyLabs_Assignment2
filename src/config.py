import os
from pathlib import Path
from typing import Optional, Dict
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load .env if present
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

class ModelPricing(BaseModel):
    prompt_per_1m: float
    completion_per_1m: float

# Pricing in USD per 1M tokens
MODEL_PRICING: Dict[str, ModelPricing] = {
    # Groq Models
    "llama-3.3-70b-versatile": ModelPricing(prompt_per_1m=0.59, completion_per_1m=0.79),
    "llama-3.1-70b-versatile": ModelPricing(prompt_per_1m=0.59, completion_per_1m=0.79),
    "llama-3.1-8b-instant": ModelPricing(prompt_per_1m=0.05, completion_per_1m=0.08),
    "mixtral-8x7b-32768": ModelPricing(prompt_per_1m=0.24, completion_per_1m=0.24),
    # OpenAI Models (Fallback / Comparison)
    "gpt-4o": ModelPricing(prompt_per_1m=2.50, completion_per_1m=10.00),
    "gpt-4o-mini": ModelPricing(prompt_per_1m=0.15, completion_per_1m=0.60),
    # Mock / Local
    "mock-judge": ModelPricing(prompt_per_1m=0.0, completion_per_1m=0.0),
}

class PipelineConfig(BaseModel):
    # Provider: "groq", "openai", "mock"
    llm_provider: str = Field(default_factory=lambda: os.getenv("LLM_PROVIDER", "groq").lower())
    
    # API Keys
    groq_api_key: Optional[str] = Field(default_factory=lambda: os.getenv("GROQ_API_KEY"))
    openai_api_key: Optional[str] = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    
    # Models
    judge_model: str = Field(default_factory=lambda: os.getenv("JUDGE_MODEL", "llama-3.3-70b-versatile"))
    generator_model: str = Field(default_factory=lambda: os.getenv("GENERATOR_MODEL", "llama-3.1-8b-instant"))
    
    # Inference settings
    judge_temperature: float = Field(default_factory=lambda: float(os.getenv("JUDGE_TEMPERATURE", "0.0")))
    judge_max_tokens: int = Field(default_factory=lambda: int(os.getenv("JUDGE_MAX_TOKENS", "2048")))
    
    # Mitigations
    enable_position_swap: bool = Field(default_factory=lambda: os.getenv("ENABLE_POSITION_SWAP_MITIGATION", "true").lower() == "true")
    enable_verbosity_penalty: bool = Field(default_factory=lambda: os.getenv("ENABLE_VERBOSITY_PENALTY", "true").lower() == "true")
    enable_evidence_grounding: bool = Field(default_factory=lambda: os.getenv("ENABLE_EVIDENCE_GROUNDING", "true").lower() == "true")
    enable_anchor_calibration: bool = Field(default_factory=lambda: os.getenv("ENABLE_ANCHOR_CALIBRATION", "true").lower() == "true")
    
    # Resilient Fallback Hierarchy
    enable_fallback: bool = Field(default_factory=lambda: os.getenv("ENABLE_FALLBACK", "true").lower() == "true")
    fallback_models: list = Field(default_factory=lambda: [
        "llama-3.1-70b-versatile",
        "llama-3.1-8b-instant",
        "mixtral-8x7b-32768",
    ])

    # Quality thresholds
    pass_threshold: float = Field(default_factory=lambda: float(os.getenv("PASS_THRESHOLD", "3.5")))
    confidence_threshold: float = Field(default_factory=lambda: float(os.getenv("CONFIDENCE_THRESHOLD", "0.7")))
    
    # Paths
    data_dir: Path = BASE_DIR / "data"
    artifacts_dir: Path = BASE_DIR / "artifacts"
    audit_logs_dir: Path = BASE_DIR / "artifacts" / "audit_logs"

    def calculate_cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        pricing = MODEL_PRICING.get(model, ModelPricing(prompt_per_1m=0.50, completion_per_1m=0.80))
        prompt_cost = (prompt_tokens / 1_000_000) * pricing.prompt_per_1m
        completion_cost = (completion_tokens / 1_000_000) * pricing.completion_per_1m
        return round(prompt_cost + completion_cost, 6)

config = PipelineConfig()
