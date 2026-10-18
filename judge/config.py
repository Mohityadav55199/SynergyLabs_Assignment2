"""Judge / generator configuration. Judge and generator models are independent (JUDGE_MODEL / GENERATOR_MODEL)."""
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

PASS_THRESHOLD = float(os.getenv("JUDGE_PASS_THRESHOLD", "3.5"))   # weighted overall needed to pass
CRITERION_FLOOR = 2                                                 # any criterion <= this fails the case regardless of the mean


def family_of(model: str) -> str:
    """Coarse model family, used for the self-enhancement guard. Unknown names are their own family."""
    m = model.lower()
    for key, fam in (("gpt-oss", "openai"), ("gpt-", "openai"), ("o1", "openai"), ("qwen", "qwen"), ("llama", "meta-llama"), ("mixtral", "mistral"),
                     ("mistral", "mistral"), ("gemma", "google"), ("gemini", "google"), ("claude", "anthropic"), ("allam", "allam"), ("deepseek", "deepseek")):
        if key in m:
            return fam
    return m


@dataclass
class JudgeConfig:
    judge_model: str = field(default_factory=lambda: os.getenv("JUDGE_MODEL", "openai/gpt-oss-120b"))
    generator_model: str = field(default_factory=lambda: os.getenv("GENERATOR_MODEL", "qwen/qwen3.8-27b"))
    api_key: str = field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))
    base_url: str = field(default_factory=lambda: os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1"))
    judge_temperature: float = field(default_factory=lambda: float(os.getenv("JUDGE_TEMPERATURE", "0")))
    judge_max_tokens: int = field(default_factory=lambda: int(os.getenv("JUDGE_MAX_TOKENS", "1500")))
    log_path: Path = field(default_factory=lambda: Path(os.getenv("JUDGE_LOG", str(ROOT / "judge" / "logs" / "judge_calls.jsonl"))))

    @property
    def same_family(self) -> bool:
        return family_of(self.judge_model) == family_of(self.generator_model)


def check_cross_family(judge_model: str, generator_model: str, allow_same: bool = False) -> dict:
    """Self-enhancement guard: a judge should not come from the generator's model family. Raises unless explicitly allowed."""
    jf, gf = family_of(judge_model), family_of(generator_model)
    info = {"judge_model": judge_model, "judge_family": jf, "generator_model": generator_model, "generator_family": gf, "cross_family": jf != gf}
    if jf == gf and not allow_same:
        raise ValueError(f"self-enhancement risk: judge '{judge_model}' and generator '{generator_model}' are both family '{jf}'. "
                         "Use a different family or pass --allow-same-family (recorded in the report).")
    return info
