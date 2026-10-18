from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from pydantic import BaseModel

class LLMResponse(BaseModel):
    content: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: float
    model: str
    raw_response: Optional[Dict[str, Any]] = None
    execution_mode: Optional[str] = "direct"
    fallback_notice: Optional[str] = None

class BaseLLMClient(ABC):
    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        response_format: Optional[str] = None,
    ) -> LLMResponse:
        """
        Sends prompt to LLM and returns standardized LLMResponse.
        """
        pass
