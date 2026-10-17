import time
import logging
import httpx
from typing import Optional, Dict, Any
from .base import BaseLLMClient, LLMResponse

logger = logging.getLogger(__name__)

class GroqClient(BaseLLMClient):
    BASE_URL = "https://api.groq.com/openai/v1/chat/completions"

    def __init__(self, api_key: str, default_model: str = "llama-3.3-70b-versatile"):
        self.api_key = api_key
        self.default_model = default_model
        self.client = httpx.Client(timeout=60.0)

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        response_format: Optional[str] = None,
    ) -> LLMResponse:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload: Dict[str, Any] = {
            "model": self.default_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        
        if response_format == "json":
            payload["response_format"] = {"type": "json_object"}

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        max_retries = 3
        backoff = 1.5

        for attempt in range(max_retries):
            start_time = time.perf_counter()
            try:
                response = self.client.post(self.BASE_URL, json=payload, headers=headers)
                latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

                if response.status_code == 200:
                    data = response.json()
                    content = data["choices"][0]["message"]["content"]
                    usage = data.get("usage", {})
                    prompt_tokens = usage.get("prompt_tokens", 0)
                    completion_tokens = usage.get("completion_tokens", 0)
                    total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens)

                    return LLMResponse(
                        content=content,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=total_tokens,
                        latency_ms=latency_ms,
                        model=self.default_model,
                        raw_response=data,
                    )
                elif response.status_code in [429, 500, 502, 503]:
                    logger.warning(f"Groq API returned {response.status_code}. Retrying in {backoff}s...")
                    time.sleep(backoff)
                    backoff *= 2
                else:
                    raise RuntimeError(f"Groq API error ({response.status_code}): {response.text}")

            except httpx.RequestError as exc:
                if attempt == max_retries - 1:
                    raise RuntimeError(f"Network error communicating with Groq: {exc}")
                time.sleep(backoff)
                backoff *= 2

        raise RuntimeError(f"Failed to obtain response from Groq after {max_retries} retries.")
