import time
import logging
from typing import Optional, List, Dict, Any
from .base import BaseLLMClient, LLMResponse
from .groq_client import GroqClient
from .mock_client import MockLLMClient

logger = logging.getLogger(__name__)

class ResilientLLMClient(BaseLLMClient):
    """
    Multi-tier resilient fallback client.
    Prevents single-point-of-failure outages when an upstream provider or model
    returns 404 (model deprecated/not found), 429 (rate limit), or network error.

    Fallback Hierarchy:
    1. Primary Model (e.g., Groq llama-3.3-70b-versatile)
    2. Provider-level Fallback Models (e.g., Groq llama-3.1-70b-versatile, llama-3.1-8b-instant)
    3. Calibrated Deterministic Benchmark Judge (MockLLMClient)
    """

    def __init__(
        self,
        primary_client: BaseLLMClient,
        api_key: Optional[str] = None,
        fallback_models: Optional[List[str]] = None,
        enable_fallback: bool = True,
        simulate_position_bias: bool = False,
        simulate_verbosity_bias: bool = False,
    ):
        self.primary_client = primary_client
        self.api_key = api_key
        self.fallback_models = fallback_models or [
            "llama-3.1-70b-versatile",
            "llama-3.1-8b-instant",
            "mixtral-8x7b-32768",
        ]
        self.enable_fallback = enable_fallback
        self.mock_fallback = MockLLMClient(
            simulate_position_bias=simulate_position_bias,
            simulate_verbosity_bias=simulate_verbosity_bias,
        )

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        response_format: Optional[str] = None,
    ) -> LLMResponse:
        primary_model = getattr(self.primary_client, "default_model", "primary-judge")
        errors: List[str] = []

        # Tier 1: Attempt Primary Client
        try:
            response = self.primary_client.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format,
            )
            response.execution_mode = "live_groq" if isinstance(self.primary_client, GroqClient) else "calibrated_mock"
            return response
        except Exception as exc:
            err_msg = str(exc)
            logger.warning(f"Primary model '{primary_model}' failed: {err_msg}. Initiating resilient fallback protocol.")
            errors.append(f"{primary_model}: {err_msg}")

            if not self.enable_fallback:
                raise RuntimeError(f"Primary judge failed and fallback is disabled: {exc}")

        # Tier 2: Attempt Alternative Live Models (if provider is Groq and API key is present)
        if self.api_key and isinstance(self.primary_client, GroqClient):
            for alt_model in self.fallback_models:
                if alt_model == primary_model:
                    continue
                try:
                    logger.info(f"Attempting live fallback model: '{alt_model}'")
                    alt_client = GroqClient(api_key=self.api_key, default_model=alt_model)
                    response = alt_client.generate(
                        prompt=prompt,
                        system_prompt=system_prompt,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        response_format=response_format,
                    )
                    response.execution_mode = "live_model_fallback"
                    response.fallback_notice = (
                        f"Primary model '{primary_model}' encountered an issue ({errors[0][:80]}...). "
                        f"Gracefully evaluated using live fallback model '{alt_model}'."
                    )
                    return response
                except Exception as alt_exc:
                    logger.warning(f"Fallback model '{alt_model}' also failed: {alt_exc}")
                    errors.append(f"{alt_model}: {str(alt_exc)}")

        # Tier 3: Deterministic Calibrated Judge
        logger.info("Live endpoints unavailable. Activating deterministic calibrated benchmark judge.")
        try:
            response = self.mock_fallback.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format,
            )
            response.execution_mode = "deterministic_fallback"
            response.fallback_notice = (
                f"Live request failed ({errors[0][:100]}...). "
                f"Evaluated using high-fidelity calibrated benchmark judge."
            )
            return response
        except Exception as mock_exc:
            all_errors = " | ".join(errors)
            raise RuntimeError(f"All evaluation tiers (live and fallback) failed: {all_errors} | Final error: {mock_exc}")
