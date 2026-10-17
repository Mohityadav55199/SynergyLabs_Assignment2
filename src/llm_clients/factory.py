import os
import logging
from .base import BaseLLMClient
from .groq_client import GroqClient
from .mock_client import MockLLMClient
from ..config import config

logger = logging.getLogger(__name__)

def get_llm_client(
    provider: Optional[str] = None, 
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    simulate_position_bias: bool = False,
    simulate_verbosity_bias: bool = False
) -> BaseLLMClient:
    """
    Factory function to instantiate the configured LLM client.
    Defaults to Groq if key is present, otherwise falls back to MockLLMClient with a clear warning.
    """
    from typing import Optional
    selected_provider = (provider or config.llm_provider).lower()
    selected_model = model or config.judge_model
    selected_key = api_key or config.groq_api_key

    if selected_provider == "groq":
        if not selected_key or selected_key.strip() == "" or "your_groq_api_key" in selected_key:
            logger.warning(
                "GROQ_API_KEY is not set or is using placeholder. "
                "Falling back to high-fidelity MockLLMClient for deterministic execution."
            )
            return MockLLMClient(
                simulate_position_bias=simulate_position_bias,
                simulate_verbosity_bias=simulate_verbosity_bias
            )
        return GroqClient(api_key=selected_key, default_model=selected_model)

    elif selected_provider == "mock":
        return MockLLMClient(
            simulate_position_bias=simulate_position_bias,
            simulate_verbosity_bias=simulate_verbosity_bias
        )

    else:
        logger.warning(f"Unknown provider '{selected_provider}', defaulting to MockLLMClient.")
        return MockLLMClient(
            simulate_position_bias=simulate_position_bias,
            simulate_verbosity_bias=simulate_verbosity_bias
        )
