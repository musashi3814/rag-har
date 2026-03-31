"""
LLM Client - Protocol-based interface for LLM classification.

Supports OpenAI API and local LLMs via OpenAI-compatible endpoints (vLLM, Ollama, etc.).
"""

import os
import re
import time
import logging
from typing import List, Optional, Protocol, runtime_checkable

from openai import OpenAI
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class ActivityPrediction(BaseModel):
    """Structured output for activity classification."""
    activity_label: str


@runtime_checkable
class LLMClient(Protocol):
    """Protocol for LLM classification clients."""

    def classify(
        self,
        system_prompt: str,
        user_prompt: str,
        valid_labels: List[str],
    ) -> str:
        """
        Classify an activity using the LLM.

        Args:
            system_prompt: System instruction for classification
            user_prompt: User prompt with candidate and retrieved samples
            valid_labels: List of valid activity labels

        Returns:
            Predicted activity label string
        """
        ...


class OpenAIClient:
    """OpenAI API client using structured output."""

    def __init__(self, model: str, api_key: str):
        self.model = model
        self.client = OpenAI(api_key=api_key)

    def classify(
        self,
        system_prompt: str,
        user_prompt: str,
        valid_labels: List[str],
    ) -> str:
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.client.beta.chat.completions.parse(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    response_format=ActivityPrediction,
                    timeout=60.0,
                )
                return response.choices[0].message.parsed.activity_label
            except Exception as e:
                import openai as openai_module
                if isinstance(e, openai_module.RateLimitError):
                    logger.warning(f"Rate limit reached: {e}. Waiting 65 seconds...")
                    time.sleep(65)
                else:
                    logger.warning(
                        f"OpenAI API error (attempt {attempt + 1}/{max_retries}): "
                        f"{type(e).__name__}: {e}"
                    )
                    if attempt < max_retries - 1:
                        time.sleep(10)
        raise RuntimeError(f"Failed to classify after {max_retries} retries")


class LocalLLMClient:
    """Local LLM client via OpenAI-compatible endpoint (vLLM, Ollama, etc.)."""

    def __init__(self, model: str, base_url: str, api_key: Optional[str] = None):
        self.model = model
        # Most OpenAI-compatible servers accept any API key
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key or "not-needed",
        )

    def classify(
        self,
        system_prompt: str,
        user_prompt: str,
        valid_labels: List[str],
    ) -> str:
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.0,
                    timeout=120.0,
                )
                raw = response.choices[0].message.content.strip()
                return self._extract_label(raw, valid_labels)
            except Exception as e:
                logger.warning(
                    f"Local LLM error (attempt {attempt + 1}/{max_retries}): "
                    f"{type(e).__name__}: {e}"
                )
                if attempt < max_retries - 1:
                    time.sleep(5)
        raise RuntimeError(f"Failed to classify after {max_retries} retries")

    @staticmethod
    def _extract_label(raw: str, valid_labels: List[str]) -> str:
        """Extract activity label from raw LLM output."""
        # Try exact match first
        for label in valid_labels:
            if raw.lower() == label.lower():
                return label

        # Try to find label within the response
        lower_raw = raw.lower()
        for label in valid_labels:
            if label.lower() in lower_raw:
                return label

        # Return raw output as fallback
        logger.warning(f"Could not match label from response: {raw}")
        return raw


def get_llm_client(config: dict) -> LLMClient:
    """
    Factory function to create LLM client from config.

    Config expects a top-level 'llm' section:
        llm:
          provider: "openai" | "local"
          model: "gpt-5-mini"
          base_url: null          # for local: "http://localhost:8000/v1"
          api_key: null           # for local: optional override

    Falls back to OPENAI_API_KEY env var for OpenAI provider if no llm section.
    """
    llm_config = config.get("llm", {})
    provider = llm_config.get("provider", "openai")
    model = llm_config.get("model", "gpt-5-mini")

    if provider == "local":
        base_url = llm_config.get("base_url")
        if not base_url:
            raise ValueError("llm.base_url is required for local provider")
        api_key = llm_config.get("api_key")
        logger.info(f"Using local LLM: {model} at {base_url}")
        return LocalLLMClient(model=model, base_url=base_url, api_key=api_key)

    # Default: OpenAI
    api_key = llm_config.get("api_key") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set and llm.api_key not provided")
    logger.info(f"Using OpenAI: {model}")
    return OpenAIClient(model=model, api_key=api_key)
