"""LLM backend adapters.

Default: MockLLM — deterministic, dataset-driven, no API keys needed.
Optional: OpenAIBackend — activated when OPENAI_API_KEY is set.
"""
from __future__ import annotations

import math
import os
import random
from typing import List, Optional


class MockLLM:
    """Deterministic mock LLM that returns dataset variants and counts tokens via
    whitespace split × 1.3 fudge factor."""

    def __init__(self, seed: int = 42, fudge: float = 1.3):
        self._rng = random.Random(seed)
        self._fudge = fudge

    def _count_tokens(self, text: str) -> int:
        return max(1, math.ceil(len(text.split()) * self._fudge))

    def complete(
        self,
        prompt: str,
        variants: Optional[List[str]] = None,
    ) -> dict:
        """Return a completion dict with message, tokens_in, tokens_out."""
        tokens_in = self._count_tokens(prompt)
        if variants:
            message = self._rng.choice(variants)
        else:
            # fallback: echo a truncated version of the prompt
            words = prompt.split()
            message = " ".join(words[:min(20, len(words))]) + "."
        tokens_out = self._count_tokens(message)
        return {"message": message, "tokens_in": tokens_in, "tokens_out": tokens_out}


class OpenAIBackend:
    """Thin wrapper around the OpenAI chat completion API."""

    def __init__(self, model: str = "gpt-4o-mini"):
        try:
            import openai  # type: ignore
        except ImportError as exc:
            raise ImportError("openai package not installed") from exc
        self._client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self._model = model

    def _count_tokens(self, text: str) -> int:
        return max(1, math.ceil(len(text.split()) * 1.3))

    def complete(self, prompt: str, variants=None) -> dict:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=256,
        )
        message = response.choices[0].message.content or ""
        tokens_in = response.usage.prompt_tokens if response.usage else self._count_tokens(prompt)
        tokens_out = response.usage.completion_tokens if response.usage else self._count_tokens(message)
        return {"message": message, "tokens_in": tokens_in, "tokens_out": tokens_out}


def get_backend(seed: int = 42, fudge: float = 1.3):
    """Return the appropriate backend based on environment."""
    if os.environ.get("OPENAI_API_KEY"):
        try:
            return OpenAIBackend()
        except Exception:
            pass
    return MockLLM(seed=seed, fudge=fudge)
