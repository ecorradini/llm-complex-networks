"""LLM backend adapters.

Default: MockLLM — deterministic, dataset-driven, no API keys needed.
Optional: OpenAIBackend — activated when OPENAI_API_KEY is set AND
          USE_LIVE_LLM=1 (otherwise we stick to MockLLM to keep the
          quantitative experiments reproducible and free of stochasticity).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import pathlib
import random
from typing import List, Optional


_CACHE_DIR = pathlib.Path(__file__).resolve().parent.parent / "data" / "llm_cache"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)


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
    """Thin wrapper around the OpenAI chat completion API with on-disk caching.

    The cache key is a hash of (model, prompt); responses are persisted under
    ``experiments/data/llm_cache`` so re-running the sanity check experiments
    is free of additional API cost.
    """

    def __init__(self, model: Optional[str] = None, cache: bool = True):
        try:
            import openai  # type: ignore
        except ImportError as exc:
            raise ImportError("openai package not installed") from exc
        self._client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self._model = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        self._cache = cache

    def _count_tokens(self, text: str) -> int:
        return max(1, math.ceil(len(text.split()) * 1.3))

    def _cache_path(self, prompt: str) -> pathlib.Path:
        key = hashlib.sha256(f"{self._model}\n{prompt}".encode("utf-8")).hexdigest()
        return _CACHE_DIR / f"{key}.json"

    def complete(self, prompt: str, variants=None) -> dict:
        if self._cache:
            cp = self._cache_path(prompt)
            if cp.exists():
                try:
                    return json.loads(cp.read_text())
                except Exception:
                    pass
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.2,
        )
        message = response.choices[0].message.content or ""
        tokens_in = response.usage.prompt_tokens if response.usage else self._count_tokens(prompt)
        tokens_out = response.usage.completion_tokens if response.usage else self._count_tokens(message)
        out = {"message": message, "tokens_in": tokens_in, "tokens_out": tokens_out}
        if self._cache:
            try:
                self._cache_path(prompt).write_text(json.dumps(out))
            except Exception:
                pass
        return out


def get_backend(seed: int = 42, fudge: float = 1.3):
    """Return the appropriate backend based on environment.

    Live LLM is enabled only when BOTH ``OPENAI_API_KEY`` is set AND
    ``USE_LIVE_LLM=1``. The default (controlled) backend is MockLLM, so the
    main quantitative results are deterministic and free of model stochasticity.
    """
    if os.environ.get("USE_LIVE_LLM") == "1" and os.environ.get("OPENAI_API_KEY"):
        try:
            return OpenAIBackend()
        except Exception:
            pass
    return MockLLM(seed=seed, fudge=fudge)
