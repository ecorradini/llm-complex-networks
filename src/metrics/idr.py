"""Information Degradation Rate (IDR).

IDR = mean over hops of (1 − cos(emb(msg_hop), emb(source))).
"""
from __future__ import annotations

from typing import Callable, List, Optional

import numpy as np


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def information_degradation_rate(
    messages_chain: List[str],
    source_directive: str,
    embedder: Optional[Callable[[str], np.ndarray]] = None,
) -> float:
    """Compute IDR as mean (1 − cosine similarity) across hops.

    Parameters
    ----------
    messages_chain : list of messages at each hop (hop 0 = source or first relay)
    source_directive : original source message to compare against
    embedder : callable str → np.ndarray; falls back to hashing embedder
    """
    if not messages_chain:
        return 0.0

    if embedder is None:
        from ..agents import _embed as embedder  # type: ignore

    src_emb = embedder(source_directive)
    degradations = []
    for msg in messages_chain:
        hop_emb = embedder(msg)
        cos_sim = _cosine(src_emb, hop_emb)
        degradations.append(1.0 - cos_sim)

    return float(np.mean(degradations))
