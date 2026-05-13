"""Meaning Preservation Entropy (MPE).

Entropy of cluster assignments of round-r embeddings (k-means k=3)
averaged across rounds.
"""
from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional

import numpy as np
from sklearn.cluster import KMeans  # type: ignore


def meaning_preservation_entropy(
    messages_per_round: List[Dict[str, str]],
    embedder: Optional[Callable[[str], np.ndarray]] = None,
    k: int = 3,
) -> float:
    """Compute MPE.

    Parameters
    ----------
    messages_per_round : list of {agent_role: message} dicts
    embedder : callable str → np.ndarray
    k : number of clusters for k-means
    """
    if not messages_per_round:
        return 0.0

    if embedder is None:
        from ..agents import _embed as embedder  # type: ignore

    entropies = []
    for round_msgs in messages_per_round:
        messages = list(round_msgs.values())
        if not messages:
            continue
        embeddings = np.stack([embedder(m) for m in messages])
        n_clusters = min(k, len(messages))
        if n_clusters < 2:
            entropies.append(0.0)
            continue
        km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = km.fit_predict(embeddings)
        # Compute entropy of label distribution
        counts = np.bincount(labels, minlength=n_clusters)
        probs = counts / counts.sum()
        H = 0.0
        for p in probs:
            if p > 1e-12:
                H -= p * math.log(p)
        entropies.append(H)

    return float(np.mean(entropies)) if entropies else 0.0
