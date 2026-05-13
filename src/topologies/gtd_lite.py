"""GTD-lite topology — competitor: proxy-scored sparse edge sampling.

Per-edge proxy score: s_ij = utility * exp(-cost) - lambda * risk
Edge activated with Bernoulli(σ(s_ij)).  No neural training required.
"""
from __future__ import annotations

import math
import random
from typing import Any, Dict, List

import networkx as nx
import numpy as np

from .base import TopologyController


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


class GTDLiteTopology(TopologyController):
    name = "GTD-lite"

    def __init__(self, lam: float = 0.1, seed: int = 42):
        self.lam = lam
        self._rng = random.Random(seed)

    def build_or_update(
        self,
        agents: List[Any],
        embeddings: Dict[str, np.ndarray],
        state: Any,
        round_idx: int,
    ) -> nx.DiGraph:
        roles = [a.role for a in agents]
        G = nx.DiGraph()
        G.add_nodes_from(roles)
        for src in roles:
            for dst in roles:
                if src == dst:
                    continue
                e_i = embeddings.get(src)
                e_j = embeddings.get(dst)
                if e_i is None or e_j is None:
                    utility = 0.5
                else:
                    norm_i = np.linalg.norm(e_i)
                    norm_j = np.linalg.norm(e_j)
                    if norm_i < 1e-9 or norm_j < 1e-9:
                        utility = 0.5
                    else:
                        utility = float(np.dot(e_i, e_j) / (norm_i * norm_j))

                cost = 0.3 + 0.4 * self._rng.random()   # proxy token cost [0.3, 0.7]
                risk = self._rng.random() * 0.5          # proxy risk [0, 0.5]

                score = utility * math.exp(-cost) - self.lam * risk
                prob = _sigmoid(score)
                if self._rng.random() < prob:
                    G.add_edge(src, dst, weight=float(utility))
        return G
