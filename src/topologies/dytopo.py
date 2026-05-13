"""DyTopo topology — competitor: semantic query/key matching via cosine similarity.

Each agent's embedding acts as both query and key. Edge (i→j) is activated when
cos(embedding_i, embedding_j) > tau. No LLM call for routing.
"""
from __future__ import annotations

from typing import Any, Dict, List

import networkx as nx
import numpy as np

from .base import TopologyController


class DyTopoTopology(TopologyController):
    name = "DyTopo"

    def __init__(self, tau: float = 0.4):
        self.tau = tau

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
        for i, src in enumerate(roles):
            for j, dst in enumerate(roles):
                if src == dst:
                    continue
                e_i = embeddings.get(src)
                e_j = embeddings.get(dst)
                if e_i is None or e_j is None:
                    continue
                norm_i = np.linalg.norm(e_i)
                norm_j = np.linalg.norm(e_j)
                if norm_i < 1e-9 or norm_j < 1e-9:
                    continue
                cos_sim = float(np.dot(e_i, e_j) / (norm_i * norm_j))
                if cos_sim > self.tau:
                    G.add_edge(src, dst, weight=cos_sim)
        return G
