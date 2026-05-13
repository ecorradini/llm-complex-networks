"""Mesh topology — fully-connected graph, all edge weights = 1."""
from __future__ import annotations

from typing import Any, Dict, List

import networkx as nx
import numpy as np

from .base import TopologyController


class MeshTopology(TopologyController):
    name = "Mesh"

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
                if src != dst:
                    G.add_edge(src, dst, weight=1.0)
        return G
