"""Star topology — hub = DecisionMaker, only hub↔leaf edges."""
from __future__ import annotations

from typing import Any, Dict, List

import networkx as nx
import numpy as np

from .base import TopologyController

HUB = "DecisionMaker"


class StarTopology(TopologyController):
    name = "Star"

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
        hub = HUB if HUB in roles else roles[0]
        for role in roles:
            if role != hub:
                G.add_edge(role, hub, weight=1.0)   # leaf → hub
                G.add_edge(hub, role, weight=1.0)   # hub → leaf
        return G
