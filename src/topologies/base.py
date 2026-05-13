"""Base topology controller interface."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List

import networkx as nx
import numpy as np


class TopologyController(ABC):
    """Abstract base class for all topology controllers."""

    name: str = "base"

    @abstractmethod
    def build_or_update(
        self,
        agents: List[Any],
        embeddings: Dict[str, np.ndarray],
        state: Any,
        round_idx: int,
    ) -> nx.DiGraph:
        """Build or update the communication graph.

        Returns a directed weighted NetworkX graph where node names are agent roles.
        """

    def active_edges(self, graph: nx.DiGraph) -> Dict[str, List[str]]:
        """Return {agent_role: [peer_roles_it_receives_from]}."""
        result: Dict[str, List[str]] = {n: [] for n in graph.nodes}
        for src, dst in graph.edges():
            result[dst].append(src)
        return result
