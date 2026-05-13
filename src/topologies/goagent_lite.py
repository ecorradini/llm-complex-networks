"""GoAgent-lite topology — competitor: group-based clustering with leader edges.

k-means on agent embeddings → G groups (default G=3).
Intra-group: dense (all-to-all).
Inter-group: only group-leader ↔ group-leader edges.
Leader = agent with highest in-group degree (proxy: closest to centroid).
"""
from __future__ import annotations

from typing import Any, Dict, List

import networkx as nx
import numpy as np
from sklearn.cluster import KMeans  # type: ignore

from .base import TopologyController


class GoAgentLiteTopology(TopologyController):
    name = "GoAgent-lite"

    def __init__(self, n_groups: int = 3, seed: int = 42):
        self.n_groups = n_groups
        self._seed = seed

    def build_or_update(
        self,
        agents: List[Any],
        embeddings: Dict[str, np.ndarray],
        state: Any,
        round_idx: int,
    ) -> nx.DiGraph:
        roles = [a.role for a in agents]
        n = len(roles)
        G = nx.DiGraph()
        G.add_nodes_from(roles)

        if n == 0:
            return G

        # Build embedding matrix
        emb_list = [embeddings.get(r, np.zeros(128)) for r in roles]
        X = np.stack(emb_list)

        n_clusters = min(self.n_groups, n)
        km = KMeans(n_clusters=n_clusters, random_state=self._seed, n_init=10)
        labels = km.fit_predict(X)

        # Build group membership
        groups: Dict[int, List[str]] = {}
        for role, lab in zip(roles, labels):
            groups.setdefault(lab, []).append(role)

        # Find leader of each group: closest to centroid
        leaders: Dict[int, str] = {}
        for lab, members in groups.items():
            centroid = km.cluster_centers_[lab]
            dists = [
                np.linalg.norm(embeddings.get(m, np.zeros(128)) - centroid)
                for m in members
            ]
            leaders[lab] = members[int(np.argmin(dists))]

        # Intra-group: dense edges
        for members in groups.values():
            for src in members:
                for dst in members:
                    if src != dst:
                        G.add_edge(src, dst, weight=1.0)

        # Inter-group: leader ↔ leader only
        leader_list = list(leaders.values())
        for i, l1 in enumerate(leader_list):
            for l2 in leader_list[i + 1:]:
                G.add_edge(l1, l2, weight=0.5)
                G.add_edge(l2, l1, weight=0.5)

        return G
