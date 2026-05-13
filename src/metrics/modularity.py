"""Modularity metric — wrapper around python-louvain / networkx."""
from __future__ import annotations

from typing import Dict

import networkx as nx

try:
    import community as community_louvain  # type: ignore
    _HAS_LOUVAIN = True
except ImportError:
    _HAS_LOUVAIN = False


def modularity(G: nx.Graph, partition: Dict[str, int]) -> float:
    """Compute modularity Q for the given partition.

    Falls back to NetworkX's nx.community.modularity if python-louvain unavailable.
    """
    if len(G.nodes) == 0 or len(G.edges) == 0:
        return 0.0

    Gu = G.to_undirected() if G.is_directed() else G

    if _HAS_LOUVAIN:
        try:
            return float(community_louvain.modularity(partition, Gu, weight="weight"))
        except Exception:
            pass

    # Fallback: build communities list and use networkx
    try:
        from networkx.algorithms.community.quality import modularity as nx_mod  # type: ignore
        communities: Dict[int, set] = {}
        for node, comm_id in partition.items():
            communities.setdefault(comm_id, set()).add(node)
        return float(nx_mod(Gu, list(communities.values()), weight="weight"))
    except Exception:
        return 0.0
