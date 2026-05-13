"""Global cost function: C_t = α·Σ τ_ij·w_ij − β·ΔI."""
from __future__ import annotations

from typing import Dict

import networkx as nx


def global_cost(
    graph: nx.DiGraph,
    tokens_dict: Dict[str, Dict[str, int]],
    delta_info: float,
    alpha: float = 1.0,
    beta: float = 0.5,
) -> float:
    """Compute C_t = α·Σ τ_ij·w_ij − β·ΔI.

    Parameters
    ----------
    graph : directed graph with edge attribute 'weight'
    tokens_dict : {src: {dst: token_count}} — τ_ij
    delta_info : ΔI — information gain (entropy reduction)
    alpha : weight on token cost
    beta : weight on information gain
    """
    token_cost = 0.0
    for src, dst, data in graph.edges(data=True):
        w = data.get("weight", 1.0)
        tau = tokens_dict.get(src, {}).get(dst, 0)
        token_cost += tau * w
    return float(alpha * token_cost - beta * delta_info)
