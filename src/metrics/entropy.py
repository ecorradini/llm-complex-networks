"""Shannon and Von Neumann graph entropy."""
from __future__ import annotations

import math

import networkx as nx
import numpy as np


def shannon_entropy(G: nx.Graph) -> float:
    """Normalize weighted degrees to probabilities, return −Σ p log p."""
    if len(G.nodes) == 0:
        return 0.0
    # Weighted degree for each node
    degrees = dict(G.degree(weight="weight"))
    total = sum(degrees.values())
    if total < 1e-12:
        return 0.0
    H = 0.0
    for w in degrees.values():
        p = w / total
        if p > 1e-12:
            H -= p * math.log(p)
    return float(H)


def von_neumann_entropy(G: nx.Graph) -> float:
    """Von Neumann entropy of the normalised Laplacian density matrix.

    Computed manually to avoid a scipy/networkx version-incompatibility in
    ``nx.normalized_laplacian_matrix`` (which raised ``AttributeError`` and
    silently returned 0 under our environment).
    """
    if len(G.nodes) == 0:
        return 0.0
    Gu = G.to_undirected() if G.is_directed() else G
    if len(Gu.edges) == 0:
        return 0.0
    try:
        A = nx.to_numpy_array(Gu, weight="weight")
        d = A.sum(axis=1)
        # D^{-1/2}
        d_safe = np.where(d > 1e-12, d, 1.0)
        dinv_sqrt = 1.0 / np.sqrt(d_safe)
        dinv_sqrt = np.where(d > 1e-12, dinv_sqrt, 0.0)
        n = A.shape[0]
        I = np.eye(n)
        L = I - (dinv_sqrt[:, None] * A * dinv_sqrt[None, :])
        trace = float(np.trace(L))
        if abs(trace) < 1e-12:
            return 0.0
        rho = L / trace
        eigvals = np.linalg.eigvalsh(rho)
        H = 0.0
        for lam in eigvals:
            if lam > 1e-10:
                H -= lam * math.log(lam)
        return float(H)
    except Exception:
        return 0.0
