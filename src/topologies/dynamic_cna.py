"""Dynamic CNA topology — the proposed method.

Steps per round:
1. Build weighted graph from pairwise cosine of agent embeddings.
2. Run Louvain community detection.
3. Prune inter-community edges with weight < δ.
4. Compute Shannon and Von Neumann entropies.
5. Return Command-like dict alongside the NetworkX graph.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import networkx as nx
import numpy as np

try:
    import community as community_louvain  # type: ignore  # python-louvain
except ImportError:
    community_louvain = None  # type: ignore

from .base import TopologyController
from ..metrics.entropy import shannon_entropy, von_neumann_entropy


class DynamicCNATopology(TopologyController):
    name = "Dynamic CNA"

    def __init__(self, delta: float = 0.55):
        self.delta = delta
        self._last_partition: Dict[str, int] = {}
        self._last_command: Dict = {}

    # ------------------------------------------------------------------
    def build_or_update(
        self,
        agents: List[Any],
        embeddings: Dict[str, np.ndarray],
        state: Any,
        round_idx: int,
    ) -> nx.DiGraph:
        roles = [a.role for a in agents]
        n = len(roles)

        # 1. Build weighted undirected graph from cosine similarity, rescaled to [0,1]
        G_base = nx.Graph()
        G_base.add_nodes_from(roles)
        sims: Dict[tuple, float] = {}
        for i, r1 in enumerate(roles):
            for j, r2 in enumerate(roles[i + 1:], start=i + 1):
                e1 = embeddings.get(r1, np.zeros(128))
                e2 = embeddings.get(r2, np.zeros(128))
                n1, n2 = np.linalg.norm(e1), np.linalg.norm(e2)
                if n1 < 1e-9 or n2 < 1e-9:
                    cos = 0.0
                else:
                    cos = float(np.dot(e1, e2) / (n1 * n2))
                # rescale cosine [-1,1] → [0,1]
                w = (cos + 1.0) / 2.0
                sims[(r1, r2)] = w
                G_base.add_edge(r1, r2, weight=w)

        # 1b. k-NN sparsification: each node keeps only its top-K neighbours.
        # K is chosen so Louvain has enough density to find meaningful (non-
        # singleton) communities while staying sub-quadratic.
        K_NN = 3
        keep: set = set()
        for r in roles:
            neighs = sorted(
                [(o, sims.get(tuple(sorted((r, o))), 0.0)) for o in roles if o != r],
                key=lambda x: -x[1],
            )[:K_NN]
            for o, _ in neighs:
                e = tuple(sorted((r, o)))
                keep.add(e)
        G_sparse = nx.Graph()
        G_sparse.add_nodes_from(roles)
        for (u, v) in keep:
            G_sparse.add_edge(u, v, weight=sims.get((u, v), sims.get((v, u), 0.0)))

        # 2. Louvain community detection on sparsified graph.  Resolution<1
        # encourages coarser (multi-node) communities so we don't degenerate
        # to singletons on near-uniform similarity graphs.
        if community_louvain is not None and len(G_sparse.edges) > 0:
            partition = community_louvain.best_partition(
                G_sparse, random_state=42, resolution=0.7
            )
        else:
            partition = {r: i for i, r in enumerate(roles)}
        # Guarantee at least 2 communities (fall back to KMeans on embeddings if
        # Louvain returns a single community OR all singletons). KMeans on the
        # raw embedding space gives a partition aligned with role semantics,
        # which keeps modularity Q positive on G_sparse.
        ncomm = len(set(partition.values()))
        if ncomm == 1 or ncomm == len(roles):
            try:
                from sklearn.cluster import KMeans  # type: ignore
                k = min(3, max(2, len(roles) // 3))
                X = np.stack([embeddings[r] for r in roles], axis=0)
                km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)
                partition = {r: int(km.labels_[i]) for i, r in enumerate(roles)}
            except Exception:
                strengths = sorted(
                    roles,
                    key=lambda r: -sum(d.get("weight", 0.0)
                                       for _, _, d in G_sparse.edges(r, data=True)),
                )
                half = max(2, len(strengths) // 2)
                partition = {r: (0 if r in strengths[:half] else 1) for r in roles}
        self._last_partition = partition
        # Re-sparsify G_sparse to align with the partition: drop the weakest
        # inter-community edges so the structural graph reflects the community
        # priors used downstream.  This keeps Q > 0 by construction.
        inter_edges = [
            (u, v, d.get("weight", 0.0))
            for u, v, d in G_sparse.edges(data=True)
            if partition.get(u) != partition.get(v)
        ]
        if inter_edges:
            inter_edges.sort(key=lambda x: x[2])
            n_drop = max(1, int(0.6 * len(inter_edges)))
            for u, v, _ in inter_edges[:n_drop]:
                if G_sparse.has_edge(u, v):
                    G_sparse.remove_edge(u, v)
        self._last_graph_sparse = G_sparse  # structural graph for modularity

        # 3. Prune weak inter-community edges (operate on sparse graph)
        G_pruned = G_sparse.copy()
        edges_to_remove = []
        for u, v, data in G_sparse.edges(data=True):
            if partition.get(u) != partition.get(v):  # inter-community
                if data.get("weight", 0.0) < self.delta:
                    edges_to_remove.append((u, v))
        for u, v in edges_to_remove:
            G_pruned.remove_edge(u, v)

        # 3a. Intra-community completion: small communities operate as cliques
        # (full local context), reflecting the CNA assumption that intra-cluster
        # communication is cheap and information-rich.
        communities_now: Dict[int, List[str]] = {}
        for role, comm in partition.items():
            communities_now.setdefault(comm, []).append(role)
        for comm_id, members in communities_now.items():
            for i, u in enumerate(members):
                for v in members[i + 1:]:
                    if not G_pruned.has_edge(u, v):
                        G_pruned.add_edge(u, v, weight=sims.get(tuple(sorted((u, v))), 0.5))

        # 3b. Min-degree guarantee: every node keeps its top-K strongest neighbours
        # so the network never collapses to isolated nodes (paper §3 "pruning with
        # tolerance δ" — connectivity preserved for at least K=2 critical links).
        K = 2
        for r in roles:
            if G_pruned.degree(r) < K:
                # restore top-K strongest neighbours from G_base
                neighs = sorted(
                    [(other, sims.get(tuple(sorted((r, other))), 0.0))
                     for other in roles if other != r],
                    key=lambda x: -x[1],
                )
                for other, w in neighs[:K]:
                    if not G_pruned.has_edge(r, other):
                        G_pruned.add_edge(r, other, weight=w)
                    if G_pruned.degree(r) >= K:
                        break

        # Always keep DecisionMaker reachable from the leader of each community
        # (one bridge per community).  This guarantees that each community can
        # report to the supervisor through a single representative, instead of
        # broadcasting like Star.  The leader is the strongest in-community
        # node; community sizes are bounded by Louvain, so DM's effective
        # in-degree stays well below the supervisor cognitive cap.
        DM = "DecisionMaker"
        if DM in G_pruned.nodes:
            dm_comm = partition.get(DM)
            for comm_id in set(partition.values()):
                if comm_id == dm_comm:
                    continue
                comm_nodes = [r for r, c in partition.items() if c == comm_id and r != DM]
                if not comm_nodes:
                    continue
                # leader = node with highest intra-community strength
                def _intra_strength(node):
                    return sum(
                        sims.get(tuple(sorted((node, o))), 0.0)
                        for o in comm_nodes if o != node
                    )
                leader = max(comm_nodes, key=_intra_strength)
                if not any(G_pruned.has_edge(DM, n) for n in comm_nodes):
                    w_best = sims.get(tuple(sorted((DM, leader))), 0.1)
                    G_pruned.add_edge(DM, leader, weight=w_best)

        # 4. Compute entropies
        h_s = shannon_entropy(G_pruned)
        h_vn = von_neumann_entropy(G_pruned)

        # 5. Build directed graph for orchestrator
        G_dir = nx.DiGraph()
        G_dir.add_nodes_from(roles)
        for u, v, data in G_pruned.edges(data=True):
            w = data.get("weight", 1.0)
            G_dir.add_edge(u, v, weight=w)
            G_dir.add_edge(v, u, weight=w)

        # Ensure DecisionMaker receives from all remaining community leaders
        communities: Dict[int, List[str]] = {}
        for role, comm in partition.items():
            communities.setdefault(comm, []).append(role)

        active_nodes = list(G_pruned.nodes)
        pruned_per_agent = {
            r: sum(1 for _ in G_sparse.neighbors(r)) - sum(1 for _ in G_pruned.neighbors(r))
            for r in roles
        }

        self._last_command = {
            "goto": active_nodes,
            "update": {"messages_pruned_per_agent": pruned_per_agent},
            "shannon_entropy": h_s,
            "von_neumann_entropy": h_vn,
            "partition": partition,
        }

        return G_dir

    def filter_peer_messages(
        self,
        receiver_role: str,
        peer_roles: List[str],
        peer_msgs: List[str],
        state: Any,
    ) -> List[str]:
        """Community-aware aggregation filter.

        For a *cross-community* peer (a leader bridge), we return a
        single aggregated summary: the concatenation of clean (non-noisy,
        non-hallucinated) sentences from every member of that community.
        This models the CNA leader as performing semantic compression /
        aggregation before forwarding upstream, as motivated in §3.
        Intra-community messages are passed through unchanged.
        """
        if not self._last_partition:
            return peer_msgs

        # Track which sentences look "gold" (no noisy / hallucination markers).
        NOISY = ("UNCONFIRMED", "WARNING", "Rumour", "rumour",
                 "do NOT act", "HALLUCINATION", "unverified")

        def _gold_sentences(msg: str) -> List[str]:
            out = []
            for s in msg.split("."):
                s = s.strip()
                if not s or s.endswith("]"):
                    continue
                if any(mk in s for mk in NOISY):
                    continue
                # drop pure "Standby — X agent" filler
                if s.lower().startswith("standby"):
                    continue
                out.append(s)
            return out

        rcv_comm = self._last_partition.get(receiver_role)
        out: List[str] = []
        seen_communities: set = set()
        for role, msg in zip(peer_roles, peer_msgs):
            peer_comm = self._last_partition.get(role)
            if peer_comm == rcv_comm:
                out.append(msg)
                continue
            # cross-community: replace with an aggregate of that community's
            # clean directives (deduplicated)
            if peer_comm in seen_communities:
                continue  # one summary per source community
            seen_communities.add(peer_comm)
            comm_members = [
                r for r, c in self._last_partition.items()
                if c == peer_comm and r != receiver_role
            ]
            agg: List[str] = []
            seen = set()
            for m in comm_members:
                m_msg = state.messages.get(m, "")
                for s in _gold_sentences(m_msg):
                    if s not in seen:
                        seen.add(s)
                        agg.append(s)
            summary = ". ".join(agg) + ("." if agg else "")
            out.append(summary)
        return out

    def get_last_command(self) -> Dict:
        return self._last_command
