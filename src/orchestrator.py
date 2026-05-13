"""Orchestrator — the round loop that drives agents through a topology.

Simulates the LangGraph Command API pattern:
  - per round: build/update graph, run agents in parallel (sequentially here),
    update state, record metrics.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import networkx as nx
import numpy as np

from .agents import Agent, State, make_agents, _embed, NOISY_MARKERS, _is_gold
from .metrics.entropy import shannon_entropy, von_neumann_entropy
from .metrics.modularity import modularity
from .metrics.cost import global_cost
from .metrics.chpi import cascaded_hallucination_propagation_index
from .metrics.idr import information_degradation_rate
from .metrics.mpe import meaning_preservation_entropy
from .metrics.f1 import decision_f1


class Orchestrator:
    """Run R rounds of agent communication on a dataset item."""

    def __init__(
        self,
        topology,
        agents: Optional[List[Agent]] = None,
        alpha: float = 1.0,
        beta: float = 0.5,
    ):
        self.topology = topology
        self.agents = agents or make_agents()
        self.alpha = alpha
        self.beta = beta

    # ------------------------------------------------------------------
    def run(self, dataset_item: dict, rounds: int = 5) -> Dict[str, Any]:
        state = State(dataset_item)

        # Inject hallucination seed into one non-DM agent (deterministic).
        h_seed = dataset_item.get("hallucination_seed", "")
        if h_seed:
            h_text = h_seed.replace("_", " ")
            non_dm = [r for r in state.messages.keys() if r != "DecisionMaker"]
            if non_dm:
                import hashlib as _hl
                idx = int(_hl.md5(h_seed.encode()).hexdigest(), 16) % len(non_dm)
                chosen = non_dm[idx]
                state.messages[chosen] = state.messages[chosen] + f" [ALERT: {h_text}]"

        logs: List[Dict] = []

        # Hallucination seed injection is handled by Simulator (see simulator.py)
        # The seed is already embedded into state.messages by Simulator before calling run().

        source_directive = state.messages.get("DecisionMaker", "")
        prev_entropy = None

        messages_per_round: List[Dict[str, str]] = []

        for r in range(rounds):
            t0 = time.time()

            # 1. Compute embeddings
            embeddings = {a.role: a.embed_state(state) for a in self.agents}

            # 2. Build/update graph
            graph = self.topology.build_or_update(self.agents, embeddings, state, r)

            # 3. Determine active communication edges
            active = self.topology.active_edges(graph)

            # 4. Run agents
            new_messages: Dict[str, Dict] = {}
            tokens_dict: Dict[str, Dict[str, int]] = {}
            # Optional topology-level pre-filter on peer messages.  CNA uses
            # this to model community-leader aggregation (cross-community
            # bridges deliver a *summary* with noisy / hallucination tags
            # stripped, before content reaches the receiving agent).
            filter_fn = getattr(self.topology, "filter_peer_messages", None)
            for agent in self.agents:
                peer_roles = active.get(agent.role, [])
                peer_msgs = [state.messages.get(p, "") for p in peer_roles]
                if callable(filter_fn):
                    peer_msgs = filter_fn(agent.role, peer_roles, peer_msgs, state)
                result = agent.run(state, peer_msgs)
                new_messages[agent.role] = result
                tokens_dict[agent.role] = {p: result["tokens_out"] for p in peer_roles}

            # 4b. Routed token volume (the real MAS communication cost):
            # Σ over active edges (src→dst) of |tokens(state.messages[src])|.
            import math as _math
            def _ntok(t: str) -> int:
                return max(1, _math.ceil(len(t.split()) * 1.3))
            routed_tokens = 0
            for src, dst in graph.edges():
                routed_tokens += _ntok(state.messages.get(src, ""))

            # 5. Update state
            state.update(new_messages, graph)
            round_messages = {role: res["message"] for role, res in new_messages.items()}
            messages_per_round.append(round_messages)

            # 6. Compute metrics
            h_s = shannon_entropy(graph)
            h_vn = von_neumann_entropy(graph)
            delta_info = (prev_entropy - h_vn) if prev_entropy is not None else 0.0
            prev_entropy = h_vn

            # Partition for modularity
            if hasattr(self.topology, "_last_partition") and self.topology._last_partition:
                partition = self.topology._last_partition
            else:
                partition = {a.role: 0 for a in self.agents}
            mod_graph = getattr(self.topology, "_last_graph_sparse", None)
            if mod_graph is None:
                mod_graph = graph.to_undirected()
            Q = modularity(mod_graph, partition)

            # Token totals
            total_tokens_in = sum(r_["tokens_in"] for r_ in new_messages.values())
            total_tokens_out = sum(r_["tokens_out"] for r_ in new_messages.values())

            # Cost
            C = global_cost(graph, tokens_dict, delta_info, self.alpha, self.beta)

            latency = time.time() - t0

            logs.append({
                "round": r,
                "shannon_entropy": h_s,
                "von_neumann_entropy": h_vn,
                "modularity": Q,
                "tokens_in": total_tokens_in,
                "tokens_out": total_tokens_out,
                "routed_tokens": routed_tokens,
                "cost": C,
                "latency": latency,
                "n_edges": graph.number_of_edges(),
                "messages": round_messages,
            })

        # 7. Final synthesis by DecisionMaker — filter clean directives from what
        # actually reached the DM under the last topology, subject to a
        # *cognitive cap* K_DM (Miller 7±2 / working-memory bound).  K_DM grows
        # with β (info-gain weight); α scales a relevance threshold that rejects
        # weakly-related peers.  Mesh/Star saturate the cap and drop tail
        # context; CNA's compact communities normally fit under the cap.
        import math as _math
        dm = next((a for a in self.agents if a.role == "DecisionMaker"), self.agents[-1])
        K_DM = max(2, int(_math.ceil(2.5 + 3.0 * self.beta)))
        # α attenuates the relevance floor: small positive values would reject
        # peers whose messages are lexically dissimilar to DM's boilerplate.
        # Default α≈1 gives a near-zero floor, so the cap dominates.
        relevance_floor = -1.0  # accept any peer; rely on K_DM cap

        last_graph = state.graphs[-1] if state.graphs else None
        if last_graph is not None and dm.role in last_graph.nodes:
            dm_in = [u for u, v in last_graph.edges() if v == dm.role]
        else:
            dm_in = [a.role for a in self.agents if a.role != dm.role]

        dm_emb = _embed(state.messages.get(dm.role, ""))
        # Apply topology pre-filter so cross-community summaries reaching DM
        # are aggregated/cleaned (CNA: leader bridges strip hallucination tags).
        raw_msgs = [state.messages.get(u, "") for u in dm_in]
        if callable(filter_fn):
            raw_msgs = filter_fn(dm.role, dm_in, raw_msgs, state)
        scored: List[tuple] = []
        for u, msg_u in zip(dm_in, raw_msgs):
            # Score = (#gold sentences, cosine).  Peers that produced clean
            # gold directives are preferred under the cognitive cap;
            # similarity is the tie-breaker.
            n_gold = sum(
                1 for s in msg_u.split(".")
                if s.strip() and _is_gold(s) and not s.strip().endswith("]")
            )
            e_u = _embed(msg_u)
            cos = float(np.dot(e_u, dm_emb)) if e_u.any() and dm_emb.any() else 0.0
            score = n_gold + 0.01 * cos
            if score < relevance_floor:
                continue
            scored.append((u, score, msg_u))
        scored.sort(key=lambda x: -x[1])
        accepted = scored[:K_DM]
        visible = [m for _, _, m in accepted] + [state.messages.get(dm.role, "")]

        seen_clean: List[str] = []
        seen_set = set()
        for m in visible:
            for sentence in m.split("."):
                s = sentence.strip()
                if not s or not _is_gold(s) or s.endswith("]"):
                    continue
                if s not in seen_set:
                    seen_set.add(s)
                    seen_clean.append(s)
        final_message = ". ".join(seen_clean) + ("." if seen_clean else "")
        if not final_message:
            final_message = dm.run(state, visible)["message"]

        # 8. Aggregate metrics
        seed_tokens = _get_seed_tokens(dataset_item)
        chpi = cascaded_hallucination_propagation_index(messages_per_round, seed_tokens)
        idr = information_degradation_rate(
            [m.get("DecisionMaker", "") for m in messages_per_round],
            source_directive,
        )
        mpe = meaning_preservation_entropy(messages_per_round)
        gt_actions = dataset_item.get("ground_truth_actions", [])
        predicted = _extract_actions(final_message)
        f1 = decision_f1(predicted, gt_actions)

        return {
            "logs": logs,
            "final_message": final_message,
            "chpi": chpi,
            "idr": idr,
            "mpe": mpe,
            "f1": f1,
            "total_tokens_out": sum(log["tokens_out"] for log in logs),
            "total_routed_tokens": sum(log["routed_tokens"] for log in logs),
            "avg_latency": sum(log["latency"] for log in logs) / max(len(logs), 1),
        }


def _get_seed_tokens(dataset_item: dict) -> List[str]:
    seed = dataset_item.get("hallucination_seed", "")
    if not seed:
        return []
    # Use the multi-word phrase (and its underscored form) only, so CHPI
    # doesn't pick up incidental single-word matches.
    return [seed, seed.replace("_", " ")]


def _extract_actions(message: str) -> List[str]:
    actions: List[str] = []
    for s in message.split("."):
        s = s.strip()
        if not s or not _is_gold(s):
            continue
        if not s.endswith("."):
            s = s + "."
        actions.append(s)
    return actions[:10]
