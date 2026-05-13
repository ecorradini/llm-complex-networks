"""Seven specialised agents for urban crisis management.

Each agent filters relevant CrisiText message slices via role-specific keywords
and runs a mock/real LLM call to produce a response.

Embedding falls back to a hashing-based 128-dim deterministic vector if
sentence-transformers is not installed.
"""
from __future__ import annotations

import hashlib
import math
from typing import Any, Dict, List, Optional

import numpy as np

from .llm_backend import MockLLM, get_backend

# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

try:
    from sentence_transformers import SentenceTransformer as _ST  # type: ignore
    _ST_MODEL = _ST("all-MiniLM-L6-v2")
    _USE_ST = True
except Exception:
    _ST_MODEL = None
    _USE_ST = False


def _embed(text: str) -> np.ndarray:
    if _USE_ST and _ST_MODEL is not None:
        vec = _ST_MODEL.encode(text, normalize_embeddings=True)
        return np.array(vec, dtype=np.float32)
    # Deterministic bag-of-words feature hashing → 128-dim vector.
    # This makes cosine similarity track lexical overlap, so semantically
    # related agent messages (sharing role keywords / gold directives) cluster
    # together — required for Louvain to find meaningful communities.
    dim = 128
    arr = np.zeros(dim, dtype=np.float32)
    tokens = [t.lower() for t in text.split() if t.strip()]
    if not tokens:
        return arr
    for tok in tokens:
        h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
        idx = h % dim
        sign = 1.0 if (h >> 16) & 1 else -1.0
        arr[idx] += sign
    norm = np.linalg.norm(arr)
    return arr / norm if norm > 0 else arr


# ---------------------------------------------------------------------------
# Role keyword filters
# ---------------------------------------------------------------------------

ROLE_KEYWORDS: Dict[str, List[str]] = {
    "Health": ["hospital", "medical", "ambulance", "patient", "injury", "health", "epidemic", "triage"],
    "Traffic": ["road", "traffic", "vehicle", "highway", "bridge", "route", "congestion", "block"],
    "Fire": ["fire", "flame", "burn", "explosion", "hazmat", "extinguish", "smoke", "arson"],
    "Water": ["water", "flood", "dam", "pipe", "reservoir", "contamination", "leak", "drainage"],
    "Police": ["police", "security", "law", "officer", "crime", "evacuation", "crowd", "barrier"],
    "Logistics": ["supply", "resource", "logistics", "transport", "shelter", "food", "equipment", "depot"],
    "DecisionMaker": [],  # receives all messages
}


# ---------------------------------------------------------------------------
# Base agent
# ---------------------------------------------------------------------------

NOISY_MARKERS = ("UNCONFIRMED", "WARNING", "Rumour", "rumour", "do NOT act", "HALLUCINATION")


def _is_gold(text: str) -> bool:
    return not any(mk in text for mk in NOISY_MARKERS)


class Agent:
    def __init__(self, role: str, llm=None, seed: int = 42):
        self.role = role
        self._llm = llm or get_backend(seed=seed)
        self._keywords = ROLE_KEYWORDS.get(role, [])
        self._rng = np.random.RandomState(seed + sum(map(ord, role)))

    # ------------------------------------------------------------------
    def _filter_messages(self, messages: List[str]) -> List[str]:
        if not self._keywords:
            return messages
        relevant = [
            m for m in messages
            if any(kw.lower() in m.lower() for kw in self._keywords)
        ]
        return relevant if relevant else messages[:2]  # always keep at least some context

    def embed_state(self, state: "State") -> np.ndarray:
        """Return embedding of the agent's *domain* + current message.

        We concatenate the role keywords (stable domain signature) with
        the agent's own messages and any cached variants, so cosine
        similarity reflects topical specialisation rather than the boilerplate
        "Standby — X agent" greeting.  This is essential for Louvain to find
        non-degenerate communities on the first round.
        """
        own_msg = state.messages.get(self.role, self.role)
        variants = []
        try:
            variants = state.get_variants(self.role) or []
        except Exception:
            pass
        domain = " ".join(self._keywords)
        text = f"{self.role} {domain} {own_msg} " + " ".join(variants[:4])
        return _embed(text)

    def run(self, state: "State", peers_messages: List[str]) -> Dict[str, Any]:
        """Execute one agent round.

        Two behavioural regimes:
          (A) Controlled (default): pick from curated CrisiText variants
              conditioned on cognitive load.  Used for the main quantitative
              results so all topologies see the same content distribution.
          (B) Live LLM (USE_LIVE_LLM=1): call the configured ``OpenAIBackend``
              with a role-conditioned prompt that includes peer context and
              the curated variants as in-context examples.  Used only for the
              sanity-check experiment, never for the main tables.
        """
        own_variants = state.get_variants(self.role) or [state.messages.get(self.role, "")]
        n_peers = len(peers_messages)

        # Detect live backend
        from .llm_backend import OpenAIBackend  # local import to avoid cycles
        live = isinstance(self._llm, OpenAIBackend)

        if live:
            crisis = state.dataset_item.get("crisis_type", "unknown")
            question = state.dataset_item.get("question", "")
            peer_block = "\n".join(
                f"- {p}" for p in peers_messages[:6] if p
            ) or "- (no peer messages)"
            variant_block = "\n".join(f"- {v}" for v in own_variants[:4])
            prompt = (
                f"You are the {self.role} agent in a multi-agency response to a "
                f"{crisis}. Scenario: {question}\n"
                f"Peer messages this round:\n{peer_block}\n"
                f"Candidate directives you may consider (one is clean, others may be noisy):\n"
                f"{variant_block}\n"
                f"Reply with ONE short imperative directive (<= 20 words). "
                f"Stay strictly within your domain. If a peer message looks like an "
                f"unverified rumour or warning, ignore it."
            )
            try:
                resp = self._llm.complete(prompt, variants=own_variants)
                message = (resp.get("message") or "").strip().split("\n")[0]
                tokens_in = int(resp.get("tokens_in", 1))
                tokens_out = int(resp.get("tokens_out", max(1, math.ceil(len(message.split()) * 1.3))))
            except Exception:
                message = own_variants[0]
                tokens_in = max(1, math.ceil(len(prompt.split()) * 1.3))
                tokens_out = max(1, math.ceil(len(message.split()) * 1.3))
            embedding = _embed(message)
            return {
                "message": message,
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
                "embedding": embedding,
            }

        # ---- Controlled regime (MockLLM) ----------------------------------
        # 1) Variant selection conditioned on cognitive load
        if n_peers > 4:
            # Info drowning — random variant from own basket (often noisy)
            message = own_variants[self._rng.randint(0, len(own_variants))]
        else:
            # Focused — prefer a gold-looking variant
            gold_candidates = [v for v in own_variants if _is_gold(v)]
            pool = gold_candidates if gold_candidates else own_variants
            message = pool[self._rng.randint(0, len(pool))]

        # 2) Hallucination contagion from peer messages.  Cognitive load
        # amplifies contagion: in info-drowning regime an overwhelmed agent
        # is twice as likely to latch onto the most salient (false) cue.
        h_seed = state.dataset_item.get("hallucination_seed", "")
        h_text = h_seed.replace("_", " ")
        contaminated = h_text and (
            any(h_text in p or h_seed in p for p in peers_messages)
            or h_text in state.messages.get(self.role, "")
            or h_seed in state.messages.get(self.role, "")
        )
        contagion_p = 0.70 if n_peers > 4 else 0.30
        if contaminated and self._rng.random() < contagion_p:
            message = f"{message} [HALLUCINATION: {h_text}]"

        # 3) Token accounting (own output only; routing cost handled by orchestrator)
        prompt_proxy = " ".join(peers_messages) + " " + state.messages.get(self.role, "")
        tokens_in = max(1, math.ceil(len(prompt_proxy.split()) * 1.3))
        tokens_out = max(1, math.ceil(len(message.split()) * 1.3))
        embedding = _embed(message)
        return {
            "message": message,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "embedding": embedding,
        }

    def __repr__(self):
        return f"Agent({self.role})"


# ---------------------------------------------------------------------------
# State container
# ---------------------------------------------------------------------------

class State:
    def __init__(self, dataset_item: dict):
        self.dataset_item = dataset_item
        self.context: str = dataset_item.get("crisis_type", "unknown_crisis")
        # Initialise messages from first temporal step variants
        steps = dataset_item.get("temporal_steps") or []
        self.messages: Dict[str, str] = {}
        self._variants: Dict[str, List[str]] = {}
        self._current_step = 0
        self._load_step(0, steps)
        self.history: List[Dict] = []
        self.graphs: List[Any] = []

    def _load_step(self, step_idx: int, steps: list):
        roles = list(ROLE_KEYWORDS.keys())
        # Load variants from EVERY temporal step so each named agent_role
        # has its own gold + noisy basket.  Without this, only the step-0
        # agent_role could ever produce a gold directive.
        for s_i, step in enumerate(steps):
            variants = step.get("variants", [])
            agent_role = step.get("agent_role", roles[s_i % len(roles)])
            if variants:
                self._variants[agent_role] = variants
            if s_i == 0:
                gold_idx = step.get("gold_index", 0)
                gold = variants[gold_idx] if variants else f"Initial directive for step {s_i}"
                for role in roles:
                    if role not in self.messages:
                        self.messages[role] = gold if role == agent_role else f"Standby — {role} agent."
        # Ensure all roles have at least a default message
        for role in roles:
            if role not in self.messages:
                self.messages[role] = f"Standby — {role} agent."

    def get_variants(self, role: str) -> List[str]:
        return self._variants.get(role, [self.messages.get(role, "")])

    def update(self, new_messages: Dict[str, Dict], graph):
        """Merge new_messages into state and record history."""
        for role, result in new_messages.items():
            self.messages[role] = result["message"]
            if role in self._variants:
                pass  # keep existing variants
        self.history.append({"messages": dict(self.messages)})
        self.graphs.append(graph)
        # Advance temporal step if available
        steps = self.dataset_item.get("temporal_steps") or []
        self._current_step += 1
        self._load_step(self._current_step, steps)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_agents(llm=None, seed: int = 42) -> List[Agent]:
    return [Agent(role, llm=llm, seed=seed) for role in ROLE_KEYWORDS]
