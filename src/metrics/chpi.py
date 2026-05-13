"""Cascaded Hallucination Propagation Index (CHPI).

CHPI = fraction of (round, agent) pairs whose message contains
the hallucination-seed token(s). Values in [0,1]; lower is better.
"""
from __future__ import annotations

from typing import Dict, List


def cascaded_hallucination_propagation_index(
    messages_per_round: List[Dict[str, str]],
    seed_tokens: List[str],
) -> float:
    if not messages_per_round or not seed_tokens:
        return 0.0
    needles = [t.lower().replace("_", " ") for t in seed_tokens] + \
              [t.lower() for t in seed_tokens]
    contaminated = 0
    total = 0
    for round_msgs in messages_per_round:
        for _, msg in round_msgs.items():
            total += 1
            ml = msg.lower()
            if any(n and n in ml for n in needles):
                contaminated += 1
    return contaminated / total if total else 0.0
