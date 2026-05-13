"""Micro-F1 over multi-label action sets."""
from __future__ import annotations

from typing import List, Set, Union


def decision_f1(
    predicted_actions: Union[List[str], Set[str]],
    ground_truth_actions: Union[List[str], Set[str]],
) -> float:
    """Compute micro-F1 over multi-label action sets.

    Both inputs are treated as sets of string labels.
    Uses substring matching to handle minor phrasing differences.
    """
    pred = set(str(p).strip().lower() for p in predicted_actions)
    gold = set(str(g).strip().lower() for g in ground_truth_actions)

    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0

    # Exact match first
    tp_exact = len(pred & gold)

    # Partial (substring) match for remaining
    matched_pred = set()
    matched_gold = set()
    for p in pred:
        for g in gold:
            if p in g or g in p:
                matched_pred.add(p)
                matched_gold.add(g)

    tp = max(tp_exact, len(matched_pred))
    precision = tp / len(pred) if pred else 0.0
    recall = tp / len(gold) if gold else 0.0

    if precision + recall < 1e-12:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


# DEBUG_ME = True
