"""Micro-F1 over multi-label action sets."""
from __future__ import annotations

import re
from typing import List, Set, Union


_STOP = {
    "a", "an", "the", "to", "for", "in", "on", "of", "at", "and", "or",
    "is", "are", "be", "with", "as", "by", "please", "immediately", "now",
    "all", "this", "that", "from", "into",
}


def _toks(s: str) -> Set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", s.lower()) if t not in _STOP}


def decision_f1(
    predicted_actions: Union[List[str], Set[str]],
    ground_truth_actions: Union[List[str], Set[str]],
) -> float:
    """Compute micro-F1 over multi-label action sets.

    Both inputs are treated as sets of string labels.
    Uses substring matching first, then a token-overlap fallback
    (Jaccard >= 0.5 on content words) to tolerate paraphrasing from a
    free-text LLM backend.
    """
    pred = [str(p).strip().lower() for p in predicted_actions if str(p).strip()]
    gold = [str(g).strip().lower() for g in ground_truth_actions if str(g).strip()]
    pred_set, gold_set = set(pred), set(gold)

    if not pred_set and not gold_set:
        return 1.0
    if not pred_set or not gold_set:
        return 0.0

    matched_pred: Set[str] = set()
    matched_gold: Set[str] = set()
    for p in pred_set:
        pt = _toks(p)
        for g in gold_set:
            if p == g or p in g or g in p:
                matched_pred.add(p)
                matched_gold.add(g)
                continue
            gt = _toks(g)
            if not pt or not gt:
                continue
            inter = len(pt & gt)
            union = len(pt | gt)
            if union and (inter / union) >= 0.5:
                matched_pred.add(p)
                matched_gold.add(g)

    tp = max(len(matched_pred), len(matched_gold))
    precision = tp / len(pred_set) if pred_set else 0.0
    recall = tp / len(gold_set) if gold_set else 0.0
    if precision + recall < 1e-12:
        return 0.0
    return float(2 * precision * recall / (precision + recall))
