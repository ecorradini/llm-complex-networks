"""Simulator — loads dataset and injects hallucination seeds.

At round 0, injects the dataset item's hallucination_seed into ONE random
non-DecisionMaker agent's message buffer.
"""
from __future__ import annotations

import json
import pathlib
import random
from typing import List, Optional

DATA_DIR = pathlib.Path(__file__).parent.parent / "data"


def load_dataset(dataset_name: str = "crisitext", seed: int = 42) -> List[dict]:
    """Load dataset, preferring real → synthetic fallback."""
    # Try real files first
    real_paths = {
        "crisitext": DATA_DIR / "crisitext.json",
        "cityemergency": DATA_DIR / "cityemergency.json",
    }
    synthetic_paths = {
        "crisitext": DATA_DIR / "crisitext_synthetic.json",
        "cityemergency": DATA_DIR / "cityemergency_synthetic.json",
    }

    real_path = real_paths.get(dataset_name)
    if real_path and real_path.exists():
        try:
            data = json.loads(real_path.read_text())
            if isinstance(data, list) and data:
                return data
        except Exception:
            pass

    # Fallback to synthetic
    syn_path = synthetic_paths.get(dataset_name)
    if syn_path and syn_path.exists():
        try:
            return json.loads(syn_path.read_text())
        except Exception:
            pass

    # Auto-generate synthetic if not on disk
    _ensure_synthetic()
    if syn_path and syn_path.exists():
        return json.loads(syn_path.read_text())

    raise FileNotFoundError(
        f"No data available for {dataset_name}. "
        "Run `python data/synthetic_fallback.py` first."
    )


def _ensure_synthetic():
    """Generate synthetic data if not already on disk."""
    import sys
    sys.path.insert(0, str(DATA_DIR.parent))
    from data.synthetic_fallback import main as gen  # type: ignore
    gen()


def inject_hallucination(state, seed: int = 42) -> None:
    """Inject the dataset item's hallucination_seed into a random non-DM agent."""
    dataset_item = state.dataset_item
    h_seed = dataset_item.get("hallucination_seed", "")
    if not h_seed:
        return
    roles = [r for r in state.messages.keys() if r != "DecisionMaker"]
    if not roles:
        return
    rng = random.Random(seed)
    chosen = rng.choice(roles)
    h_text = h_seed.replace("_", " ")
    state.messages[chosen] = state.messages[chosen] + f" [ALERT: {h_text}]"


def sample_items(dataset: List[dict], n: int, seed: int = 42) -> List[dict]:
    """Sample n items deterministically."""
    rng = random.Random(seed)
    if n >= len(dataset):
        return list(dataset)
    return rng.sample(dataset, n)
