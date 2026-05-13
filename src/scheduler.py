"""Adaptive parameter scheduler for (alpha, beta, delta).

Returns the operating point of a small control loop:
- alpha_t grows with token-budget pressure (B_used / B_max)^gamma.
- beta_t scales with crisis urgency (per crisis-type prior).
- delta_t tracks the Von Neumann entropy slope: if H_VN is decreasing
  (consolidating), delta is reduced to retain weak useful edges; if it
  plateaus, delta is raised to force partitioning.

The static configuration (alpha_0, beta_0, delta_0) used in the main
experiments is the steady-state operating point of this loop.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List


# Crisis-type urgency priors in [0, 1]. Higher → higher beta.
URGENCY_PRIOR = {
    "earthquake": 1.0,
    "explosion": 1.0,
    "fire": 0.9,
    "flood": 0.8,
    "bridge_collapse": 0.85,
    "industrial_accident": 0.85,
    "epidemic": 0.7,
    "blackout": 0.6,
    "contamination": 0.7,
    "default": 0.7,
}


@dataclass
class AdaptiveScheduler:
    alpha0: float = 1.0
    beta0: float = 2.0
    delta0: float = 0.55
    gamma: float = 1.0
    token_budget: float = 1000.0  # B_max per scenario
    # Clamping ranges
    alpha_min: float = 0.5
    alpha_max: float = 2.5
    beta_min: float = 0.5
    beta_max: float = 3.0
    delta_min: float = 0.30
    delta_max: float = 0.70

    def step(
        self,
        round_idx: int,
        tokens_used: float,
        hvn_history: List[float],
        crisis_type: str = "",
    ):
        # 1) alpha: token-budget pressure (higher → more aggressive cost weight)
        pressure = min(1.5, max(0.0, tokens_used / max(self.token_budget, 1.0)))
        alpha_t = self.alpha0 * (1.0 + pressure ** self.gamma)
        alpha_t = max(self.alpha_min, min(self.alpha_max, alpha_t))

        # 2) beta: urgency prior on the current crisis type
        urgency = URGENCY_PRIOR.get(crisis_type, URGENCY_PRIOR["default"])
        beta_t = self.beta0 * (0.5 + urgency)
        beta_t = max(self.beta_min, min(self.beta_max, beta_t))

        # 3) delta: tighten under pressure (drop more inter-community edges),
        # relax under information consolidation (negative H_VN slope).
        slope = 0.0
        if len(hvn_history) >= 2:
            slope = hvn_history[-1] - hvn_history[-2]
        delta_t = self.delta0 + 0.20 * pressure - 0.10 * slope
        delta_t = max(self.delta_min, min(self.delta_max, delta_t))

        return alpha_t, beta_t, delta_t
