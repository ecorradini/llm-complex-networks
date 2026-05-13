"""Experiment: Ablation study on α, β, δ (Figure 1).

Sweeps each hyperparameter independently while holding others constant,
plotting F1 vs. parameter value.
"""
from __future__ import annotations

import statistics
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ._utils import base_parser, RESULTS_FIGURES
from src.simulator import load_dataset, sample_items, inject_hallucination
from src.agents import make_agents, State
from src.orchestrator import Orchestrator
from src.topologies.dynamic_cna import DynamicCNATopology


def eval_f1(items, rounds, seed, alpha=1.0, beta=0.5, delta=0.35):
    topo = DynamicCNATopology(delta=delta)
    agents = make_agents(seed=seed)
    orch = Orchestrator(topo, agents=agents, alpha=alpha, beta=beta)
    f1s = []
    for item in items:
        state = State(item)
        inject_hallucination(state, seed=seed)
        result = orch.run(item, rounds=rounds)
        f1s.append(result["f1"])
    return statistics.mean(f1s)


def main():
    p = base_parser("Ablation experiment — Figure 1")
    args = p.parse_args()

    dataset = load_dataset("crisitext", seed=args.seed)
    items = sample_items(dataset, args.n_crises, seed=args.seed)

    alpha_vals = [0.2, 0.5, 1.0, 2.0, 5.0]
    beta_vals  = [0.1, 0.3, 0.5, 1.0, 2.0]
    delta_vals = [0.1, 0.2, 0.35, 0.5, 0.7]

    print("Sweeping α...")
    f1_alpha = [eval_f1(items, args.rounds, args.seed, alpha=a) for a in tqdm(alpha_vals)]
    print("Sweeping β...")
    f1_beta  = [eval_f1(items, args.rounds, args.seed, beta=b)  for b in tqdm(beta_vals)]
    print("Sweeping δ...")
    f1_delta = [eval_f1(items, args.rounds, args.seed, delta=d) for d in tqdm(delta_vals)]

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))

    axes[0].plot(alpha_vals, f1_alpha, marker="o", color="steelblue")
    axes[0].set_xlabel(r"$\alpha$ (token-cost weight)")
    axes[0].set_ylabel("F1")
    axes[0].set_title(r"Ablation: $\alpha$")
    axes[0].grid(True, linestyle="--", alpha=0.5)

    axes[1].plot(beta_vals, f1_beta, marker="s", color="darkorange")
    axes[1].set_xlabel(r"$\beta$ (info-gain weight)")
    axes[1].set_title(r"Ablation: $\beta$")
    axes[1].grid(True, linestyle="--", alpha=0.5)

    axes[2].plot(delta_vals, f1_delta, marker="^", color="seagreen")
    axes[2].set_xlabel(r"$\delta$ (pruning threshold)")
    axes[2].set_title(r"Ablation: $\delta$")
    axes[2].grid(True, linestyle="--", alpha=0.5)

    for ax in axes:
        ax.set_ylim(0, 1.05)

    fig.suptitle("Ablation Study — Dynamic CNA", fontsize=12)
    fig.tight_layout()
    out = RESULTS_FIGURES / "fig_ablation.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {out}")
    print("Ablation experiment done.")


if __name__ == "__main__":
    main()
