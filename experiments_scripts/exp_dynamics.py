"""Experiment: Topology dynamics (Figure 2).

Plots modularity Q and Von Neumann entropy H over communication rounds
for Mesh, Star, Dynamic CNA.
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
from src.topologies.mesh import MeshTopology
from src.topologies.star import StarTopology
from src.topologies.dynamic_cna import DynamicCNATopology


def collect_round_metrics(topology, items, rounds, seed):
    agents = make_agents(seed=seed)
    orch = Orchestrator(topology, agents=agents)
    Q_per_round = [[] for _ in range(rounds)]
    H_per_round = [[] for _ in range(rounds)]
    for item in items:
        state = State(item)
        inject_hallucination(state, seed=seed)
        result = orch.run(item, rounds=rounds)
        for log in result["logs"]:
            r = log["round"]
            Q_per_round[r].append(log["modularity"])
            H_per_round[r].append(log["von_neumann_entropy"])
    Q_means = [statistics.mean(v) if v else 0.0 for v in Q_per_round]
    H_means = [statistics.mean(v) if v else 0.0 for v in H_per_round]
    return Q_means, H_means


def main():
    p = base_parser("Dynamics experiment — Figure 2")
    args = p.parse_args()

    dataset = load_dataset("crisitext", seed=args.seed)
    items = sample_items(dataset, args.n_crises, seed=args.seed)

    configs = [
        ("Mesh",        MeshTopology(),        "steelblue",  "--"),
        ("Star",        StarTopology(),         "darkorange", "-."),
        ("Dynamic CNA", DynamicCNATopology(),   "seagreen",   "-"),
    ]

    rounds_range = list(range(args.rounds))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.5))

    for name, topo, color, ls in configs:
        print(f"  {name}...")
        Q, H = collect_round_metrics(topo, items, args.rounds, args.seed)
        ax1.plot(rounds_range, Q, label=name, color=color, linestyle=ls, marker="o")
        ax2.plot(rounds_range, H, label=name, color=color, linestyle=ls, marker="s")

    ax1.set_xlabel("Round")
    ax1.set_ylabel("Modularity Q")
    ax1.set_title("Modularity over Rounds")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2.set_xlabel("Round")
    ax2.set_ylabel("Von Neumann Entropy H")
    ax2.set_title("Entropy over Rounds")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.5)

    fig.suptitle("Topology Dynamics", fontsize=12)
    fig.tight_layout()
    out = RESULTS_FIGURES / "fig_dynamics.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {out}")
    print("Dynamics experiment done.")


if __name__ == "__main__":
    main()
