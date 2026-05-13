"""Experiment: CHPI heatmap per topology (Figure 3).

Heatmap: rows = topologies, columns = crisis categories, values = mean CHPI.
"""
from __future__ import annotations

import statistics
from collections import defaultdict
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ._utils import base_parser, RESULTS_FIGURES
from src.simulator import load_dataset, sample_items, inject_hallucination
from src.agents import make_agents, State
from src.orchestrator import Orchestrator
from src.topologies.mesh import MeshTopology
from src.topologies.star import StarTopology
from src.topologies.dynamic_cna import DynamicCNATopology


CRISIS_CATEGORIES = [
    "flood", "earthquake", "fire", "hurricane", "industrial_explosion",
    "bridge_collapse", "blackout", "epidemic", "landslide", "gas_leak",
]


def collect_chpi_by_category(topology, items, rounds, seed):
    agents = make_agents(seed=seed)
    orch = Orchestrator(topology, agents=agents)
    chpi_by_cat = defaultdict(list)
    for item in items:
        state = State(item)
        inject_hallucination(state, seed=seed)
        result = orch.run(item, rounds=rounds)
        cat = item.get("crisis_type", "unknown")
        chpi_by_cat[cat].append(result["chpi"])
    return {cat: statistics.mean(v) for cat, v in chpi_by_cat.items()}


def main():
    p = base_parser("CHPI heatmap — Figure 3")
    args = p.parse_args()

    dataset = load_dataset("crisitext", seed=args.seed)
    items = sample_items(dataset, args.n_crises, seed=args.seed)

    topologies = [
        ("Mesh",        MeshTopology()),
        ("Star",        StarTopology()),
        ("Dynamic CNA", DynamicCNATopology()),
    ]

    matrix = []
    topo_names = []
    for name, topo in topologies:
        print(f"  {name}...")
        chpi_map = collect_chpi_by_category(topo, items, args.rounds, args.seed)
        row = [chpi_map.get(cat, 0.0) for cat in CRISIS_CATEGORIES]
        matrix.append(row)
        topo_names.append(name)

    data = np.array(matrix)

    fig, ax = plt.subplots(figsize=(11, 3))
    im = ax.imshow(data, aspect="auto", cmap="YlOrRd", vmin=0.0, vmax=1.0)
    fig.colorbar(im, ax=ax, label="CHPI")

    ax.set_xticks(range(len(CRISIS_CATEGORIES)))
    ax.set_xticklabels(
        [c.replace("_", "\n") for c in CRISIS_CATEGORIES], fontsize=8
    )
    ax.set_yticks(range(len(topo_names)))
    ax.set_yticklabels(topo_names)
    ax.set_title("CHPI Heatmap per Topology and Crisis Type")

    # Annotate cells
    for i in range(len(topo_names)):
        for j in range(len(CRISIS_CATEGORIES)):
            val = data[i, j]
            color = "white" if val > 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=7, color=color)

    fig.tight_layout()
    out = RESULTS_FIGURES / "fig_chpi_heatmap.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {out}")
    print("CHPI heatmap experiment done.")


if __name__ == "__main__":
    main()
