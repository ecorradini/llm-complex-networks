"""Experiment: K_DM sweep — addresses the cognitive-cap reviewer concern.

Sweeps the supervisor cognitive cap K_DM in {2, 3, 4, 5, 7, 10, inf} on
Star and Dynamic CNA, reports F1 and CHPI per value.

Outputs:
  results/tables/tab_kdm.csv / .tex
  results/figures/fig_kdm.pdf
"""
from __future__ import annotations

import math
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

from ._utils import base_parser, RESULTS_TABLES, RESULTS_FIGURES, write_tex_table, write_csv
from src.simulator import load_dataset, sample_items, inject_hallucination
from src.agents import make_agents, State
from src.orchestrator import Orchestrator
from src.topologies.star import StarTopology
from src.topologies.dynamic_cna import DynamicCNATopology


K_VALUES = [2, 3, 4, 5, 7, 10, float("inf")]


def run_topology(topology_factory, k_dm, items, rounds, seed):
    agents = make_agents(seed=seed)
    orch = Orchestrator(topology_factory(), agents=agents, k_dm=k_dm)
    chpi_list, f1_list = [], []
    for item in items:
        state = State(item)
        inject_hallucination(state, seed=seed)
        result = orch.run(item, rounds=rounds)
        chpi_list.append(result["chpi"])
        f1_list.append(result["f1"])
    return statistics.mean(chpi_list), statistics.mean(f1_list)


def main():
    p = base_parser("K_DM sweep — reviewer concern (cognitive cap)")
    args = p.parse_args()

    dataset = load_dataset("cityemergency", seed=args.seed)
    items = sample_items(dataset, args.n_crises, seed=args.seed)

    methods = [
        ("Star", lambda: StarTopology()),
        ("Dynamic CNA", lambda: DynamicCNATopology()),
    ]

    rows = []
    series = {name: {"k": [], "chpi": [], "f1": []} for name, _ in methods}

    for name, factory in methods:
        for k in tqdm(K_VALUES, desc=name, leave=False):
            chpi, f1 = run_topology(factory, k, items, args.rounds, args.seed)
            k_label = "inf" if (isinstance(k, float) and math.isinf(k)) else str(k)
            rows.append([name, k_label, f"{chpi:.4f}", f"{f1:.4f}"])
            x = 12 if (isinstance(k, float) and math.isinf(k)) else k  # plot anchor
            series[name]["k"].append(x)
            series[name]["chpi"].append(chpi)
            series[name]["f1"].append(f1)

    header = ["Method", "K_DM", "CHPI ↓", "F1 ↑"]
    write_tex_table(RESULTS_TABLES / "tab_kdm.tex", header, rows,
                    caption="K_DM sweep")
    write_csv(RESULTS_TABLES / "tab_kdm.csv", header, rows)

    # Figure: F1 (left axis) and CHPI (right axis) vs K_DM
    fig, ax1 = plt.subplots(figsize=(5.2, 3.2))
    colors = {"Star": "darkorange", "Dynamic CNA": "seagreen"}
    ax2 = ax1.twinx()
    for name in series:
        c = colors[name]
        ax1.plot(series[name]["k"], series[name]["f1"], "-o", color=c,
                 label=f"{name} F1")
        ax2.plot(series[name]["k"], series[name]["chpi"], "--s", color=c,
                 alpha=0.6, label=f"{name} CHPI")
    ax1.set_xlabel(r"$K_{DM}$ (12 = $\infty$)")
    ax1.set_ylabel("F1 (solid)")
    ax2.set_ylabel("CHPI (dashed)")
    ax1.grid(True, alpha=0.3)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, fontsize=8, loc="best")
    fig.tight_layout()
    out = RESULTS_FIGURES / "fig_kdm.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Wrote {out}")
    print("K_DM sweep done.")


if __name__ == "__main__":
    main()
