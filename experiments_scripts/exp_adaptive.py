"""Experiment: Adaptive vs Fixed hyperparameters — addresses the
hyperparameter-rigidity reviewer concern.

Compares Dynamic CNA with the static defaults (alpha=1, beta=2, delta=0.55)
against Dynamic CNA driven by ``AdaptiveScheduler`` under three token-budget
regimes (B_max in {0.5, 1.0, 2.0} × default 1000).

Outputs:
  results/tables/tab_adaptive.csv / .tex
  results/figures/fig_adaptive.pdf
"""
from __future__ import annotations

import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

from ._utils import base_parser, RESULTS_TABLES, RESULTS_FIGURES, write_tex_table, write_csv
from src.simulator import load_dataset, sample_items, inject_hallucination
from src.agents import make_agents, State
from src.orchestrator import Orchestrator
from src.topologies.dynamic_cna import DynamicCNATopology
from src.scheduler import AdaptiveScheduler


BUDGETS = [0.5, 1.0, 2.0]
BASE_BUDGET = 1000.0


def run(mode, budget_scale, items, rounds, seed):
    agents = make_agents(seed=seed)
    topo = DynamicCNATopology()
    scheduler = None
    if mode == "adaptive":
        scheduler = AdaptiveScheduler(
            alpha0=1.0, beta0=2.0, delta0=0.55,
            token_budget=BASE_BUDGET * budget_scale,
        )
    orch = Orchestrator(topo, agents=agents, alpha=1.0, beta=2.0, scheduler=scheduler)

    chpi_list, f1_list, tok_list = [], [], []
    for item in items:
        state = State(item)
        inject_hallucination(state, seed=seed)
        result = orch.run(item, rounds=rounds)
        chpi_list.append(result["chpi"])
        f1_list.append(result["f1"])
        tok_list.append(result["total_routed_tokens"])
    return (statistics.mean(chpi_list),
            statistics.mean(f1_list),
            statistics.mean(tok_list))


def main():
    p = base_parser("Adaptive vs Fixed scheduler (reviewer concern)")
    args = p.parse_args()

    dataset = load_dataset("cityemergency", seed=args.seed)
    items = sample_items(dataset, args.n_crises, seed=args.seed)

    rows = []
    chart = {"Fixed": [], "Adaptive": []}
    labels = []
    for scale in BUDGETS:
        labels.append(f"{scale:g}x")
        for mode in ("fixed", "adaptive"):
            chpi, f1, tok = run(mode, scale, items, args.rounds, args.seed)
            rows.append([
                mode.capitalize(), f"{scale:g}x", f"{tok:.0f}",
                f"{chpi:.4f}", f"{f1:.4f}",
            ])
            chart[mode.capitalize()].append(f1)

    header = ["Mode", "Budget", "Tokens", "CHPI ↓", "F1 ↑"]
    write_tex_table(RESULTS_TABLES / "tab_adaptive.tex", header, rows,
                    caption="Adaptive vs fixed hyperparameters")
    write_csv(RESULTS_TABLES / "tab_adaptive.csv", header, rows)

    # Bar plot: F1 per budget regime
    import numpy as np
    x = np.arange(len(BUDGETS))
    w = 0.35
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    ax.bar(x - w / 2, chart["Fixed"], w, label="Fixed", color="steelblue")
    ax.bar(x + w / 2, chart["Adaptive"], w, label="Adaptive", color="seagreen")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Token-budget regime")
    ax.set_ylabel("Decision F1")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out = RESULTS_FIGURES / "fig_adaptive.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Wrote {out}")
    print("Adaptive experiment done.")


if __name__ == "__main__":
    main()
