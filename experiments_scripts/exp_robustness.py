"""Experiment: Structural Robustness (Table 2).

Metrics: CHPI, IDR, F1 for Mesh, Star, Dynamic CNA.
"""
from __future__ import annotations

import statistics
from tqdm import tqdm

from ._utils import base_parser, RESULTS_TABLES, write_tex_table, write_csv
from src.simulator import load_dataset, sample_items, inject_hallucination
from src.agents import make_agents, State
from src.orchestrator import Orchestrator
from src.topologies.mesh import MeshTopology
from src.topologies.star import StarTopology
from src.topologies.dynamic_cna import DynamicCNATopology


def run_topology(topology, items, rounds, seed):
    agents = make_agents(seed=seed)
    orch = Orchestrator(topology, agents=agents)
    chpi_list, idr_list, f1_list = [], [], []
    for item in tqdm(items, desc=topology.name, leave=False):
        state = State(item)
        inject_hallucination(state, seed=seed)
        result = orch.run(item, rounds=rounds)
        chpi_list.append(result["chpi"])
        idr_list.append(result["idr"])
        f1_list.append(result["f1"])
    return (
        statistics.mean(chpi_list),
        statistics.mean(idr_list),
        statistics.mean(f1_list),
    )


def main():
    p = base_parser("Robustness experiment — Table 2")
    args = p.parse_args()

    dataset = load_dataset("crisitext", seed=args.seed)
    items = sample_items(dataset, args.n_crises, seed=args.seed)

    topologies = [MeshTopology(), StarTopology(), DynamicCNATopology()]
    results = {}
    for topo in topologies:
        chpi, idr, f1 = run_topology(topo, items, args.rounds, args.seed)
        results[topo.name] = (chpi, idr, f1)

    header = ["Topology", "CHPI ↓", "IDR ↓", "F1 ↑"]
    rows = []
    for name, (chpi, idr, f1) in results.items():
        rows.append([name, f"{chpi:.4f}", f"{idr:.4f}", f"{f1:.4f}"])

    write_tex_table(RESULTS_TABLES / "tab_robustness.tex", header, rows,
                    caption="Table 2: Structural Robustness")
    write_csv(RESULTS_TABLES / "tab_robustness.csv", header, rows)
    print("Robustness experiment done.")


if __name__ == "__main__":
    main()
