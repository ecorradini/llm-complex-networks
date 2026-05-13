"""Experiment: Comparison of all six topology methods (Table 3).

Topologies: Mesh, Star, DyTopo, GTD-lite, GoAgent-lite, Dynamic CNA.
Columns: routing mechanism, training cost, interpretability,
         token overhead (median tokens/round), CHPI, F1.
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
from src.topologies.dytopo import DyTopoTopology
from src.topologies.gtd_lite import GTDLiteTopology
from src.topologies.goagent_lite import GoAgentLiteTopology
from src.topologies.dynamic_cna import DynamicCNATopology


# Static metadata for the table (qualitative columns)
METADATA = {
    "Mesh":         ("Full broadcast",         "None",    "Low"),
    "Star":         ("Hub supervisor",          "None",    "Medium"),
    "DyTopo":       ("Semantic query/key",      "LLM infer.", "Medium"),
    "GTD-lite":     ("Proxy-scored sampling",   "None",    "Low"),
    "GoAgent-lite": ("Group clustering",        "K-means", "Medium"),
    "Dynamic CNA":  ("Louvain + entropy prune", "None",    "High"),
}


def run_topology(topology, items, rounds, seed):
    agents = make_agents(seed=seed)
    orch = Orchestrator(topology, agents=agents)
    tokens_list, chpi_list, f1_list = [], [], []
    for item in tqdm(items, desc=topology.name, leave=False):
        state = State(item)
        inject_hallucination(state, seed=seed)
        result = orch.run(item, rounds=rounds)
        logs = result["logs"]
        tokens_list.append(statistics.median([l["routed_tokens"] for l in logs]))
        chpi_list.append(result["chpi"])
        f1_list.append(result["f1"])
    return (
        statistics.mean(tokens_list),
        statistics.mean(chpi_list),
        statistics.mean(f1_list),
    )


def main():
    p = base_parser("Comparison experiment — Table 3")
    args = p.parse_args()

    dataset = load_dataset("cityemergency", seed=args.seed)
    items = sample_items(dataset, args.n_crises, seed=args.seed)

    topologies = [
        MeshTopology(),
        StarTopology(),
        DyTopoTopology(tau=0.4),
        GTDLiteTopology(lam=0.1, seed=args.seed),
        GoAgentLiteTopology(n_groups=3, seed=args.seed),
        DynamicCNATopology(delta=0.35),
    ]

    header = [
        "Topology", "Routing Mechanism", "Training Cost",
        "Interpretability", "Tokens/Round", "CHPI ↓", "F1 ↑"
    ]
    rows = []
    for topo in topologies:
        print(f"Running {topo.name}...")
        tokens, chpi, f1 = run_topology(topo, items, args.rounds, args.seed)
        meta = METADATA.get(topo.name, ("--", "--", "--"))
        rows.append([
            topo.name,
            meta[0],
            meta[1],
            meta[2],
            f"{tokens:.0f}",
            f"{chpi:.4f}",
            f"{f1:.4f}",
        ])

    write_tex_table(RESULTS_TABLES / "tab_comparison.tex", header, rows,
                    caption="Table 3: Comparison of all topology methods")
    write_csv(RESULTS_TABLES / "tab_comparison.csv", header, rows)
    print("Comparison experiment done.")


if __name__ == "__main__":
    main()
