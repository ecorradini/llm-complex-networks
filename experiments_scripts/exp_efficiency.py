"""Experiment: Computational Efficiency (Table 1).

Metrics: tokens/round and latency for Mesh, Star, Dynamic CNA.
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
    tokens_list, latency_list = [], []
    for item in tqdm(items, desc=topology.name, leave=False):
        state = State(item)
        inject_hallucination(state, seed=seed)
        result = orch.run(item, rounds=rounds)
        logs = result["logs"]
        tokens_per_round = [l["routed_tokens"] for l in logs]
        latency_per_round = [l["latency"] for l in logs]
        tokens_list.append(statistics.median(tokens_per_round))
        latency_list.append(statistics.mean(latency_per_round))
    return (
        statistics.mean(tokens_list),
        statistics.mean(latency_list),
    )


def main():
    p = base_parser("Efficiency experiment — Table 1")
    args = p.parse_args()

    dataset = load_dataset("crisitext", seed=args.seed)
    items = sample_items(dataset, args.n_crises, seed=args.seed)

    topologies = [MeshTopology(), StarTopology(), DynamicCNATopology()]
    results = {}
    for topo in topologies:
        tokens, latency = run_topology(topo, items, args.rounds, args.seed)
        results[topo.name] = (tokens, latency)

    # Compute reductions vs Mesh
    mesh_tokens = results["Mesh"][0]
    header = ["Topology", "Tokens/Round (median)", "Reduction vs Mesh", "Latency/Round (s)"]
    rows = []
    for name, (tok, lat) in results.items():
        reduction = f"{100*(mesh_tokens - tok)/mesh_tokens:.1f}\\%" if mesh_tokens > 0 else "--"
        rows.append([name, f"{tok:.0f}", reduction, f"{lat:.3f}"])

    write_tex_table(RESULTS_TABLES / "tab_efficiency.tex", header, rows,
                    caption="Table 1: Computational Efficiency")
    write_csv(RESULTS_TABLES / "tab_efficiency.csv", header,
              [[r[0], r[1], r[2].replace("\\%", "%"), r[3]] for r in rows])
    print("Efficiency experiment done.")


if __name__ == "__main__":
    main()
