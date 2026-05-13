"""Experiment: Live-LLM sanity check — addresses the evaluation-constraints
reviewer concern.

Runs Mesh, Star, and Dynamic CNA on a small sample of CityEmergency-QA-150
with a real generative backend (gpt-4.1-nano by default). The purpose is
NOT to produce headline numbers — the curated-variant experiment in the
main tables already controls model stochasticity. The purpose is to show
that the **ordering** of the topologies is preserved when agents emit free
text.

Activated by:
    export OPENAI_API_KEY=...
    USE_LIVE_LLM=1 python -m experiments_scripts.exp_live_llm --n-crises 10

Outputs:
  results/tables/tab_live_llm.csv / .tex
"""
from __future__ import annotations

import os
import statistics

from tqdm import tqdm

from ._utils import base_parser, RESULTS_TABLES, write_tex_table, write_csv
from src.simulator import load_dataset, sample_items, inject_hallucination
from src.agents import Agent, ROLE_KEYWORDS, State
from src.orchestrator import Orchestrator
from src.topologies.mesh import MeshTopology
from src.topologies.star import StarTopology
from src.topologies.dynamic_cna import DynamicCNATopology
from src.llm_backend import OpenAIBackend, MockLLM


def make_live_agents(seed: int = 42):
    backend = OpenAIBackend()
    return [Agent(role, llm=backend, seed=seed) for role in ROLE_KEYWORDS]


def run_topology(topology, items, rounds, seed, live):
    if live:
        agents = make_live_agents(seed=seed)
    else:
        agents = [Agent(role, llm=MockLLM(seed=seed), seed=seed) for role in ROLE_KEYWORDS]
    orch = Orchestrator(topology, agents=agents)
    chpi_list, f1_list, tok_list = [], [], []
    for item in tqdm(items, desc=topology.name, leave=False):
        state = State(item)
        inject_hallucination(state, seed=seed)
        result = orch.run(item, rounds=rounds)
        chpi_list.append(result["chpi"])
        f1_list.append(result["f1"])
        tok_list.append(result["total_routed_tokens"])
    return (statistics.mean(tok_list),
            statistics.mean(chpi_list),
            statistics.mean(f1_list))


def main():
    p = base_parser("Live-LLM sanity check")
    p.set_defaults(n_crises=10, rounds=3)
    args = p.parse_args()

    if os.environ.get("USE_LIVE_LLM") != "1":
        print("USE_LIVE_LLM != 1 — falling back to deterministic backend.")
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY not set; aborting.")

    dataset = load_dataset("cityemergency", seed=args.seed)
    items = sample_items(dataset, args.n_crises, seed=args.seed)

    rows = []
    for name, topo in [
        ("Mesh", MeshTopology()),
        ("Star", StarTopology()),
        ("Dynamic CNA", DynamicCNATopology()),
    ]:
        print(f"Running {name} (live={os.environ.get('USE_LIVE_LLM')})...")
        tok, chpi, f1 = run_topology(topo, items, args.rounds, args.seed, live=True)
        rows.append([name, f"{tok:.0f}", f"{chpi:.4f}", f"{f1:.4f}"])

    header = ["Topology", "Tokens", "CHPI ↓", "F1 ↑"]
    write_tex_table(RESULTS_TABLES / "tab_live_llm.tex", header, rows,
                    caption="Live-LLM sanity check (gpt-4.1-nano)")
    write_csv(RESULTS_TABLES / "tab_live_llm.csv", header, rows)
    print("Live-LLM experiment done.")


if __name__ == "__main__":
    main()
