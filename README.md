# Dynamic Topology Optimization — Experiments

Pipeline for the paper **"Dynamic Topology Optimization in LLM Multi-Agent Systems for Urban Crisis Management: A Complex Network Approach"**.

## Quick Start

```bash
cd experiments
bash run_all.sh
```

Requires Python 3.9+. All heavy dependencies (`sentence-transformers`, `datasets`) are optional — the pipeline degrades gracefully to deterministic fallbacks.

## Outputs

| File | Description |
|------|-------------|
| `results/tables/tab_efficiency.tex` | Table 1 — token & latency comparison |
| `results/tables/tab_robustness.tex` | Table 2 — CHPI / IDR / F1 comparison |
| `results/tables/tab_comparison.tex` | Table 3 — all six topology methods |
| `results/figures/fig_ablation.pdf` | Figure 1 — ablation on α, β, δ |
| `results/figures/fig_dynamics.pdf` | Figure 2 — Q and H over rounds |
| `results/figures/fig_chpi_heatmap.pdf` | Figure 3 — CHPI heatmap per topology |

CSV mirrors of every table are also written alongside the `.tex` files.

## License

Experimental code — research use only.
