#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Installing dependencies ==="
pip install -r requirements.txt

echo ""
echo "=== Generating synthetic data ==="
python data/synthetic_fallback.py

echo ""
echo "=== Attempting real dataset downloads (failures are non-fatal) ==="
python data/download_crisitext.py || true
python data/download_cityemergency.py || true

echo ""
echo "=== Running experiment scripts ==="
python -m experiments_scripts.exp_efficiency
python -m experiments_scripts.exp_robustness
python -m experiments_scripts.exp_ablation
python -m experiments_scripts.exp_dynamics
python -m experiments_scripts.exp_chpi_heatmap
python -m experiments_scripts.exp_comparison

echo ""
echo "=== Summary of generated artifacts ==="
echo "Tables:"
ls -lh results/tables/ 2>/dev/null || echo "  (none)"
echo "Figures:"
ls -lh results/figures/ 2>/dev/null || echo "  (none)"
echo ""
echo "Done."
