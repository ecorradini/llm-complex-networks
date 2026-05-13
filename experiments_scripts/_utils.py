"""Shared utilities for experiment scripts."""
from __future__ import annotations

import argparse
import pathlib
import sys

# Ensure src is on path when running as module
_EXPERIMENTS_DIR = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(_EXPERIMENTS_DIR))

RESULTS_TABLES = _EXPERIMENTS_DIR / "results" / "tables"
RESULTS_FIGURES = _EXPERIMENTS_DIR / "results" / "figures"
RESULTS_TABLES.mkdir(parents=True, exist_ok=True)
RESULTS_FIGURES.mkdir(parents=True, exist_ok=True)


def base_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--n-crises", type=int, default=20, help="Number of crisis scenarios")
    p.add_argument("--rounds", type=int, default=5, help="Communication rounds per crisis")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    return p


def write_tex_table(path: pathlib.Path, header: list, rows: list, caption: str = "") -> None:
    """Write a booktabs-formatted .tex table (no \\begin{table} wrapper)."""
    col_fmt = "l" + "r" * (len(header) - 1)
    lines = [
        f"% {caption}",
        f"\\begin{{tabular}}{{{col_fmt}}}",
        "\\toprule",
        " & ".join(f"\\textbf{{{h}}}" for h in header) + " \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(str(v) for v in row) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    path.write_text("\n".join(lines) + "\n")
    print(f"  Wrote {path}")


def write_csv(path: pathlib.Path, header: list, rows: list) -> None:
    import csv
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"  Wrote {path}")
