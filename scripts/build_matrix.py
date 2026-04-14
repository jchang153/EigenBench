#!/usr/bin/env python3
"""Build the character-train matrix from per-constitution bootstrap results.

Reads btd_d{dim}/bootstrap/summary.json from each constitution sub-run,
computes A[i,j] = Elo(model_Cj, eval_Ci) - Elo(base, eval_Ci), and saves
the matrix as a heatmap PNG + CSV.

Works for both LoRA-based (matrix) and prompted runs.

Usage:
    python scripts/build_matrix.py runs/prompted/
    python scripts/build_matrix.py runs/matrix_new/ --nick-prefix "Qwen2.5-7B-Instruct_"
    python scripts/build_matrix.py runs/prompted/ --nick-prefix "prompted_"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

# Allow running from repo root
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


CONSTITUTIONS = [
    "goodness",
    "humor",
    "impulsiveness",
    "loving",
    "mathematical",
    "misalignment",
    "nonchalance",
    "poeticism",
    "remorse",
    "sarcasm",
    "sycophancy",
]

BASE_NICK = "base"
REF_NICKS = ["gpt-4o", "claude-4-sonnet", "gemini-2.5-pro"]
REF_ANCHOR = 1500
DIM = 2


def load_bootstrap_summary(run_dir: Path, constitution: str, dim: int = DIM) -> dict | None:
    summary_path = run_dir / constitution / f"btd_d{dim}" / "bootstrap" / "summary.json"
    if not summary_path.exists():
        return None
    with open(summary_path) as f:
        data = json.load(f)
    # summary.json is a list of {model_name, elo_mean, elo_std, ...}
    # Convert to dict keyed by model_name
    return {entry["model_name"]: entry for entry in data}


def find_model_nick(summary: dict, constitution: str, nick_prefix: str) -> str | None:
    """Find the model nick for a constitution in the bootstrap summary."""
    target = f"{nick_prefix}{constitution}"
    if target in summary:
        return target
    # Fallback: search for partial match
    for nick in summary:
        if constitution in nick.lower():
            return nick
    return None


def build_matrix(
    runs_dir: Path,
    nick_prefix: str | None = None,
    dim: int = DIM,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Build the Elo difference matrix from bootstrap summaries.

    Returns (A_elo_mean, A_elo_std, constitutions_found).
    """
    constitutions = []
    summaries = {}

    for c in CONSTITUTIONS:
        bs = load_bootstrap_summary(runs_dir, c, dim)
        if bs is None:
            print(f"  {c}: no bootstrap summary — skipping")
            continue
        constitutions.append(c)
        summaries[c] = bs

    # Auto-detect nick prefix if not provided
    if nick_prefix is None:
        for c, bs in summaries.items():
            nicks = [n for n in bs if n != BASE_NICK and c in n.lower()]
            if nicks:
                nick = nicks[0]
                nick_prefix = nick[:nick.lower().index(c)]
                break
        if nick_prefix is None:
            print("Could not auto-detect nick prefix. Use --nick-prefix.")
            sys.exit(1)
    print(f"Nick prefix: '{nick_prefix}'")

    N = len(constitutions)
    if N == 0:
        print("No valid runs found.")
        sys.exit(1)

    # Columns: constitutions + base
    col_labels = constitutions + [BASE_NICK]
    M = len(col_labels)
    A_mean = np.full((N, M), np.nan)
    A_std = np.full((N, M), np.nan)

    for i, ci in enumerate(constitutions):
        bs = summaries[ci]
        # Reference = average Elo of API models present in this run
        ref_elos = [bs[r]["elo_mean"] for r in REF_NICKS if r in bs]
        if not ref_elos:
            if BASE_NICK in bs:
                ref_elos = [bs[BASE_NICK]["elo_mean"]]
            else:
                print(f"  {ci}: no reference models — skipping row")
                continue
        ref_mean = sum(ref_elos) / len(ref_elos)
        offset = REF_ANCHOR - ref_mean

        for j, cj in enumerate(constitutions):
            nick = find_model_nick(bs, cj, nick_prefix)
            if nick and nick in bs:
                A_mean[i, j] = bs[nick]["elo_mean"] + offset
                A_std[i, j] = bs[nick]["elo_std"]

        # Base column
        if BASE_NICK in bs:
            A_mean[i, N] = bs[BASE_NICK]["elo_mean"] + offset
            A_std[i, N] = bs[BASE_NICK]["elo_std"]

    return A_mean, A_std, constitutions, col_labels


def plot_matrix(
    A_mean: np.ndarray,
    A_std: np.ndarray,
    row_labels: list[str],
    output_path: Path,
    col_labels: list[str] | None = None,
    title: str = "Character-Train Matrix (Elo vs Base)",
):
    """Plot the matrix as a heatmap and save to PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if col_labels is None:
        col_labels = row_labels
    nrows, ncols = A_mean.shape
    dev = np.nanmax(np.abs(A_mean - REF_ANCHOR))
    if dev == 0:
        dev = 1

    fig, ax = plt.subplots(figsize=(max(10, ncols * 0.9), max(8, nrows * 0.75)))
    im = ax.imshow(A_mean, cmap="RdBu_r", vmin=REF_ANCHOR - dev, vmax=REF_ANCHOR + dev, aspect="auto")

    ax.set_xticks(range(ncols))
    ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(nrows))
    ax.set_yticklabels(row_labels, fontsize=8)
    ax.set_xlabel("Trained on (column)", fontsize=10)
    ax.set_ylabel("Evaluated under (row)", fontsize=10)
    ax.set_title(title, fontsize=12, pad=12)

    # Annotate cells
    for i in range(nrows):
        for j in range(ncols):
            val = A_mean[i, j]
            std = A_std[i, j]
            if not np.isnan(val):
                color = "white" if abs(val - REF_ANCHOR) > dev * 0.6 else "black"
                label = f"{val:.0f}\n±{std:.0f}" if not np.isnan(std) else f"{val:.0f}"
                ax.text(j, i, label, ha="center", va="center", fontsize=6.5, color=color)

    plt.colorbar(im, ax=ax, label=f"Elo (anchored: API avg = {REF_ANCHOR})", shrink=0.8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {output_path}")


def save_csv(A_mean: np.ndarray, row_labels: list[str], output_path: Path, col_labels: list[str] | None = None):
    """Save the matrix as CSV."""
    import csv
    if col_labels is None:
        col_labels = row_labels
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([""] + col_labels)
        for i, c in enumerate(row_labels):
            row = [c] + [f"{A_mean[i,j]:.1f}" if not np.isnan(A_mean[i,j]) else "" for j in range(A_mean.shape[1])]
            writer.writerow(row)
    print(f"Saved CSV: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Build character-train matrix from EigenBench runs")
    parser.add_argument("runs_dir", help="Directory containing per-constitution sub-runs")
    parser.add_argument("--nick-prefix", default=None,
                        help="Model nick prefix (e.g., 'prompted_' or 'Qwen2.5-7B-Instruct_'). "
                             "Auto-detected if not specified.")
    parser.add_argument("--dim", type=int, default=DIM, help=f"BTD dimensionality (default: {DIM})")
    parser.add_argument("--title", default=None, help="Plot title")
    parser.add_argument("--output", default=None, help="Output prefix (default: <runs_dir>/character_train_matrix)")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir).resolve()
    if not runs_dir.exists():
        print(f"Error: {runs_dir} does not exist")
        sys.exit(1)

    A_mean, A_std, constitutions, col_labels = build_matrix(runs_dir, args.nick_prefix, args.dim)

    output_prefix = args.output or str(runs_dir / "character_train_matrix")
    title = args.title or f"Character-Train Matrix — {runs_dir.name} (Elo, API avg = {REF_ANCHOR})"

    plot_matrix(A_mean, A_std, constitutions, Path(f"{output_prefix}.png"), col_labels=col_labels, title=title)
    save_csv(A_mean, constitutions, Path(f"{output_prefix}.csv"), col_labels=col_labels)

    # Print diagonal analysis (square portion only)
    N = len(constitutions)
    print(f"\n=== Diagonal Analysis (self-alignment) ===")
    for i, c in enumerate(constitutions):
        diag = A_mean[i, i]
        if not np.isnan(diag):
            print(f"  {c}: {diag:.0f} Elo")
    diag_vals = np.array([A_mean[i, i] for i in range(N) if not np.isnan(A_mean[i, i])])
    off_diag = A_mean[:, :N][~np.eye(N, dtype=bool)]
    off_diag = off_diag[~np.isnan(off_diag)]
    if len(diag_vals):
        print(f"\n  Mean diagonal: {np.mean(diag_vals):.0f} Elo")
    if len(off_diag):
        print(f"  Mean off-diagonal: {np.mean(off_diag):.0f} Elo")
    # Base column
    base_col = A_mean[:, N]
    base_vals = base_col[~np.isnan(base_col)]
    if len(base_vals):
        print(f"  Mean base: {np.mean(base_vals):.0f} Elo")


if __name__ == "__main__":
    main()
