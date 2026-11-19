"""Bootstrap intervals for pairwise rankings."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .bt_models import CriteriaVectorBTD
from .eigentrust import compute_trust_matrix_ties, eigentrust
from .fit import (
    CriteriaComparisons,
    eigentrust_to_elo,
    train_vector_bt,
)
from .plots import save_bootstrap_plot


def _build_model_and_loader(
    sampled_comparisons: list,
    model_kind: str,
    num_criteria: int,
    num_models: int,
    dim: int,
    batch_size: int,
):
    if model_kind != "btd_ties":
        raise ValueError(f"Unsupported model kind: {model_kind!r}; EigenBench rankings use 'btd_ties'")

    model = CriteriaVectorBTD(num_criteria, num_models, dim)
    loader = DataLoader(
        CriteriaComparisons(sampled_comparisons),
        batch_size=batch_size,
        shuffle=True,
    )
    return model, loader


def _train_one_sample(
    sampled_comparisons: list,
    model_kind: str,
    num_criteria: int,
    num_models: int,
    dim: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    max_epochs: int,
    device: str,
):
    model, loader = _build_model_and_loader(
        sampled_comparisons, model_kind, num_criteria, num_models, dim, batch_size,
    )
    train_vector_bt(
        model=model,
        dataloader=loader,
        lr=lr,
        weight_decay=weight_decay,
        max_epochs=max_epochs,
        device=device,
        save_path=None,
        normalize=False,
        use_btd=True,
        criterion_mode=True,
        verbose=False,
        plot=False,
    )

    trust_matrix = compute_trust_matrix_ties(model, device=device)
    trust_vector = eigentrust(trust_matrix, alpha=0, verbose=False)
    return (
        trust_matrix.detach().cpu().numpy(),
        trust_vector.detach().cpu().numpy(),
        model,
    )


def run_bootstrap(
    comparisons: list,
    num_models: int,
    num_criteria: int,
    model_kind: str,
    dim: int,
    model_labels: list[str],
    output_dir: str | None,
    *,
    n_bootstraps: int = 100,
    random_seed: int = 42,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    max_epochs: int = 1000,
    device: str = "cpu",
    save_models: bool = False,
    save_trust_matrices: bool = True,
    verbose: bool = False,
) -> dict:
    """Run bootstrap resampling and save results.

    Returns a summary dict with per-model Elo means, stds, and 95% CIs.
    """
    out = Path(output_dir) if output_dir is not None else None
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
    models_dir = out / "models" if out is not None else None
    if save_models and models_dir is not None:
        models_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(random_seed)
    bootstrap_records: list[dict] = []
    elo_vectors: list[np.ndarray] = []

    n = len(comparisons)
    for sample_idx in range(n_bootstraps):
        sampled = [comparisons[rng.randrange(n)] for _ in range(n)]
        trust_matrix, trust_vector, model = _train_one_sample(
            sampled,
            model_kind=model_kind,
            num_criteria=num_criteria,
            num_models=num_models,
            dim=dim,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            max_epochs=max_epochs,
            device=device,
        )
        elo_vector = eigentrust_to_elo(trust_vector, num_models)

        record: dict = {
            "sample_idx": sample_idx,
            "trust_vector": trust_vector.tolist(),
            "elo_vector": elo_vector.tolist(),
        }
        if save_trust_matrices:
            record["trust_matrix"] = trust_matrix.tolist()

        bootstrap_records.append(record)
        elo_vectors.append(elo_vector)

        if save_models and models_dir is not None:
            torch.save(model.state_dict(), models_dir / f"model_{sample_idx:04d}.pt")

        if verbose and (sample_idx + 1) % 10 == 0:
            print(f"  Bootstrap sample {sample_idx + 1}/{n_bootstraps}")

    # Save raw samples
    if out is not None:
        samples_path = out / "samples.json"
        with samples_path.open("w", encoding="utf-8") as f:
            json.dump(bootstrap_records, f, indent=2)

    # Compute summary statistics
    elo_arr = np.asarray(elo_vectors, dtype=float)
    elo_means = np.mean(elo_arr, axis=0)
    elo_std = np.std(elo_arr, axis=0, ddof=1)
    elo_lower = np.percentile(elo_arr, 2.5, axis=0)
    elo_upper = np.percentile(elo_arr, 97.5, axis=0)

    summary_rows = []
    for idx, label in enumerate(model_labels):
        summary_rows.append({
            "model_index": idx,
            "model_name": label,
            "elo_mean": float(elo_means[idx]),
            "elo_std": float(elo_std[idx]),
            "elo_ci_lower": float(elo_lower[idx]),
            "elo_ci_upper": float(elo_upper[idx]),
        })
    summary_rows.sort(key=lambda r: r["elo_mean"], reverse=True)

    if out is not None:
        summary_path = out / "summary.json"
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary_rows, f, indent=2)
        _save_bootstrap_plot(summary_rows, out / "bootstrap_elo.png")

    if verbose:
        for row in summary_rows:
            print(
                f"  [{row['model_index']:>2}] {row['model_name']}: "
                f"mean={row['elo_mean']:.2f}, std={row['elo_std']:.2f}, "
                f"95% CI=[{row['elo_ci_lower']:.2f}, {row['elo_ci_upper']:.2f}]"
            )

    return {"summary": summary_rows, "samples": bootstrap_records,
            "output_dir": str(out) if out is not None else None}


def _save_bootstrap_plot(summary_rows: list[dict], save_path: Path) -> None:
    save_bootstrap_plot(summary_rows, save_path, title="Bootstrap Elo Means with 95% Confidence Intervals")
