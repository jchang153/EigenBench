"""Output and bootstrap analysis for direct-rating runs."""

from __future__ import annotations

import json
from pathlib import Path
import random

import numpy as np

from pipeline.trust.direct_rating import build_direct_trust
from .plots import save_eigenbench_plot
from .train import eigentrust_to_elo


def _save_matrix(path: Path, matrix: np.ndarray) -> None:
    np.savetxt(path, matrix, delimiter=",", fmt="%.10g")


def _resample_scenarios(records: list[dict], rng: random.Random) -> list[dict]:
    grouped: dict[int, list[dict]] = {}
    for record in records:
        grouped.setdefault(int(record["scenario_index"]), []).append(record)
    scenario_indices = sorted(grouped)
    sampled_records: list[dict] = []
    for draw_index in range(len(scenario_indices)):
        selected = scenario_indices[rng.randrange(len(scenario_indices))]
        for record in grouped[selected]:
            copied = dict(record)
            copied["scenario_index"] = draw_index
            sampled_records.append(copied)
    return sampled_records


def _save_bootstrap_plot(summary_rows: list[dict], path: Path) -> None:
    from matplotlib import pyplot as plt

    labels = [row["model_name"] for row in summary_rows]
    means = np.array([row["elo_mean"] for row in summary_rows])
    lower = np.array([row["elo_ci_lower"] for row in summary_rows])
    upper = np.array([row["elo_ci_upper"] for row in summary_rows])
    x = np.arange(len(summary_rows))
    fig, ax = plt.subplots(figsize=(max(10, len(summary_rows) * 0.45), 6))
    ax.errorbar(x, means, yerr=np.vstack([means - lower, upper - means]), fmt="o", capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("EigenBench Elo")
    ax.set_title("Direct-Rating Bootstrap Elo with 95% Confidence Intervals")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def run_direct_bootstrap(
    *,
    records: list[dict],
    model_labels: list[str],
    num_criteria: int,
    direct_cfg: dict,
    output_dir: Path,
    bootstrap_cfg: dict,
    verbose: bool = False,
) -> dict:
    n_bootstraps = int(bootstrap_cfg.get("n_bootstraps", 100))
    if n_bootstraps <= 0:
        raise ValueError("bootstrap.n_bootstraps must be positive")
    rng = random.Random(int(bootstrap_cfg.get("random_seed", 42)))
    output_dir.mkdir(parents=True, exist_ok=True)
    samples = []
    elo_vectors = []
    for sample_idx in range(n_bootstraps):
        sampled = _resample_scenarios(records, rng)
        result = build_direct_trust(
            sampled,
            num_models=len(model_labels),
            num_criteria=num_criteria,
            include_self=bool(direct_cfg.get("include_self", True)),
            normalization=direct_cfg.get("normalization", "zscore_softmax"),
            softmax_temperature=float(direct_cfg.get("softmax_temperature", 1.0)),
            scale_min=float(direct_cfg.get("scale_min", 1)),
            scale_max=float(direct_cfg.get("scale_max", 10)),
            eigentrust_alpha=float(direct_cfg.get("eigentrust_alpha", 0.0)),
            verbose=False,
        )
        elo = eigentrust_to_elo(result.eigentrust_scores, len(model_labels))
        record = {
            "sample_idx": sample_idx,
            "trust_vector": result.eigentrust_scores.tolist(),
            "elo_vector": elo.tolist(),
        }
        if bool(bootstrap_cfg.get("save_trust_matrices", True)):
            record["trust_matrix"] = result.trust_matrix.tolist()
        samples.append(record)
        elo_vectors.append(elo)
        if verbose and (sample_idx + 1) % 10 == 0:
            print(f"  Direct bootstrap sample {sample_idx + 1}/{n_bootstraps}")

    (output_dir / "samples.json").write_text(json.dumps(samples, indent=2) + "\n", encoding="utf-8")
    elo_array = np.asarray(elo_vectors, dtype=float)
    ddof = 1 if n_bootstraps > 1 else 0
    summary = []
    for idx, label in enumerate(model_labels):
        summary.append(
            {
                "model_index": idx,
                "model_name": label,
                "elo_mean": float(np.mean(elo_array[:, idx])),
                "elo_std": float(np.std(elo_array[:, idx], ddof=ddof)),
                "elo_ci_lower": float(np.percentile(elo_array[:, idx], 2.5)),
                "elo_ci_upper": float(np.percentile(elo_array[:, idx], 97.5)),
            }
        )
    summary.sort(key=lambda row: row["elo_mean"], reverse=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    _save_bootstrap_plot(summary, output_dir / "bootstrap_elo.png")
    return {"summary": summary, "output_dir": str(output_dir)}


def run_direct_analysis(
    *,
    records: list[dict],
    models: dict[str, object],
    num_criteria: int,
    evaluation_cfg: dict,
    training_cfg: dict,
    output_root: str | Path,
    verbose: bool = False,
) -> dict:
    direct_cfg = evaluation_cfg.get("direct_rating", {})
    labels = list(models)
    result = build_direct_trust(
        records,
        num_models=len(labels),
        num_criteria=num_criteria,
        include_self=bool(direct_cfg.get("include_self", True)),
        normalization=direct_cfg.get("normalization", "zscore_softmax"),
        softmax_temperature=float(direct_cfg.get("softmax_temperature", 1.0)),
        scale_min=float(direct_cfg.get("scale_min", 1)),
        scale_max=float(direct_cfg.get("scale_max", 10)),
        eigentrust_alpha=float(direct_cfg.get("eigentrust_alpha", 0.0)),
        verbose=verbose,
    )
    output_dir = Path(output_root) / "direct_rating"
    output_dir.mkdir(parents=True, exist_ok=True)
    _save_matrix(output_dir / "raw_mean_scores.csv", result.raw_means)
    _save_matrix(output_dir / "normalization_intermediate.csv", result.intermediate)
    _save_matrix(output_dir / "trust_matrix.csv", result.trust_matrix)
    _save_matrix(output_dir / "observation_counts.csv", result.observation_counts)
    criteria_dir = output_dir / "criteria"
    criteria_dir.mkdir(exist_ok=True)
    for criterion_idx, matrix in enumerate(result.criterion_means, start=1):
        _save_matrix(criteria_dir / f"criterion_{criterion_idx}_mean_scores.csv", matrix)

    elo = eigentrust_to_elo(result.eigentrust_scores, len(labels))
    summary = [
        {
            "model_index": idx,
            "model_name": label,
            "eigentrust": float(result.eigentrust_scores[idx]),
            "eigenbench_elo": float(elo[idx]),
        }
        for idx, label in enumerate(labels)
    ]
    summary.sort(key=lambda row: row["eigentrust"], reverse=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (output_dir / "eigentrust.txt").write_text(
        "EigenTrust scores:\n" + np.array2string(result.eigentrust_scores, separator=", ") + "\n",
        encoding="utf-8",
    )
    log = {
        "num_models": len(labels),
        "model_order": labels,
        "num_criteria": num_criteria,
        "num_scenarios": len(result.scenario_indices),
        "include_self": bool(direct_cfg.get("include_self", True)),
        "criterion_aggregation": direct_cfg.get("criterion_aggregation", "mean"),
        "scenario_aggregation": direct_cfg.get("scenario_aggregation", "mean"),
        "normalization": direct_cfg.get("normalization", "zscore_softmax"),
        "softmax_temperature": float(direct_cfg.get("softmax_temperature", 1.0)),
        "eigentrust_alpha": float(direct_cfg.get("eigentrust_alpha", 0.0)),
        "scale_min": int(direct_cfg.get("scale_min", 1)),
        "scale_max": int(direct_cfg.get("scale_max", 10)),
    }
    (output_dir / "analysis_config.json").write_text(json.dumps(log, indent=2) + "\n", encoding="utf-8")
    save_eigenbench_plot(model_names=labels, eigentrust_elo=elo, save_path=str(output_dir / "eigenbench.png"))

    bootstrap_cfg = training_cfg.get("bootstrap") or {}
    if bool(bootstrap_cfg.get("enabled", False)):
        run_direct_bootstrap(
            records=records,
            model_labels=labels,
            num_criteria=num_criteria,
            direct_cfg=direct_cfg,
            output_dir=output_dir / "bootstrap",
            bootstrap_cfg=bootstrap_cfg,
            verbose=verbose,
        )
    print(f"Direct-rating analysis complete. Outputs in {output_dir}")
    return {"summary": summary, "output_dir": str(output_dir), "result": result}


__all__ = ["run_direct_analysis", "run_direct_bootstrap"]
