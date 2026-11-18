"""Output and bootstrap analysis for direct-rating runs."""

from __future__ import annotations

import json
from pathlib import Path
import random
import warnings

import numpy as np

from pipeline.trust.direct_rating import build_direct_trust
from pipeline.trust.eigentrust import NonUniqueStationaryError
from .plots import save_bootstrap_plot, save_eigenbench_plot
from .train import check_model_order, eigentrust_to_elo


def direct_trust_options(direct_cfg: dict) -> dict:
    """Resolve run-spec options shared by direct point estimates and bootstrap draws."""
    return {
        "include_self": bool(direct_cfg.get("include_self", True)),
        "normalization": direct_cfg.get("normalization", "zscore_softmax"),
        "softmax_temperature": float(direct_cfg.get("softmax_temperature", 1.0)),
        "scale_min": float(direct_cfg.get("scale_min", 1)),
        "scale_max": float(direct_cfg.get("scale_max", 10)),
        "eigentrust_alpha": float(direct_cfg.get("eigentrust_alpha", 0.0)),
    }


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
    # ValueArena's analysis worker patches this function.
    save_bootstrap_plot(summary_rows, path, title="Direct-Rating Bootstrap Elo with 95% Confidence Intervals")


def _save_trust_matrix_plot(matrix: np.ndarray, labels: list[str], path: Path) -> None:
    from matplotlib import pyplot as plt

    size = max(7.0, len(labels) * 0.65)
    fig, ax = plt.subplots(figsize=(size, size))
    image = ax.imshow(matrix, cmap="viridis", vmin=0.0)
    positions = np.arange(len(labels))
    ax.set_xticks(positions)
    ax.set_yticks(positions)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Evaluee")
    ax.set_ylabel("Judge")
    ax.set_title("Direct-Rating Trust Matrix")
    fig.colorbar(image, ax=ax, label="Normalized trust")
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def compute_direct_bootstrap(
    *,
    records: list[dict],
    model_labels: list[str],
    num_criteria: int,
    direct_cfg: dict,
    bootstrap_cfg: dict,
    allow_sparse: bool = False,
    verbose: bool = False,
) -> dict | None:
    n_bootstraps = int(bootstrap_cfg.get("n_bootstraps", 100))
    if n_bootstraps <= 0:
        raise ValueError("bootstrap.n_bootstraps must be positive")
    rng = random.Random(int(bootstrap_cfg.get("random_seed", 42)))
    samples = []
    elo_vectors = []
    trust_options = direct_trust_options(direct_cfg)
    for sample_idx in range(n_bootstraps):
        sampled = _resample_scenarios(records, rng)
        try:
            result = build_direct_trust(
                sampled,
                num_models=len(model_labels),
                num_criteria=num_criteria,
                allow_sparse=allow_sparse,
                verbose=False,
                **trust_options,
            )
        except NonUniqueStationaryError:
            warnings.warn(
                f"Scenario bootstrap draw {sample_idx + 1} has no unique EigenTrust vector. "
                "Bootstrap intervals are unavailable.", RuntimeWarning, stacklevel=2,
            )
            return None
        elo = eigentrust_to_elo(result.eigentrust_scores, len(model_labels), direct_rating=True)
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
    return {"summary": summary, "samples": samples}


def run_direct_bootstrap(*, output_dir: Path, **kwargs) -> dict | None:
    """Save the shared scenario-bootstrap calculation and its existing plot."""
    result = compute_direct_bootstrap(**kwargs)
    if result is None:
        return None
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("samples", "summary"):
        (output_dir / f"{name}.json").write_text(
            json.dumps(result[name], indent=2) + "\n", encoding="utf-8"
        )
    _save_bootstrap_plot(result["summary"], output_dir / "bootstrap_elo.png")
    return {"summary": result["summary"], "output_dir": str(output_dir)}


def run_direct_analysis(
    *,
    records: list[dict],
    models: dict[str, object],
    num_criteria: int,
    evaluation_cfg: dict,
    training_cfg: dict,
    output_root: str | Path,
    collection_cfg: dict | None = None,
    verbose: bool = False,
) -> dict:
    direct_cfg = evaluation_cfg.get("direct_rating", {})
    collection_cfg = collection_cfg or {}
    sampler_mode = str(collection_cfg.get("sampler_mode", "all_to_all")).strip().lower()
    allow_sparse = sampler_mode != "all_to_all" or collection_cfg.get("failure_policy") == "omit_invalid_judgments"
    labels = list(models)
    check_model_order(
        ((record[role]["index"], record[role].get("name"))
         for record in records if record.get("record_type") == "direct_rating"
         for role in ("judge", "evaluee") if isinstance(record.get(role), dict)),
        labels,
    )
    trust_options = direct_trust_options(direct_cfg)
    result = build_direct_trust(
        records,
        num_models=len(labels),
        num_criteria=num_criteria,
        allow_sparse=allow_sparse,
        verbose=verbose,
        **trust_options,
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

    elo = eigentrust_to_elo(result.eigentrust_scores, len(labels), direct_rating=True)
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
        "evaluation_mode": "direct_rating",
        "collection_type": "direct_rating",
        "analysis_kind": "direct_eigentrust",
        "bootstrap_unit": "scenario",
        "num_models": len(labels),
        "model_order": labels,
        "num_criteria": num_criteria,
        "num_scenarios": len(result.scenario_indices),
        "sampler_mode": sampler_mode,
        "group_size": (
            int(collection_cfg.get("group_size", 4)) if allow_sparse else None
        ),
        "response_redundancy": (
            int(collection_cfg.get("response_redundancy", 1)) if allow_sparse else None
        ),
        "sampler_seed": collection_cfg.get("sampler_seed") if allow_sparse else None,
        "total_direct_judgments": len(records),
        "eligible_directed_edges": int(
            len(labels) * (
                len(labels)
                if trust_options["include_self"]
                else max(0, len(labels) - 1)
            )
        ),
        "observed_directed_edges": int(np.sum(result.observation_counts > 0)),
        "include_self": trust_options["include_self"],
        "criterion_aggregation": direct_cfg.get("criterion_aggregation", "mean"),
        "scenario_aggregation": direct_cfg.get("scenario_aggregation", "mean"),
        "normalization": trust_options["normalization"],
        "softmax_temperature": trust_options["softmax_temperature"],
        "eigentrust_alpha": trust_options["eigentrust_alpha"],
        "scale_min": int(direct_cfg.get("scale_min", 1)),
        "scale_max": int(direct_cfg.get("scale_max", 10)),
    }
    log["observed_edge_coverage"] = (
        log["observed_directed_edges"] / log["eligible_directed_edges"]
        if log["eligible_directed_edges"]
        else 0.0
    )
    # Invalidate any earlier draws before running the requested bootstrap.
    log["bootstrap_available"] = False
    config_path = output_dir / "analysis_config.json"
    config_path.write_text(json.dumps(log, indent=2) + "\n", encoding="utf-8")
    save_eigenbench_plot(model_names=labels, eigentrust_elo=elo, save_path=str(output_dir / "eigenbench.png"))
    _save_trust_matrix_plot(result.trust_matrix, labels, output_dir / "trust_matrix.png")

    bootstrap_cfg = training_cfg.get("bootstrap") or {}
    bootstrap = None
    if bool(bootstrap_cfg.get("enabled", False)):
        bootstrap = run_direct_bootstrap(
            records=records,
            model_labels=labels,
            num_criteria=num_criteria,
            direct_cfg=direct_cfg,
            output_dir=output_dir / "bootstrap",
            bootstrap_cfg=bootstrap_cfg,
            allow_sparse=allow_sparse,
            verbose=verbose,
        )
        if bootstrap is not None:
            log["bootstrap_available"] = True
            config_path.write_text(json.dumps(log, indent=2) + "\n", encoding="utf-8")
    print(f"Direct-rating analysis complete. Outputs in {output_dir}")
    return {"summary": summary, "output_dir": str(output_dir), "result": result, "bootstrap": bootstrap}


__all__ = ["run_direct_analysis", "run_direct_bootstrap", "compute_direct_bootstrap"]
