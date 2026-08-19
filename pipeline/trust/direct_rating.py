"""Aggregation and normalization for direct criterion-wise ratings."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .eigentrust import eigentrust


SUPPORTED_DIRECT_NORMALIZATIONS = frozenset(
    {
        "zscore_softmax",
        "rank_softmax",
        "raw_l1",
        "minmax_l1",
        "positive_centered_l1",
    }
)


@dataclass(frozen=True)
class DirectTrustResult:
    scenario_indices: tuple[int, ...]
    scenario_values: np.ndarray
    criterion_means: np.ndarray
    raw_means: np.ndarray
    observation_counts: np.ndarray
    intermediate: np.ndarray
    trust_matrix: np.ndarray
    eigentrust_scores: np.ndarray


def _record_index(value, field: str) -> int:
    if not isinstance(value, dict) or not isinstance(value.get("index"), int):
        raise ValueError(f"direct record has invalid {field} object")
    return int(value["index"])


def aggregate_direct_records(
    records: list[dict],
    *,
    num_models: int,
    num_criteria: int,
    include_self: bool = True,
    scale_min: int = 1,
    scale_max: int = 10,
) -> tuple[tuple[int, ...], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return scenarios, scenario values, criterion means, means, counts, mask."""

    if num_models <= 0 or num_criteria <= 0:
        raise ValueError("num_models and num_criteria must be positive")
    direct_records = [record for record in records if record.get("record_type") == "direct_rating"]
    if not direct_records:
        raise ValueError("no direct_rating records found")

    scenario_indices = tuple(sorted({int(record["scenario_index"]) for record in direct_records}))
    scenario_position = {value: idx for idx, value in enumerate(scenario_indices)}
    values = np.full(
        (len(scenario_indices), num_criteria, num_models, num_models),
        np.nan,
        dtype=float,
    )

    for record in direct_records:
        scenario_idx = int(record["scenario_index"])
        judge_idx = _record_index(record.get("judge"), "judge")
        eval_idx = _record_index(record.get("evaluee"), "evaluee")
        if not 0 <= judge_idx < num_models or not 0 <= eval_idx < num_models:
            raise ValueError(
                f"model index out of range in scenario {scenario_idx}: "
                f"judge={judge_idx}, evaluee={eval_idx}"
            )
        if not include_self and judge_idx == eval_idx:
            raise ValueError("self-rating record present while include_self=False")
        rating_rows = record.get("ratings")
        if not isinstance(rating_rows, list):
            raise ValueError(f"ratings must be a list in scenario {scenario_idx}")
        parsed: dict[int, float] = {}
        for rating_row in rating_rows:
            if not isinstance(rating_row, dict):
                raise ValueError(f"invalid rating row in scenario {scenario_idx}")
            criterion_idx = rating_row.get("criterion_index")
            rating = rating_row.get("rating")
            if type(criterion_idx) is not int or type(rating) is not int:
                raise ValueError(f"criterion_index and rating must be integers in scenario {scenario_idx}")
            if not scale_min <= rating <= scale_max:
                raise ValueError(
                    f"rating {rating} is outside [{scale_min}, {scale_max}] in "
                    f"scenario {scenario_idx}"
                )
            if criterion_idx in parsed:
                raise ValueError(
                    f"duplicate criterion {criterion_idx} for scenario={scenario_idx}, "
                    f"judge={judge_idx}, evaluee={eval_idx}"
                )
            parsed[criterion_idx] = float(rating)
        if set(parsed) != set(range(num_criteria)):
            raise ValueError(
                f"incomplete criteria for scenario={scenario_idx}, judge={judge_idx}, "
                f"evaluee={eval_idx}: got {sorted(parsed)}"
            )
        s_pos = scenario_position[scenario_idx]
        for criterion_idx, rating in parsed.items():
            if not np.isnan(values[s_pos, criterion_idx, judge_idx, eval_idx]):
                raise ValueError(
                    f"duplicate direct edge for scenario={scenario_idx}, criterion={criterion_idx}, "
                    f"judge={judge_idx}, evaluee={eval_idx}"
                )
            values[s_pos, criterion_idx, judge_idx, eval_idx] = rating

    mask = np.ones((num_models, num_models), dtype=bool)
    if not include_self:
        np.fill_diagonal(mask, False)
    expected = np.broadcast_to(mask, values.shape)
    if np.isnan(values[expected]).any():
        missing = int(np.isnan(values[expected]).sum())
        raise ValueError(f"direct rating matrix is incomplete: {missing} expected ratings are missing")
    if np.any(~np.isnan(values[~expected])):
        raise ValueError("direct rating matrix contains unexpected masked ratings")

    counts = np.sum(~np.isnan(values), axis=(0, 1)).astype(int)
    criterion_counts = np.sum(~np.isnan(values), axis=0)
    criterion_means = np.full((num_criteria, num_models, num_models), np.nan, dtype=float)
    np.divide(
        np.nansum(values, axis=0),
        criterion_counts,
        out=criterion_means,
        where=criterion_counts > 0,
    )
    criterion_presence = np.sum(~np.isnan(criterion_means), axis=0)
    raw_means = np.full((num_models, num_models), np.nan, dtype=float)
    np.divide(
        np.nansum(criterion_means, axis=0),
        criterion_presence,
        out=raw_means,
        where=criterion_presence > 0,
    )
    return scenario_indices, values, criterion_means, raw_means, counts, mask


def _uniform_over_mask(mask_row: np.ndarray) -> np.ndarray:
    count = int(mask_row.sum())
    if count <= 0:
        raise ValueError("a trust row has no eligible evaluees")
    result = np.zeros(mask_row.shape, dtype=float)
    result[mask_row] = 1.0 / count
    return result


def _softmax(values: np.ndarray, mask: np.ndarray, temperature: float) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("softmax_temperature must be positive")
    result = np.zeros_like(values, dtype=float)
    selected = values[mask] / temperature
    selected = selected - np.max(selected)
    weights = np.exp(selected)
    result[mask] = weights / weights.sum()
    return result


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def normalize_direct_scores(
    scores: np.ndarray,
    *,
    method: str = "zscore_softmax",
    softmax_temperature: float = 1.0,
    scale_min: float = 1.0,
    scale_max: float = 10.0,
    mask: np.ndarray | None = None,
    epsilon: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the normalization intermediate and row-stochastic matrix."""

    score_array = np.asarray(scores, dtype=float)
    if score_array.ndim != 2 or score_array.shape[0] != score_array.shape[1]:
        raise ValueError("direct score matrix must be square")
    if method not in SUPPORTED_DIRECT_NORMALIZATIONS:
        raise ValueError(
            f"unknown direct normalization {method!r}; expected one of "
            f"{sorted(SUPPORTED_DIRECT_NORMALIZATIONS)}"
        )
    if mask is None:
        mask = np.ones_like(score_array, dtype=bool)
    else:
        mask = np.asarray(mask, dtype=bool)
    if mask.shape != score_array.shape:
        raise ValueError("normalization mask shape does not match scores")

    intermediate = np.full_like(score_array, np.nan, dtype=float)
    trust = np.zeros_like(score_array, dtype=float)
    for row_idx in range(score_array.shape[0]):
        row_mask = mask[row_idx]
        row = score_array[row_idx, row_mask]
        if not len(row) or not np.isfinite(row).all():
            raise ValueError(f"score row {row_idx} has no finite eligible ratings")

        if method == "zscore_softmax":
            std = float(np.std(row, ddof=0))
            transformed = np.zeros_like(row) if std < epsilon else (row - np.mean(row)) / std
            intermediate[row_idx, row_mask] = transformed
            trust[row_idx] = _softmax(intermediate[row_idx], row_mask, softmax_temperature)
        elif method == "rank_softmax":
            ranks = _average_ranks(row)
            std = float(np.std(ranks, ddof=0))
            transformed = np.zeros_like(ranks) if std < epsilon else (ranks - np.mean(ranks)) / std
            intermediate[row_idx, row_mask] = transformed
            trust[row_idx] = _softmax(intermediate[row_idx], row_mask, softmax_temperature)
        elif method == "raw_l1":
            transformed = np.maximum(row, 0.0)
            intermediate[row_idx, row_mask] = transformed
            total = transformed.sum()
            trust[row_idx] = _uniform_over_mask(row_mask) if total < epsilon else np.where(
                row_mask,
                np.nan_to_num(intermediate[row_idx], nan=0.0) / total,
                0.0,
            )
        elif method == "minmax_l1":
            span = float(np.max(row) - np.min(row))
            transformed = np.zeros_like(row) if span < epsilon else (row - np.min(row)) / span
            intermediate[row_idx, row_mask] = transformed
            total = transformed.sum()
            trust[row_idx] = _uniform_over_mask(row_mask) if total < epsilon else np.where(
                row_mask,
                np.nan_to_num(intermediate[row_idx], nan=0.0) / total,
                0.0,
            )
        else:
            midpoint = 0.5 * (scale_min + scale_max)
            transformed = np.maximum(row - midpoint, 0.0)
            intermediate[row_idx, row_mask] = transformed
            total = transformed.sum()
            trust[row_idx] = _uniform_over_mask(row_mask) if total < epsilon else np.where(
                row_mask,
                np.nan_to_num(intermediate[row_idx], nan=0.0) / total,
                0.0,
            )

    if not np.isfinite(trust).all() or np.any(trust < 0):
        raise ValueError("normalization produced invalid trust weights")
    if not np.allclose(trust.sum(axis=1), 1.0, atol=1e-10):
        raise ValueError("normalization did not produce row-stochastic output")
    return intermediate, trust


def build_direct_trust(
    records: list[dict],
    *,
    num_models: int,
    num_criteria: int,
    include_self: bool = True,
    normalization: str = "zscore_softmax",
    softmax_temperature: float = 1.0,
    scale_min: float = 1.0,
    scale_max: float = 10.0,
    eigentrust_alpha: float = 0.0,
    verbose: bool = False,
) -> DirectTrustResult:
    scenario_indices, values, criterion_means, raw_means, counts, mask = aggregate_direct_records(
        records,
        num_models=num_models,
        num_criteria=num_criteria,
        include_self=include_self,
        scale_min=int(scale_min),
        scale_max=int(scale_max),
    )
    intermediate, trust = normalize_direct_scores(
        raw_means,
        method=normalization,
        softmax_temperature=softmax_temperature,
        scale_min=scale_min,
        scale_max=scale_max,
        mask=mask,
    )
    trust_scores = eigentrust(
        torch.tensor(trust, dtype=torch.float64),
        alpha=float(eigentrust_alpha),
        verbose=verbose,
    ).detach().cpu().numpy()
    return DirectTrustResult(
        scenario_indices=scenario_indices,
        scenario_values=values,
        criterion_means=criterion_means,
        raw_means=raw_means,
        observation_counts=counts,
        intermediate=intermediate,
        trust_matrix=trust,
        eigentrust_scores=trust_scores,
    )


__all__ = [
    "DirectTrustResult",
    "SUPPORTED_DIRECT_NORMALIZATIONS",
    "aggregate_direct_records",
    "build_direct_trust",
    "normalize_direct_scores",
]
