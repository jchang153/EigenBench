"""Generate judge-wise D1-versus-D3 comparison tables and matrices."""

from __future__ import annotations

import csv
import json
import math
import random
import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from statistics import fmean


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) < 2:
        return None
    left_mean = fmean(left)
    right_mean = fmean(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right))
    left_ss = sum((x - left_mean) ** 2 for x in left)
    right_ss = sum((y - right_mean) ** 2 for y in right)
    denominator = math.sqrt(left_ss * right_ss)
    return numerator / denominator if denominator else None


def score_metrics(pairs: list[tuple[int, int]]) -> dict:
    """Summarize matched D1 and D3 integer ratings."""

    if not pairs:
        raise ValueError("score metrics require at least one paired score")
    d1 = [float(item[0]) for item in pairs]
    d3 = [float(item[1]) for item in pairs]
    signed = [right - left for left, right in zip(d1, d3)]
    absolute = [abs(value) for value in signed]
    return {
        "count": len(pairs),
        "d1_mean": fmean(d1),
        "d3_mean": fmean(d3),
        "d3_minus_d1_mean": fmean(signed),
        "mean_absolute_distance": fmean(absolute),
        "root_mean_square_distance": math.sqrt(fmean(value**2 for value in signed)),
        "exact_match_rate": sum(value == 0 for value in absolute) / len(absolute),
        "within_one_rate": sum(value <= 1 for value in absolute) / len(absolute),
        "within_two_rate": sum(value <= 2 for value in absolute) / len(absolute),
        "pearson_correlation": _pearson(d1, d3),
        "absolute_distance_counts": {
            str(key): value for key, value in sorted(Counter(absolute).items())
        },
        "signed_difference_counts": {
            str(key): value for key, value in sorted(Counter(signed).items())
        },
    }


def _bootstrap_scenario_mae(
    pairs_by_scenario: dict[int, list[tuple[int, int]]],
    *,
    seed: int = 42,
    repetitions: int = 10_000,
) -> list[float]:
    scenarios = sorted(pairs_by_scenario)
    rng = random.Random(seed)
    estimates = []
    for _ in range(repetitions):
        sampled = [rng.choice(scenarios) for _ in scenarios]
        distances = [
            abs(d3 - d1)
            for scenario_index in sampled
            for d1, d3 in pairs_by_scenario[scenario_index]
        ]
        estimates.append(fmean(distances))
    estimates.sort()
    low = estimates[int(0.025 * (len(estimates) - 1))]
    high = estimates[int(0.975 * (len(estimates) - 1))]
    return [low, high]


def _d1_reflection_complete(
    text: str, criterion_ids: list[str]
) -> tuple[bool, list[str]]:
    positions: list[tuple[str, int]] = []
    missing = []
    for criterion_id in criterion_ids:
        number = int(criterion_id.rsplit("_", 1)[1])
        match = re.search(
            rf"\bcriterion[\s_#-]*0?{number}\b",
            text,
            flags=re.IGNORECASE,
        )
        if match is None:
            missing.append(criterion_id)
        else:
            positions.append((criterion_id, match.start()))
    if missing:
        return False, missing
    ordered_positions = [position for _, position in positions]
    if ordered_positions != sorted(ordered_positions):
        return False, criterion_ids
    short = []
    for index, (criterion_id, position) in enumerate(positions):
        end = positions[index + 1][1] if index + 1 < len(positions) else len(text)
        if len(text[position:end].strip()) < 20:
            short.append(criterion_id)
    return not short, short


def _matrix_rows(
    judges: list[str],
    evaluees: list[str],
    values: dict[tuple[str, str], float],
) -> list[dict]:
    return [
        {
            "judge": judge,
            **{evaluee: values[(judge, evaluee)] for evaluee in evaluees},
        }
        for judge in judges
    ]


def generate_reports(
    *,
    output_dir: Path,
    cell_records: list[dict],
    stage_records: list[dict],
    raw_calls: list[dict],
    manifest: dict,
) -> dict:
    """Write paired metrics, judge/evaluee summaries, and matrix CSV files."""

    by_cell: dict[tuple[int, str, str], dict[str, dict]] = defaultdict(dict)
    for record in cell_records:
        key = (
            int(record["scenario_index"]),
            str(record["judge_name"]),
            str(record["evaluee_name"]),
        )
        design = str(record["design"])
        if design in by_cell[key]:
            raise ValueError(f"duplicate cell/design record: {key} design={design}")
        by_cell[key][design] = record

    incomplete = {
        key: sorted({"01", "03"} - set(designs))
        for key, designs in by_cell.items()
        if set(designs) != {"01", "03"}
    }
    if incomplete:
        raise ValueError(f"incomplete paired cells: {incomplete}")

    judges = [item["name"] for item in manifest["judges"]]
    evaluees = [item["name"] for item in manifest["evaluees"]]
    criterion_ids = list(manifest["criterion_ids"])
    all_pairs: list[tuple[int, int]] = []
    by_pair: dict[tuple[str, str], list[tuple[int, int]]] = defaultdict(list)
    by_judge: dict[str, list[tuple[int, int]]] = defaultdict(list)
    by_scenario: dict[int, list[tuple[int, int]]] = defaultdict(list)

    for (scenario_index, judge, evaluee), designs in sorted(by_cell.items()):
        d1 = designs["01"]["ratings"]
        d3 = designs["03"]["ratings"]
        for criterion_id in criterion_ids:
            pair = (int(d1[criterion_id]), int(d3[criterion_id]))
            all_pairs.append(pair)
            by_pair[(judge, evaluee)].append(pair)
            by_judge[judge].append(pair)
            by_scenario[scenario_index].append(pair)

    metric_fields = [
        "count",
        "d1_mean",
        "d3_mean",
        "d3_minus_d1_mean",
        "mean_absolute_distance",
        "root_mean_square_distance",
        "exact_match_rate",
        "within_one_rate",
        "within_two_rate",
        "pearson_correlation",
    ]
    pair_rows = [
        {
            "judge": judge,
            "evaluee": evaluee,
            **{
                key: score_metrics(by_pair[(judge, evaluee)])[key]
                for key in metric_fields
            },
        }
        for judge in judges
        for evaluee in evaluees
    ]
    _write_csv(
        output_dir / "pair_summary.csv",
        ["judge", "evaluee", *metric_fields],
        pair_rows,
    )

    judge_rows = [
        {
            "judge": judge,
            **{key: score_metrics(by_judge[judge])[key] for key in metric_fields},
        }
        for judge in judges
    ]
    _write_csv(
        output_dir / "judge_summary.csv",
        ["judge", *metric_fields],
        judge_rows,
    )

    d1_matrix = {
        key: score_metrics(values)["d1_mean"] for key, values in by_pair.items()
    }
    d3_matrix = {
        key: score_metrics(values)["d3_mean"] for key, values in by_pair.items()
    }
    difference_matrix = {
        key: score_metrics(values)["d3_minus_d1_mean"]
        for key, values in by_pair.items()
    }
    mae_matrix = {
        key: score_metrics(values)["mean_absolute_distance"]
        for key, values in by_pair.items()
    }
    for filename, values in (
        ("matrix_d1.csv", d1_matrix),
        ("matrix_d3.csv", d3_matrix),
        ("matrix_difference.csv", difference_matrix),
        ("matrix_mae.csv", mae_matrix),
    ):
        _write_csv(
            output_dir / filename,
            ["judge", *evaluees],
            _matrix_rows(judges, evaluees, values),
        )

    d1_reflections = [
        record
        for record in stage_records
        if record["design"] == "01" and record["stage"] == "reflection"
    ]
    reflection_details = []
    for record in d1_reflections:
        complete, missing_or_short = _d1_reflection_complete(
            str(record["parsed"]["reflection"]),
            criterion_ids,
        )
        reflection_details.append(
            {
                "scenario_index": record["scenario_index"],
                "judge": record["judge_name"],
                "evaluee": record["evaluee_name"],
                "complete": complete,
                "missing_or_short_criterion_ids": missing_or_short,
            }
        )

    token_groups: dict[tuple[str, str, str], dict[str, int]] = defaultdict(
        lambda: {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}
    )
    for record in raw_calls:
        key = (record["judge_name"], record["design"], record["stage"])
        token_groups[key]["calls"] += 1
        token_groups[key]["prompt_tokens"] += int(record.get("prompt_tokens", 0))
        token_groups[key]["completion_tokens"] += int(
            record.get("completion_tokens", 0)
        )
    token_usage = [
        {
            "judge": judge,
            "design": design,
            "stage": stage,
            **values,
            "total_tokens": values["prompt_tokens"] + values["completion_tokens"],
        }
        for (judge, design, stage), values in sorted(token_groups.items())
    ]

    overall = score_metrics(all_pairs)
    overall["scenario_cluster_bootstrap_mae_95_percent_ci"] = _bootstrap_scenario_mae(
        by_scenario
    )
    summary = {
        "run_shape": {
            "scenarios": len(manifest["scenario_indices"]),
            "judges": len(judges),
            "evaluees": len(evaluees),
            "cells_per_design": len(by_cell),
            "criteria": len(criterion_ids),
            "scores_per_design": len(all_pairs),
            "paired_scores": len(all_pairs),
            "rating_cells_across_designs": len(cell_records),
            "planned_calls": len(raw_calls),
        },
        "axes": {
            "judge_rows": judges,
            "evaluee_columns": evaluees,
            "scenario_indices": manifest["scenario_indices"],
            "criterion_ids": criterion_ids,
        },
        "overall_score_distance": overall,
        "judge_summaries": {
            row["judge"]: {key: row[key] for key in metric_fields} for row in judge_rows
        },
        "reflection_completeness": {
            "d1_complete_cells": sum(item["complete"] for item in reflection_details),
            "d1_incomplete_cells": sum(
                not item["complete"] for item in reflection_details
            ),
            "d1_cell_details": reflection_details,
            "d3_complete_cells": len(by_cell),
            "d3_incomplete_cells": 0,
        },
        "local_token_usage": token_usage,
    }
    _write_json(output_dir / "summary.json", summary)
    return summary


__all__ = ["generate_reports", "score_metrics"]
