"""Merge evaluation files and remap indices from model names.

Usage:
    python scripts/run_merge_evaluations.py <spec_module> <output_path> <input1> [input2 ...]
"""

from __future__ import annotations

import json
import sys

from pipeline.config import load_run_spec
from pipeline.io import load_records, save_records


def _canonicalize_record(record: dict) -> str:
    return json.dumps(record, sort_keys=True, ensure_ascii=True)


def _remap_indices(record: dict, model_name_to_index: dict[str, int]) -> dict:
    row = dict(record)

    for index_field, name_field in (
        ("eval1", "eval1_name"),
        ("eval2", "eval2_name"),
        ("judge", "judge_name"),
    ):
        if name_field not in row:
            raise ValueError(f"Missing required field '{name_field}' in record.")
        model_nick = row[name_field]
        if model_nick not in model_name_to_index:
            raise ValueError(
                f"Model name '{model_nick}' is not present in run spec model list."
            )
        row[index_field] = int(model_name_to_index[model_nick])

    return row


def main(spec_ref: str, output_path: str, input_paths: list[str]) -> None:
    spec, _ = load_run_spec(spec_ref)
    model_name_to_index = {name: idx for idx, name in enumerate(spec["models"].keys())}

    merged = []
    seen = set()

    for path in input_paths:
        rows = load_records(path)
        kept = 0
        for row in rows:
            remapped = _remap_indices(row, model_name_to_index)
            key = _canonicalize_record(remapped)
            if key in seen:
                continue
            seen.add(key)
            merged.append(remapped)
            kept += 1
        print(f"Merged {kept} rows from {path}")

    save_records(output_path, merged)
    print(f"Wrote {len(merged)} merged rows -> {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 5:
        raise SystemExit(
            "Usage: python scripts/run_merge_evaluations.py "
            "<spec_module> <output_path> <input1> [input2 ...]"
        )
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
