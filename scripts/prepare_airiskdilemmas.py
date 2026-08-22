"""Materialize AIRiskDilemmas as the JSON scenario list EigenBench expects."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DATASET_ID = "kellycyy/AIRiskDilemmas"
DATASET_REVISION = "8674d1f5844c3909b05e06d9f30bbc2b7c753f39"


def paired_dilemmas(rows) -> list[str]:
    """Collapse each consecutive pair of action rows into one scenario."""

    iterator = iter(rows)
    scenarios: list[str] = []
    while True:
        try:
            first = next(iterator)
        except StopIteration:
            break
        try:
            second = next(iterator)
        except StopIteration as exc:
            raise ValueError("AIRiskDilemmas contains an unpaired final action row") from exc
        first_value = first.get("dilemma") if isinstance(first, dict) else None
        second_value = second.get("dilemma") if isinstance(second, dict) else None
        if not isinstance(first_value, str) or not first_value.strip():
            raise ValueError("AIRiskDilemmas contains an empty or non-string dilemma")
        if first_value != second_value:
            raise ValueError("consecutive AIRiskDilemmas action rows have different dilemmas")
        scenarios.append(first_value)
    return scenarios


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="data/scenarios/airiskdilemmas.json",
        help="Destination JSON array used by EigenBench run specs",
    )
    args = parser.parse_args()

    from huggingface_hub import hf_hub_download

    source_path = hf_hub_download(
        repo_id=DATASET_ID,
        filename="model_eval.jsonl",
        repo_type="dataset",
        revision=DATASET_REVISION,
    )
    with Path(source_path).open("r", encoding="utf-8") as handle:
        scenarios = paired_dilemmas(json.loads(line) for line in handle if line.strip())
    if len(scenarios) < 120:
        raise RuntimeError(
            f"Expected at least 120 unique AIRiskDilemmas scenarios; found {len(scenarios)}"
        )

    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(scenarios, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(scenarios)} scenarios to {output}")


if __name__ == "__main__":
    main()
