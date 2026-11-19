"""Materialize AIRiskDilemmas as the JSON scenario list EigenBench expects."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys

# Keep the optional materialization CLI and built-in dataset loader identical.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pipeline.config.airisk import (  # noqa: E402, F401
    DATASET_ID,
    DATASET_REVISION,
    load_airisk_scenarios,
    paired_dilemmas,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="data/scenarios/airiskdilemmas.json",
        help="Destination JSON array used by EigenBench run specs",
    )
    args = parser.parse_args()

    scenarios = load_airisk_scenarios()

    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(scenarios, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(scenarios)} scenarios to {output}")


if __name__ == "__main__":
    main()
