from scripts.upload_results import build_index_entry


def _meta(dataset: dict) -> dict:
    return {
        "timestamp": "2026-08-26T00:00:00+00:00",
        "dataset": dataset,
        "models": {},
    }


def test_build_index_entry_uses_inclusive_counted_range() -> None:
    entry = build_index_entry(
        "run",
        _meta({"id": "airisk", "start": 0, "count": 200}),
        [],
    )

    assert entry["scenario"] == "airisk [0-199]"


def test_build_index_entry_omits_range_when_count_is_unspecified() -> None:
    entry = build_index_entry(
        "run",
        _meta({"id": "airisk", "start": 50}),
        [],
    )

    assert entry["scenario"] == "airisk"
