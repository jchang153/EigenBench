"""Retired collection command. Use the Inspect-backed runner."""


def main(*_args, **_kwargs):
    raise SystemExit(
        "This legacy collector has been retired. "
        "Use: python scripts/run.py <spec_module_or_path> (Inspect engine). "
        "collection.cached_responses_path is read and completed by Inspect."
    )


if __name__ == "__main__":
    main()
