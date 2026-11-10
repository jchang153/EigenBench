"""Regressions for the #41 audit flags fixed in #47 (offline; Hub calls are simulated)."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture
def mapping(monkeypatch):
    import inspect_pipeline.model_mapping as mm

    monkeypatch.setattr(mm, "resolve_max_model_len", lambda *args, **kwargs: 8192)
    mm._root_adapter_base.cache_clear()
    yield mm
    mm._root_adapter_base.cache_clear()


def test_catalogue_update_never_writes_a_stale_cached_index(tmp_path, monkeypatch):
    """F23: after a timed-out Hub read the client can return a stale cached index.json."""
    import huggingface_hub
    from scripts import upload_results

    stale = tmp_path / "index.json"
    stale.write_text(json.dumps({"last_updated": None, "runs": [{"slug": "old-a", "timestamp": "1"}]}))

    def download(repo_id, filename, repo_type=None, force_download=False, **kwargs):
        if force_download:
            raise TimeoutError("read timed out")
        return str(stale)  # what a cached read returns when the fresh check fails

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    written = []
    api = SimpleNamespace(upload_file=lambda path_or_fileobj, **kwargs: written.append(path_or_fileobj))
    with pytest.raises(TimeoutError):
        upload_results.update_index("new", "new", {}, [], "org/dataset", api)
    assert written == []


def test_catalogue_starts_only_when_the_index_is_missing(tmp_path, monkeypatch):
    import huggingface_hub
    from huggingface_hub.utils import EntryNotFoundError
    from scripts import upload_results

    def missing(repo_id, filename, **kwargs):
        raise EntryNotFoundError("index.json not found")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", missing)
    assert upload_results._load_index("org/dataset") == {"last_updated": None, "runs": []}


def test_staging_reads_only_the_configured_output_dir(tmp_path):
    """F22: an older run/btd_d8 must not stand in for training.output_dir results."""
    from scripts.upload_results import find_btd_dir

    (tmp_path / "btd_d8").mkdir()
    configured = {"training": {"dims": [8], "output_dir": "fresh"}}
    assert find_btd_dir(tmp_path, configured) is None
    (tmp_path / "fresh" / "btd_d8").mkdir(parents=True)
    assert find_btd_dir(tmp_path, configured) == tmp_path / "fresh" / "btd_d8"
    assert find_btd_dir(tmp_path, {"training": {"dims": [8]}}) == tmp_path / "btd_d8"


def test_spec_and_cli_models_on_one_base_at_two_revisions_are_refused(mapping):
    """F16: a --model added to the panel is an Inspect Model, not a spec reference."""
    from inspect_ai.model import get_model

    spec_member = {"provider": "hf_local", "repo_id": "org/base", "kind": "base", "revision": "a" * 40}
    cli_member = get_model("vllm/org/base", revision="b" * 40, memoize=False)
    with pytest.raises(ValueError, match="share a base model at different revisions"):
        mapping.check_distinct_local_servers({"spec": spec_member, "cli": cli_member})
    same = get_model("vllm/org/base", revision="a" * 40, memoize=False)
    mapping.check_distinct_local_servers({"spec": spec_member, "cli": same})


def test_vllm_revision_guard_uses_effective_provider_args(mapping, monkeypatch):
    from inspect_ai.model import get_model

    monkeypatch.setenv("VLLM_DEFAULT_SERVER_ARGS", json.dumps({"revision": "commit-a"}))
    spec_member = {"provider": "hf_local", "repo_id": "org/base", "kind": "base"}
    same = get_model("vllm/org/base", revision="commit-a", memoize=False)
    mapping.check_distinct_local_servers({"spec": spec_member, "cli": same})

    unpinned = get_model("vllm/org/base", revision=None, memoize=False)
    with pytest.raises(ValueError, match="share a base model at different revisions"):
        mapping.check_distinct_local_servers({"spec": spec_member, "cli": unpinned})


@pytest.mark.parametrize("revision", ["commit-a", None])
def test_vllm_spec_ignores_models_memoized_under_previous_defaults(mapping, monkeypatch, revision):
    from inspect_ai.model import get_model
    from inspect_pipeline.eigenbench import _model_resolver

    spec = {"provider": "hf_local", "repo_id": "org/memoized-base", "kind": "base"}
    monkeypatch.setenv("VLLM_DEFAULT_SERVER_ARGS", json.dumps({"revision": "commit-b"}))
    ref = mapping.to_inspect_model(spec)
    old = get_model(ref.name, **ref.model_args)
    assert old.api.server_args["revision"] == "commit-b"

    if revision is None:
        monkeypatch.delenv("VLLM_DEFAULT_SERVER_ARGS")
        with pytest.raises(RuntimeError, match="different vLLM revision"):
            _model_resolver({"spec": spec})("spec")
        fresh = _model_resolver({"spec": spec}, memoize=False)("spec")
        assert "revision" not in fresh.api.server_args
    else:
        monkeypatch.setenv("VLLM_DEFAULT_SERVER_ARGS", json.dumps({"revision": revision}))
        model = _model_resolver({"spec": spec})("spec")
        assert model is not old
        assert model.api.server_args["revision"] == revision
    assert mapping.model_identity(spec)["args"].get("revision") == revision


def test_inconclusive_adapter_lookup_is_retried_not_cached(tmp_path, monkeypatch, mapping):
    """F15: a timed-out lookup must not be remembered as a full model."""
    import huggingface_hub

    config = tmp_path / "adapter_config.json"
    config.write_text(json.dumps({"base_model_name_or_path": "org/base"}))
    calls = []

    def download(repo_id, filename, revision=None, **kwargs):
        calls.append(repo_id)
        if len(calls) == 1:
            raise TimeoutError("read timed out")
        return str(config)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    with pytest.raises(RuntimeError, match="declare kind='base' or kind='lora'"):
        mapping.to_inspect_model("hf_local:org/adapter")
    assert mapping.to_inspect_model("hf_local:org/adapter").name == "vllm/org/base:org/adapter"
    assert len(calls) == 2


def test_confirmed_full_model_and_declared_kind(monkeypatch, mapping):
    import huggingface_hub
    from huggingface_hub.utils import EntryNotFoundError

    def no_adapter(repo_id, filename, revision=None, **kwargs):
        raise EntryNotFoundError(f"{filename} not in {repo_id}")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", no_adapter)
    assert mapping.to_inspect_model("hf_local:org/full-model").name == "vllm/org/full-model"
    declared = {"provider": "hf_local", "repo_id": "org/adapter", "kind": "base"}
    assert mapping.to_inspect_model(declared).name == "vllm/org/adapter"
