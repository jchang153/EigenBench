"""Provider routing, vLLM revisions, and adapter detection."""

import json

import pytest


@pytest.fixture
def mapping(monkeypatch):
    import inspect_pipeline.model_mapping as mm

    monkeypatch.setattr(mm, "resolve_max_model_len", lambda *args, **kwargs: 8192)
    mm._root_adapter_base.cache_clear()
    yield mm
    mm._root_adapter_base.cache_clear()


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
