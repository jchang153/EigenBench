"""Focused regressions for this change."""
from inspect_pipeline.model_mapping import to_inspect_model

def test_adapter_snapshot_download_is_limited_to_selected_subfolder(tmp_path, monkeypatch):
    import huggingface_hub
    calls = []
    def download(**kwargs):
        calls.append(kwargs)
        return str(tmp_path)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    ref = to_inspect_model({
        "provider": "hf_local", "kind": "lora", "repo_id": "org/adapters",
        "subfolder": "humor/checkpoint", "revision": "pinned-commit",
        "base_model_id": "org/base",
    })
    assert calls == [{"repo_id": "org/adapters", "revision": "pinned-commit", "allow_patterns": ["humor/checkpoint/*"]}]
    assert ref.name == f"vllm/org/base:{tmp_path}/humor/checkpoint"
