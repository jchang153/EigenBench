"""max_model_len for local models: the default, or the model's own window when shorter."""
from types import SimpleNamespace

import pytest

from pipeline.providers import model_window
from pipeline.providers.model_window import DEFAULT_MAX_MODEL_LEN, resolve_max_model_len


@pytest.fixture
def configs(monkeypatch):
    transformers = pytest.importorskip("transformers")
    known = {}
    def from_pretrained(model_id, revision=None):
        config = known[(model_id, revision)]
        if isinstance(config, Exception):
            raise config
        return config
    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", from_pretrained)
    resolve_max_model_len.cache_clear()
    yield known
    resolve_max_model_len.cache_clear()


def test_shorter_window_wins_and_float_windows_are_read(configs):
    configs[("allenai/OLMo-2-1124-7B", None)] = SimpleNamespace(max_position_embeddings=4096.0)
    assert resolve_max_model_len("allenai/OLMo-2-1124-7B") == 4096


def test_mpt_window_caps_native_and_inspect_models(configs):
    from transformers import MptConfig
    from inspect_pipeline.model_mapping import to_inspect_model

    model_id = "mosaicml/mpt-7b-instruct"
    configs[(model_id, None)] = MptConfig(max_seq_len=2048)
    assert resolve_max_model_len(model_id) == 2048
    assert to_inspect_model(f"hf_local:{model_id}").model_args["max_model_len"] == 2048


def test_longer_windows_keep_the_default(configs):
    configs[("Qwen/Qwen2.5-7B-Instruct", "abc")] = SimpleNamespace(max_position_embeddings=32768)
    assert resolve_max_model_len("Qwen/Qwen2.5-7B-Instruct", "abc") == DEFAULT_MAX_MODEL_LEN
    assert resolve_max_model_len("Qwen/Qwen2.5-7B-Instruct", "abc", 65536) == 32768


def test_multimodal_models_are_read_from_their_text_config(configs):
    text = SimpleNamespace(max_position_embeddings=2048)
    configs[("org/vision-language", None)] = SimpleNamespace(get_text_config=lambda: text)
    assert resolve_max_model_len("org/vision-language") == 2048


def test_unreadable_configs_keep_the_default(configs):
    configs[("org/gated", None)] = OSError("gated repo")
    configs[("org/no-window", None)] = SimpleNamespace(max_position_embeddings=True)
    assert resolve_max_model_len("org/gated") == DEFAULT_MAX_MODEL_LEN
    assert resolve_max_model_len("org/no-window") == DEFAULT_MAX_MODEL_LEN
