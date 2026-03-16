"""Helpers for running local Hugging Face models through vLLM."""

from __future__ import annotations

import gc
import json
from typing import Dict, Optional, Tuple

import torch
from huggingface_hub import hf_hub_download, snapshot_download
from transformers import AutoTokenizer
from vllm import LLM
from vllm.lora.request import LoRARequest


def group_models_for_vllm(
    models: Dict[str, str]
) -> Tuple[Dict[str, Dict[str, object]], Dict[str, AutoTokenizer], Dict[str, str]]:
    """Split the run spec into local HF models and OpenRouter ones."""

    local_base_models: Dict[str, Dict[str, object]] = {}
    local_tokenizers: Dict[str, AutoTokenizer] = {}
    openrouter_models: Dict[str, str] = {}
    lora_repo_cache: Dict[str, str] = {}

    for nick, model_path in models.items():
        if not model_path.startswith("hf_local:"):
            openrouter_models[nick] = model_path
            continue

        hf_path = model_path.split("hf_local:")[1]
        print(f"Inspecting local HF model: {hf_path}")

        try:
            adapter_config_path = hf_hub_download(
                repo_id=hf_path,
                filename="adapter_config.json",
            )
            with open(adapter_config_path, "r") as f:
                adapter_cfg = json.load(f)
            base_model_id = adapter_cfg["base_model_name_or_path"]
            print(f"Detected LoRA adapter. Base model: {base_model_id}")
            is_lora = True
        except Exception:
            base_model_id = hf_path
            is_lora = False

        if base_model_id not in local_base_models:
            local_base_models[base_model_id] = {"loras": {}, "base_only": False}
            local_tokenizers[base_model_id] = AutoTokenizer.from_pretrained(
                base_model_id
            )

        if is_lora:
            if hf_path not in lora_repo_cache:
                print(f"Downloading LoRA weights for {nick} from {hf_path}")
                lora_repo_cache[hf_path] = snapshot_download(repo_id=hf_path)
            local_base_models[base_model_id]["loras"][nick] = lora_repo_cache[hf_path]
        else:
            local_base_models[base_model_id]["base_only"] = nick

    return local_base_models, local_tokenizers, openrouter_models


class VLLMEngineManager:
    """Context manager to spin up and tear down a vLLM engine."""

    def __init__(self, base_model_id: str, enable_lora: bool = False):
        self.base_model_id = base_model_id
        self.enable_lora = enable_lora
        self.llm: Optional[LLM] = None

    def __enter__(self) -> LLM:
        print(f"\n--- Starting vLLM engine for {self.base_model_id} ---")
        self.llm = LLM(
            model=self.base_model_id,
            enable_lora=self.enable_lora,
            max_lora_rank=64 if self.enable_lora else None,
            gpu_memory_utilization=0.9,
            enforce_eager=True,
        )
        return self.llm

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        print(f"\n--- Shutting down vLLM engine for {self.base_model_id} ---")
        del self.llm
        gc.collect()
        torch.cuda.empty_cache()


def prepare_lora_requests(llm: LLM, lora_paths: Dict[str, str]):
    """Load LoRA adapters once per base model and reuse their requests."""

    if not lora_paths:
        return {}

    lora_requests = {}
    for idx, (adapter_name, adapter_path) in enumerate(lora_paths.items(), start=1):
        lora_requests[adapter_name] = LoRARequest(adapter_name, idx, adapter_path)

    try:
        llm.load_lora_adapters(list(lora_requests.values()))
    except AttributeError:
        # Older vLLM versions lazily load adapters on first request.
        pass

    return lora_requests


__all__ = [
    "group_models_for_vllm",
    "prepare_lora_requests",
    "VLLMEngineManager",
]
