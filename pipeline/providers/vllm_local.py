"""Helpers for running local Hugging Face models through vLLM."""

from __future__ import annotations

import gc
import json
import os
import signal
import time
from typing import Dict, Optional, Tuple

import torch
import torch.distributed as dist
from huggingface_hub import hf_hub_download, snapshot_download
from transformers import AutoTokenizer
from vllm import LLM
from vllm.lora.request import LoRARequest

from pipeline.model_refs import is_hf_local_model, parse_hf_local_model


def group_models_for_vllm(
    models: Dict[str, object]
) -> Tuple[Dict[str, Dict[str, object]], Dict[str, AutoTokenizer], Dict[str, str]]:
    """Split the run spec into local HF models and OpenRouter ones."""

    local_base_models: Dict[str, Dict[str, object]] = {}
    local_tokenizers: Dict[str, AutoTokenizer] = {}
    openrouter_models: Dict[str, str] = {}
    snapshot_cache: Dict[tuple[str, Optional[str]], str] = {}

    for nick, model_ref in models.items():
        if not is_hf_local_model(model_ref):
            if not isinstance(model_ref, str):
                raise ValueError(
                    f"OpenRouter model {nick!r} must be a model-ID string; got {type(model_ref).__name__}"
                )
            openrouter_models[nick] = model_ref
            continue

        local_ref = parse_hf_local_model(model_ref)
        repo_id = local_ref.repo_id
        subfolder = local_ref.subfolder
        adapter_revision = local_ref.revision
        revision_label = f"@{adapter_revision}" if adapter_revision else ""
        location = f"{repo_id}{revision_label}"
        if subfolder:
            location = f"{location}/{subfolder}"
        print(f"Inspecting local HF model: {location}")

        try:
            if local_ref.kind == "base":
                raise FileNotFoundError("Structured reference explicitly selects a base model")
            if subfolder:
                # Subfolder-based repo: download entire repo once, resolve locally
                adapter_cache_key = (repo_id, adapter_revision)
                if adapter_cache_key not in snapshot_cache:
                    print(f"Downloading repo {repo_id}{revision_label} (all subfolders)...")
                    snapshot_cache[adapter_cache_key] = snapshot_download(
                        repo_id=repo_id,
                        revision=adapter_revision,
                    )
                local_repo_dir = snapshot_cache[adapter_cache_key]
                adapter_config_path = os.path.join(local_repo_dir, subfolder, "adapter_config.json")
            else:
                # Standard single-adapter repo
                adapter_config_path = hf_hub_download(
                    repo_id=repo_id,
                    filename="adapter_config.json",
                    revision=adapter_revision,
                )

            with open(adapter_config_path, "r") as f:
                adapter_cfg = json.load(f)
            metadata_base_model_id = adapter_cfg["base_model_name_or_path"]
            base_model_id = local_ref.base_model_id or metadata_base_model_id
            base_revision = local_ref.base_revision
            if local_ref.base_model_id and local_ref.base_model_id != metadata_base_model_id:
                print(
                    "Detected LoRA adapter. Overriding adapter metadata base "
                    f"{metadata_base_model_id!r} with {base_model_id!r}."
                )
            else:
                print(f"Detected LoRA adapter. Base model: {base_model_id}")
            is_lora = True
        except Exception as e:
            if local_ref.expects_lora:
                raise RuntimeError(
                    f"Failed to load LoRA adapter from {location}: {e}"
                ) from e
            base_model_id = repo_id
            base_revision = adapter_revision
            is_lora = False

        base_key = base_model_id if base_revision is None else f"{base_model_id}@{base_revision}"
        if base_key not in local_base_models:
            base_model_path = base_model_id
            if base_revision is not None:
                base_cache_key = (base_model_id, base_revision)
                if base_cache_key not in snapshot_cache:
                    print(f"Downloading pinned base model {base_key}...")
                    snapshot_cache[base_cache_key] = snapshot_download(
                        repo_id=base_model_id,
                        revision=base_revision,
                    )
                base_model_path = snapshot_cache[base_cache_key]

            local_base_models[base_key] = {
                "base_model_id": base_model_id,
                "base_revision": base_revision,
                "base_model_path": base_model_path,
                "loras": {},
                "base_only": False,
            }
            tokenizer = AutoTokenizer.from_pretrained(base_model_path)
            # Fix special token attributes that may be non-string types
            for attr in ('eos_token', 'bos_token', 'unk_token', 'pad_token',
                         'sep_token', 'cls_token', 'mask_token'):
                val = getattr(tokenizer, attr, None)
                if val is not None and not isinstance(val, str):
                    if isinstance(val, (tuple, list)):
                        setattr(tokenizer, attr, val[0] if val else "")
                    elif hasattr(val, 'content'):  # AddedToken
                        setattr(tokenizer, attr, str(val))
            local_tokenizers[base_key] = tokenizer

        if is_lora:
            if subfolder:
                # Repo already downloaded above; just point to the subfolder
                lora_local_path = os.path.join(snapshot_cache[(repo_id, adapter_revision)], subfolder)
            else:
                adapter_cache_key = (repo_id, adapter_revision)
                if adapter_cache_key not in snapshot_cache:
                    print(f"Downloading LoRA weights for {nick} from {repo_id}{revision_label}")
                    snapshot_cache[adapter_cache_key] = snapshot_download(
                        repo_id=repo_id,
                        revision=adapter_revision,
                    )
                lora_local_path = snapshot_cache[adapter_cache_key]
            local_base_models[base_key]["loras"][nick] = lora_local_path
        else:
            local_base_models[base_key]["base_only"] = nick

    return local_base_models, local_tokenizers, openrouter_models


class VLLMEngineManager:
    """Context manager to spin up and tear down a vLLM engine."""

    def __init__(
        self,
        base_model_id: str,
        enable_lora: bool = False,
        lora_count: int = 0,
    ):
        self.base_model_id = base_model_id
        self.enable_lora = enable_lora
        self.lora_count = int(lora_count)
        self.llm: Optional[LLM] = None

    def __enter__(self) -> LLM:
        print(f"\n--- Starting vLLM engine for {self.base_model_id} ---")
        engine_args = {
            "model": self.base_model_id,
            "gpu_memory_utilization": 0.9,
            "enforce_eager": True,
            "max_model_len": 8192,
        }
        if self.enable_lora:
            engine_args.update({
                "enable_lora": True,
                "max_lora_rank": 64,
                "max_loras": 1,
                "max_cpu_loras": max(1, self.lora_count),
            })
        self.llm = LLM(
            **engine_args,
        )
        return self.llm

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        print(f"\n--- Shutting down vLLM engine for {self.base_model_id} ---")
        if self.llm is not None:
            # Shutdown the engine explicitly if available
            if hasattr(self.llm, 'llm_engine') and self.llm.llm_engine is not None:
                try:
                    self.llm.llm_engine.shutdown()
                except Exception:
                    pass
            # Kill ALL lingering child processes that may hold GPU memory
            try:
                import multiprocessing
                for child in multiprocessing.active_children():
                    child.kill()
                    child.join(timeout=10)
            except Exception:
                pass
            del self.llm
            self.llm = None
        # Destroy any leftover distributed process groups
        try:
            if dist.is_initialized():
                dist.destroy_process_group()
        except Exception:
            pass
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        # Give GPU time to fully release memory
        time.sleep(5)
        free_mem = torch.cuda.mem_get_info()[0] / (1024**3)
        total_mem = torch.cuda.mem_get_info()[1] / (1024**3)
        print(f"GPU memory after cleanup: {free_mem:.1f}/{total_mem:.1f} GiB free")


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
