"""Check native engine arguments without importing GPU-only dependencies."""
import ast
from pathlib import Path
from typing import Optional
import pytest


@pytest.mark.parametrize('enable_lora', [False, True])
def test_native_engine_lora_rank_ceiling(enable_lora):
    source = Path(__file__).resolve().parents[1]/'pipeline/providers/vllm_local.py'
    tree = ast.parse(source.read_text())
    manager = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'VLLMEngineManager')
    class FakeLLM:
        def __init__(self, **kwargs): self.arguments = kwargs
    namespace = {'LLM': FakeLLM, 'Optional': Optional}
    exec(compile(ast.Module(body=[manager], type_ignores=[]), str(source), 'exec'), namespace)
    engine = namespace['VLLMEngineManager']('test/base', enable_lora=enable_lora, lora_count=3).__enter__()
    assert engine.arguments['model'] == 'test/base'
    if enable_lora:
        assert engine.arguments['max_lora_rank'] == 512
        assert engine.arguments['max_cpu_loras'] == 3
    else:
        assert 'max_lora_rank' not in engine.arguments
