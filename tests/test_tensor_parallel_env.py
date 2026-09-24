"""Exercise the actual engine constructor without importing GPU dependencies."""
import ast
import os
from pathlib import Path
from typing import Optional
import pytest


def manager():
    path=Path(__file__).parents[1]/'pipeline/providers/vllm_local.py'
    tree=ast.parse(path.read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='VLLMEngineManager')
    scope={'os':os,'Optional':Optional,'LLM':lambda **kwargs:kwargs}
    exec(compile(ast.Module(body=[cls],type_ignores=[]),str(path),'exec'),scope)
    return scope['VLLMEngineManager']


@pytest.mark.parametrize('count',[1,2,4,8])
def test_requested_gpu_count_reaches_vllm(monkeypatch,count):
    monkeypatch.setenv('EIGENBENCH_TENSOR_PARALLEL_SIZE',str(count))
    assert manager()('base').__enter__()['tensor_parallel_size']==count


def test_default_and_invalid_values(monkeypatch):
    monkeypatch.delenv('EIGENBENCH_TENSOR_PARALLEL_SIZE',raising=False)
    assert manager()('base').__enter__()['tensor_parallel_size']==1
    for value in ['0','-1','invalid']:
        monkeypatch.setenv('EIGENBENCH_TENSOR_PARALLEL_SIZE',value)
        with pytest.raises(ValueError):manager()('base')
