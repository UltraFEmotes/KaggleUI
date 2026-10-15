"""Engine dispatch: the registry routes each spec to its runner, and GPU/TPU refusals fire before any install."""

from __future__ import annotations

import asyncio

import pytest

from kaggleui.engines import registry
from kaggleui.engines.context import DeployError, EngineContext
from kaggleui.schemas import DeploymentSpec


def _ctx(paths, hw) -> EngineContext:
    return EngineContext(paths=paths, dep_id="t", served_name="m", gpus=[0], hw=hw,
                         log=lambda _line: None, progress=lambda *_args: None, internal_key="k", main_port=1)


def _spec(engine: str) -> DeploymentSpec:
    return DeploymentSpec(modality="text", model={"source": "hf", "id": "Qwen/Qwen3-0.6B", "engine": engine})


def test_sglang_refuses_tpu_before_installing(paths, hw):
    with pytest.raises(DeployError, match="TPU"):
        asyncio.run(registry.deploy(_spec("sglang"), _ctx(paths, hw("tpu")), mock=False))


def test_sglang_refuses_pre_t4_gpus(paths, hw):
    with pytest.raises(DeployError, match="T4"):
        asyncio.run(registry.deploy(_spec("sglang"), _ctx(paths, hw("p100")), mock=False))


def test_vllm_refuses_pre_t4_gpus(paths, hw):
    with pytest.raises(DeployError, match="T4"):
        asyncio.run(registry.deploy(_spec("vllm"), _ctx(paths, hw("p100")), mock=False))
