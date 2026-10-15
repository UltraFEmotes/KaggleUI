"""Dispatch a deployment spec to the engine that runs it."""

from __future__ import annotations

from ..schemas import DeploymentSpec
from .base import Backend
from .context import DeployError, EngineContext

WORKER_ENGINES = {"diffusers", "diffusers_video", "diffusers_world", "kokoro", "chatterbox", "transformers_tts", "transformers_audio", "faster_whisper",
                  "transformers_asr", "transformers"}


async def deploy(spec: DeploymentSpec, ctx: EngineContext, *, mock: bool) -> tuple[Backend, dict]:
    if mock or spec.model.engine == "mock":
        from . import mock as mock_engine

        return await mock_engine.deploy(spec, ctx)
    engine = spec.model.engine
    if engine == "ollama":
        from . import ollama

        return await ollama.deploy(spec, ctx)
    if engine == "vllm":
        from . import vllm

        return await vllm.deploy(spec, ctx)
    if engine == "sglang":
        from . import sglang

        return await sglang.deploy(spec, ctx)
    if engine == "world_recipe":
        from . import world_engine

        return await world_engine.deploy(spec, ctx)
    if engine in WORKER_ENGINES:
        from . import workers_engine

        return await workers_engine.deploy(spec, ctx)
    raise DeployError(f"Unknown engine '{engine}'")


async def undeploy(spec: DeploymentSpec, dep_id: str, *, mock: bool) -> None:
    if not mock and spec.model.engine == "ollama":
        from . import ollama

        await ollama.undeploy(dep_id)
