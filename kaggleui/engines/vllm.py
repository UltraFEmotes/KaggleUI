"""vLLM engine (isolated venv; tensor-parallel across both T4s, or across all chips of a Kaggle TPU)."""

from __future__ import annotations

import asyncio

import httpx

from ..capabilities import context_length, local_profile
from ..catalog import hf
from ..proc import ManagedProcess
from ..schemas import DeploymentSpec
from ..util import free_port
from .base import OpenAIUpstream
from .context import DeployError, EngineContext
from .download import hf_download, repo_files
from .install import ensure_venv
from .model_support import model_config


def tpu_parallel(chips: int, config: dict) -> int:
    """Largest chip count (halving from all of them) that evenly divides the attention heads.

    vLLM rejects a tensor-parallel size that doesn't divide them (e.g. 28 heads across 8 chips).
    """
    text = config.get("text_config")
    heads = (text if isinstance(text, dict) else config).get("num_attention_heads")
    tp = max(chips, 1)
    while tp > 1 and isinstance(heads, int) and heads % tp:
        tp //= 2
    return tp


async def deploy(spec: DeploymentSpec, ctx: EngineContext) -> tuple[OpenAIUpstream, dict]:
    tpu = ctx.hw.accelerator == "tpu" and bool(ctx.hw.tpus)
    if tpu:
        if spec.modality != "text":
            raise DeployError("On a TPU, KaggleUI serves text models only. "
                              "Switch the accelerator to GPU T4 ×2 for this one.")
    elif (ctx.hw.min_sm or 0) < 70:
        raise DeployError("vLLM needs a T4 or newer GPU. Use the Ollama or Transformers engine instead.")
    if spec.model.source == "attached":
        path = spec.model.id
    else:
        revision = spec.options.get("revision")
        sel = hf.transformers_files(await repo_files(spec.model.id, revision=revision))
        path = await hf_download(ctx, spec.model.id, sel["files"], sel["size"], revision=revision)
    if tpu:
        # vllm-tpu bundles tpu-inference (the TPU backend) for a matching torch; keep it isolated like vLLM.
        py = await ensure_venv(ctx, "vllm-tpu", ["vllm-tpu"], python="3.11", label="vLLM TPU")
        tp = tpu_parallel(len(ctx.hw.tpus), model_config(path))
        if tp < len(ctx.hw.tpus):
            ctx.log(f"Sharding across {tp} of {len(ctx.hw.tpus)} TPU chips (the model's attention heads "
                    f"don't divide evenly across all of them)")
        extra_env: dict[str, str] | None = {"PJRT_DEVICE": "TPU"}
        ready_timeout = 5400  # first start compiles XLA graphs, which can take well over 30 minutes
    else:
        py = await ensure_venv(ctx, "vllm", ["vllm"], python="3.12", label="vLLM")
        tp = max(1, len(ctx.gpus))
        extra_env = None
        ready_timeout = 1800
    port = free_port()
    opts = spec.options
    detected = local_profile(path, spec.model.id)
    max_len = int(opts.get("ctx") or 8192)
    limit = context_length(model_config(path))
    if limit and max_len > limit:
        ctx.log(f"Context length {max_len} exceeds this model's maximum; using {limit}")
        max_len = limit
    cmd = [py, "-m", "vllm.entrypoints.openai.api_server", "--model", path,
           "--served-model-name", ctx.served_name, "--host", "127.0.0.1", "--port", str(port),
           "--max-model-len", str(max_len),
           "--tensor-parallel-size", str(tp)]
    if tpu:
        # TPUs are bf16-first and the TPU scheduler manages HBM across the sharded chips itself.
        cmd += ["--dtype", str(opts.get("dtype") or "bfloat16")]
    else:
        cmd += ["--dtype", str(opts.get("dtype") or ("half" if (ctx.hw.min_sm or 0) < 80 else "auto")),
                "--gpu-memory-utilization", str(float(opts.get("gpu_memory_utilization") or 0.90))]
        if opts.get("enforce_eager", (ctx.hw.min_sm or 0) < 80):
            cmd.append("--enforce-eager")
    if opts.get("quantization"):
        cmd += ["--quantization", str(opts["quantization"])]
    if opts.get("trust_remote_code") is True:
        cmd.append("--trust-remote-code")
    parser = opts.get("reasoning_parser")
    if not parser and detected["reasoning"].get("supported") and str(model_config(path).get("model_type", "")).startswith("qwen3"):
        parser = "qwen3"
    if parser:
        cmd += ["--reasoning-parser", str(parser)]
    proc = ManagedProcess(f"vllm-{ctx.dep_id}", cmd, env=ctx.env(extra_env),
                          log_path=ctx.paths.logs / f"{ctx.dep_id}-vllm.log", on_line=ctx.log)
    ctx.processes.append(proc)
    ctx.progress("loading", None, "Starting vLLM on the TPU (loading weights and compiling XLA graphs; "
                                  "the first start can take 10–30 minutes)" if tpu else
                "Starting vLLM (loading weights + warming up)")
    await proc.start()
    base = f"http://127.0.0.1:{port}"
    async with httpx.AsyncClient(timeout=5) as c:
        for _ in range(ready_timeout):
            if not proc.running:
                raise DeployError("vLLM exited: " + " | ".join(list(proc.tail)[-5:]))
            try:
                r = await c.get(f"{base}/health")
                if r.status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            await asyncio.sleep(1)
        else:
            limit = "90 minutes" if tpu else "30 minutes"
            raise DeployError(f"vLLM didn't become ready within {limit}")
    return OpenAIUpstream(base, ctx.served_name, template_thinking=True, profile=detected), {
        "port": port, "tensor_parallel": tp, "path": path, "device": "tpu" if tpu else "gpu", **detected}
