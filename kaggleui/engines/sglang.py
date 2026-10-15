"""SGLang engine (isolated venv; tensor-parallel across both T4s).

GPU only: SGLang's TPU support lives in the separate sglang-jax runtime and doesn't cover the
v5e/v3 chips Kaggle offers, so TPU sessions use vLLM. Pinned to 0.5.19, the last release whose
PyPI wheels target the CUDA 12 driver Kaggle T4 machines ship.
"""

from __future__ import annotations

import asyncio

import httpx

from ..capabilities import local_profile
from ..catalog import hf
from ..proc import ManagedProcess
from ..schemas import DeploymentSpec
from ..util import free_port
from .base import OpenAIUpstream
from .context import DeployError, EngineContext
from .download import hf_download, repo_files
from .install import ensure_venv
from .model_support import model_config


async def deploy(spec: DeploymentSpec, ctx: EngineContext) -> tuple[OpenAIUpstream, dict]:
    if ctx.hw.accelerator == "tpu":
        raise DeployError("SGLang has no Kaggle TPU backend (its TPU support needs the separate sglang-jax "
                          "runtime). Switch the accelerator to GPU T4 ×2, or use vLLM, which runs on the TPU.")
    if (ctx.hw.min_sm or 0) < 75:
        raise DeployError("SGLang needs a T4 (sm_75) or newer GPU. Use the Ollama or Transformers engine instead.")
    if spec.model.source == "attached":
        path = spec.model.id
    else:
        revision = spec.options.get("revision")
        sel = hf.transformers_files(await repo_files(spec.model.id, revision=revision))
        path = await hf_download(ctx, spec.model.id, sel["files"], sel["size"], revision=revision)
    py = await ensure_venv(ctx, "sglang", ["sglang==0.5.19"], python="3.12", label="SGLang")
    tp = max(1, len(ctx.gpus))
    port = free_port()
    opts = spec.options
    detected = local_profile(path, spec.model.id)
    sm = ctx.hw.min_sm or 0
    cmd = [py, "-m", "sglang.launch_server", "--model-path", path,
           "--served-model-name", ctx.served_name, "--host", "127.0.0.1", "--port", str(port),
           "--context-length", str(int(opts.get("ctx") or 8192)),
           "--tp-size", str(tp),
           "--dtype", str(opts.get("dtype") or ("half" if sm < 80 else "auto")),
           "--mem-fraction-static", str(float(opts.get("mem_fraction_static") or 0.88))]
    if sm < 80:
        # T4s can hit FlashInfer problems; the Triton attention + PyTorch sampling kernels are SGLang's
        # documented fallback for sm75. CUDA graphs come off too: eager mode keeps the first load stable.
        cmd += ["--attention-backend", "triton", "--sampling-backend", "pytorch"]
        if opts.get("disable_cuda_graph", True):
            cmd.append("--disable-cuda-graph")
    if opts.get("quantization"):
        cmd += ["--quantization", str(opts["quantization"])]
    if opts.get("trust_remote_code") is True:
        cmd.append("--trust-remote-code")
    parser = opts.get("reasoning_parser")
    if not parser and detected["reasoning"].get("supported") and str(model_config(path).get("model_type", "")).startswith("qwen3"):
        parser = "qwen3"
    if parser:
        cmd += ["--reasoning-parser", str(parser)]
    proc = ManagedProcess(f"sglang-{ctx.dep_id}", cmd, env=ctx.env(),
                          log_path=ctx.paths.logs / f"{ctx.dep_id}-sglang.log", on_line=ctx.log)
    ctx.processes.append(proc)
    ctx.progress("loading", None, "Starting SGLang (loading weights + warming up)")
    await proc.start()
    base = f"http://127.0.0.1:{port}"
    async with httpx.AsyncClient(timeout=10) as c:
        for _ in range(1800):
            if not proc.running:
                raise DeployError("SGLang exited: " + " | ".join(list(proc.tail)[-5:]))
            try:
                r = await c.get(f"{base}/health")
                if r.status_code == 200:
                    # /health can answer before the model serves; one tiny generation confirms readiness.
                    g = await c.get(f"{base}/health_generate")
                    if g.status_code in (200, 404):  # 404: this build doesn't expose health_generate
                        break
            except httpx.HTTPError:
                pass
            await asyncio.sleep(1)
        else:
            raise DeployError("SGLang didn't become ready within 30 minutes")
    return OpenAIUpstream(base, ctx.served_name, template_thinking=True, profile=detected), {
        "port": port, "tensor_parallel": tp, "path": path, "device": "gpu", **detected}
