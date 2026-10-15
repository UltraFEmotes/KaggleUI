"""Ollama engine: one shared ``ollama serve`` for all text deployments (GGUF from Ollama, HF or /kaggle/input)."""

from __future__ import annotations

import asyncio
import json
import os
import platform
import tempfile

import httpx

from ..capabilities import profile
from ..proc import ManagedProcess, run_cmd
from ..schemas import DeploymentSpec
from ..util import which
from .base import OpenAIUpstream
from .context import DeployError, EngineContext

OLLAMA_PORT = int(os.environ.get("KAGGLEUI_OLLAMA_PORT", "11434"))


class OllamaServer:
    _instance: OllamaServer | None = None

    def __init__(self) -> None:
        self.proc: ManagedProcess | None = None
        self.base = f"http://127.0.0.1:{OLLAMA_PORT}"
        self.exe: str | None = None
        self._lock = asyncio.Lock()
        self.version: str | None = None

    @classmethod
    def get(cls) -> OllamaServer:
        if cls._instance is None:
            cls._instance = OllamaServer()
        return cls._instance

    async def healthy(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=3) as c:
                r = await c.get(f"{self.base}/api/version")
                if r.status_code == 200:
                    self.version = r.json().get("version")
                    return True
        except httpx.HTTPError:
            pass
        return False

    async def ensure(self, ctx: EngineContext) -> str:
        async with self._lock:
            if await self.healthy():
                return self.base
            self.exe = await self._binary(ctx)
            ctx.progress("installing", None, "Starting Ollama")
            env = {
                "OLLAMA_HOST": f"127.0.0.1:{OLLAMA_PORT}",
                "OLLAMA_MODELS": os.environ.get("OLLAMA_MODELS", str(ctx.paths.ollama_models)),
                "OLLAMA_KEEP_ALIVE": "-1",
                "OLLAMA_MAX_LOADED_MODELS": "4",
                "OLLAMA_NUM_PARALLEL": "2",
                "OLLAMA_ORIGINS": "*",
            }
            env.pop("CUDA_VISIBLE_DEVICES", None)
            self.proc = ManagedProcess("ollama", [self.exe, "serve"], env=env,
                                       log_path=ctx.paths.logs / "ollama.log")
            await self.proc.start()
            for _ in range(120):
                if await self.healthy():
                    ctx.log(f"Ollama {self.version} is up")
                    return self.base
                if not self.proc.running:
                    raise DeployError("Ollama exited: " + " | ".join(list(self.proc.tail)[-4:]))
                await asyncio.sleep(0.5)
            raise DeployError("Ollama didn't start within 60s")

    async def _binary(self, ctx: EngineContext) -> str:
        exe = which("ollama", [ctx.paths.bin])
        if exe:
            return exe
        if platform.system() != "Linux":
            raise DeployError("Ollama isn't installed. Get it from https://ollama.com/download "
                              "(or start KaggleUI with --mock-engines for UI development).")
        ctx.progress("installing", None, "Installing Ollama (about 30s)")
        script = ("command -v zstd >/dev/null 2>&1 || (apt-get update -qq && apt-get install -y -qq zstd) "
                  ">/dev/null 2>&1; curl -fsSL https://ollama.com/install.sh | sh")
        rc = await run_cmd(script, on_line=ctx.log)
        exe = which("ollama", [ctx.paths.bin])
        if rc != 0 or not exe:
            raise DeployError("Installing Ollama failed. Is Internet enabled in the notebook settings?")
        return exe

    async def stop(self) -> None:
        if self.proc:
            await self.proc.stop()


def model_ref(spec: DeploymentSpec) -> str | None:
    m = spec.model
    if m.source == "attached":
        return None
    if m.source == "hf":
        return f"hf.co/{m.id}:{m.variant}" if m.variant else f"hf.co/{m.id}"
    if ":" in m.id:
        return m.id
    return f"{m.id}:{m.variant or 'latest'}"


async def _pull(base: str, ref: str, ctx: EngineContext) -> None:
    ctx.progress("downloading", 0.0, f"Pulling {ref}")
    totals: dict[str, int] = {}
    done: dict[str, int] = {}
    async with httpx.AsyncClient(timeout=httpx.Timeout(connect=10, read=None, write=60, pool=10)) as c:
        async with c.stream("POST", f"{base}/api/pull", json={"model": ref, "stream": True}) as r:
            if r.status_code != 200:
                raise DeployError(f"Pull failed ({r.status_code}): {(await r.aread()).decode()[:300]}")
            last = 0.0
            async for line in r.aiter_lines():
                if not line.strip():
                    continue
                try:
                    ev = json.loads(line)
                except ValueError:
                    continue
                if ev.get("error"):
                    raise DeployError(f"Pull failed: {ev['error']}")
                digest = ev.get("digest")
                if digest and ev.get("total"):
                    totals[digest] = int(ev["total"])
                    done[digest] = int(ev.get("completed") or 0)
                total = sum(totals.values())
                now = asyncio.get_running_loop().time()
                if total and now - last > 0.7:
                    got = sum(done.values())
                    ctx.progress("downloading", min(got / total, 0.999),
                                 f"Pulling {ref}: {got / 1e9:.2f} / {total / 1e9:.2f} GB")
                    last = now
                elif ev.get("status") and not digest:
                    ctx.log(f"ollama: {ev['status']}")
    ctx.progress("downloading", 1.0, f"Pulled {ref}")


def model_context(show: dict) -> int | None:
    """Trained context window from ``/api/show`` (``model_info["<arch>.context_length"]``)."""
    info = show.get("model_info") or {}
    value = info.get(f"{info.get('general.architecture')}.context_length")
    if value is None:
        value = next((v for k, v in info.items() if k.endswith(".context_length")), None)
    return int(value) if isinstance(value, int | float) and value > 0 else None


async def deploy(spec: DeploymentSpec, ctx: EngineContext) -> tuple[OpenAIUpstream, dict]:
    server = OllamaServer.get()
    base = await server.ensure(ctx)
    ref = model_ref(spec)
    if ref:
        await _pull(base, ref, ctx)
    alias = f"kui-{ctx.dep_id}"
    num_ctx = int(spec.options.get("ctx") or 8192)
    ctx.progress("loading", None, "Creating model with your settings")
    async with httpx.AsyncClient(timeout=httpx.Timeout(connect=10, read=1800, write=600, pool=10)) as c:
        if ref:
            r = await c.post(f"{base}/api/show", json={"model": ref})
            limit = model_context(r.json()) if r.status_code == 200 else None
            if limit and num_ctx > limit:
                ctx.log(f"Context length {num_ctx} exceeds this model's maximum; using {limit}")
                num_ctx = limit
            r = await c.post(f"{base}/api/create", json={
                "model": alias, "from": ref, "parameters": {"num_ctx": num_ctx}, "stream": False})
            if r.status_code != 200:
                raise DeployError(f"Couldn't create model: {r.text[:300]}")
        else:
            # Attached GGUF on disk: the CLI uploads local files into Ollama's blob store.
            mf = tempfile.NamedTemporaryFile("w", suffix=".Modelfile", delete=False)
            path = spec.model.id
            if any(c in path for c in ('\n', '\r', '"')):
                raise DeployError("The model path contains unsupported quote or newline characters")
            mf.write(f'FROM "{path}"\nPARAMETER num_ctx {num_ctx}\n')
            mf.close()
            rc = await run_cmd([server.exe or "ollama", "create", alias, "-f", mf.name],
                               env={"OLLAMA_HOST": base.replace("http://", "")}, on_line=ctx.log)
            os.unlink(mf.name)
            if rc != 0:
                raise DeployError("ollama create failed for the attached GGUF")
        r = await c.post(f"{base}/api/show", json={"model": alias})
        metadata = r.json() if r.status_code == 200 else {}
        caps = metadata.get("capabilities", [])
        detected = profile(spec.model.id, tags=caps, thinking=metadata.get("thinking"), authoritative=True)
        ctx.progress("loading", None, "Loading into GPU memory")
        if "embedding" in caps and "completion" not in caps:
            r = await c.post(f"{base}/api/embed", json={"model": alias, "input": "warmup", "keep_alive": -1})
        else:
            r = await c.post(f"{base}/api/generate", json={"model": alias, "prompt": "", "keep_alive": -1})
        if r.status_code != 200:
            raise DeployError(f"Loading failed: {r.text[:300]}")
    info = {"capabilities": caps, "upstream": alias, "ref": ref or spec.model.id, "num_ctx": num_ctx,
            "ollama_version": server.version, **detected}
    return OpenAIUpstream(base, alias, native=frozenset({"anthropic", "ollama"}), profile=detected), info


async def undeploy(dep_id: str) -> None:
    server = OllamaServer.get()
    alias = f"kui-{dep_id}"
    try:
        async with httpx.AsyncClient(timeout=30) as c:
            await c.post(f"{server.base}/api/generate", json={"model": alias, "keep_alive": 0})
            await c.request("DELETE", f"{server.base}/api/delete", json={"model": alias})
    except httpx.HTTPError:
        pass
