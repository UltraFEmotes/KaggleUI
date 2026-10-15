"""Will this model run well on *this* machine? VRAM/RAM/disk estimates, compat rules and tags.

Verdicts: ``best`` (assigned by :func:`rank`), ``good``, ``tight``, ``worst``.
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import asdict, dataclass, field

from ..system.probe import HardwareProfile
from ..util import GB, human_bytes

# Effective bits per weight for llama.cpp quant types (incl. scales/mins overhead).
QUANT_BITS: dict[str, float] = {
    "IQ1_S": 1.6, "IQ1_M": 1.75, "IQ2_XXS": 2.1, "IQ2_XS": 2.3, "IQ2_S": 2.5, "IQ2_M": 2.7,
    "Q2_K": 3.0, "Q2_K_L": 3.2, "IQ3_XXS": 3.1, "IQ3_XS": 3.3, "IQ3_S": 3.5, "IQ3_M": 3.6,
    "Q3_K_S": 3.5, "Q3_K_M": 3.9, "Q3_K_L": 4.3, "Q3_K_XL": 4.4, "IQ4_XS": 4.3, "IQ4_NL": 4.5,
    "Q4_0": 4.55, "Q4_1": 5.0, "Q4_K_S": 4.6, "Q4_K_M": 4.85, "Q4_K_L": 5.0, "Q4_K_XL": 5.0,
    "MXFP4": 4.25, "Q5_0": 5.5, "Q5_1": 6.0, "Q5_K_S": 5.55, "Q5_K_M": 5.7, "Q5_K_L": 5.8,
    "Q5_K_XL": 5.9, "Q6_K": 6.56, "Q6_K_L": 6.7, "Q6_K_XL": 6.8, "Q8_0": 8.5, "Q8_K_XL": 8.7,
    "F16": 16.0, "BF16": 16.0, "F32": 32.0,
}
QUANT_PREFERENCE = ("Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "IQ4_XS", "Q3_K_M", "Q2_K")

DTYPE_BYTES: dict[str, float] = {
    "F64": 8, "F32": 4, "BF16": 2, "F16": 2, "FP16": 2, "F8_E4M3": 1, "F8_E5M2": 1, "I64": 8,
    "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1, "U4": 0.5, "I4": 0.5,
}
FLOAT_DTYPES = {"F64", "F32", "BF16", "F16", "FP16"}

# Engines that can spread one model over several GPUs.
SPLIT_ENGINES = {"ollama", "vllm", "sglang", "transformers"}
# Engines that can keep part of the model in system RAM (slower).
OFFLOAD_ENGINES = {"diffusers", "diffusers_video", "diffusers_world", "transformers", "transformers_tts"}
# Minimum CUDA compute capability (sm) per engine.
ENGINE_MIN_SM = {"vllm": 70, "sglang": 75, "diffusers_world": 80}
# Engines that are fine on CPU for small models.
CPU_FRIENDLY = {"ollama", "faster_whisper", "kokoro", "transformers_asr", "transformers_tts", "mock"}
# Engines with a TPU path. vLLM's TPU backend shards one model across every chip in the slice.
TPU_ENGINES = {"vllm"}

GPU_HEADROOM = 0.97  # fraction of free VRAM we let a model plan to use
COMFORT = 0.85  # above this fraction of capacity a fit is "tight"


@dataclass
class Candidate:
    modality: str
    engine: str | None
    weights_bytes: int | None = None
    params: float | None = None
    largest_component_bytes: int | None = None
    ctx: int = 8192
    min_vram: int | None = None
    gated: bool = False
    bf16_only: bool = False
    experimental: bool = False
    curated: bool = False
    downloads: int = 0
    likes: int = 0
    updated: float | None = None  # unix time
    attached: bool = False  # already on disk under /kaggle/input
    min_sm: int | None = None


@dataclass
class Fit:
    verdict: str  # best | good | tight | worst
    vram_needed: int | None
    capacity: int | None
    placement: str  # single | split | offload | cpu | none
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    disk_needed: int | None = None
    score: float = 0.0
    suggest: dict = field(default_factory=dict)
    unproven: bool = False
    """Engine options that make this fit work, e.g. ``{"offload": "model"}`` or ``{"quantize": "nf4"}``."""

    def to_dict(self) -> dict:
        return asdict(self)


def gguf_size(params: float, quant: str) -> int:
    bits = QUANT_BITS.get(quant.upper(), 4.85)
    return int(params * bits / 8 * 1.02)


def weights_from_safetensors(parameters: dict[str, int] | None, load_dtype_bytes: float = 2) -> int | None:
    """Bytes in VRAM when float weights are loaded as fp16 (T4/P100 have no fast bf16)."""
    if not parameters:
        return None
    total = 0.0
    for dtype, count in parameters.items():
        dt = dtype.upper()
        if dt in FLOAT_DTYPES:
            total += count * load_dtype_bytes
        else:
            total += count * DTYPE_BYTES.get(dt, 2)
    return int(total)


def runtime_vram(c: Candidate) -> int | None:
    if c.min_vram:
        return c.min_vram
    w = c.weights_bytes
    if w is None and c.params:
        w = int(c.params * 2)
    if w is None:
        return None
    m = c.modality
    if m == "text":
        p_b = (c.params or w / 0.6) / 1e9
        kv_per_token = 40_000 * max(p_b, 0.1) ** 0.6  # ≈128 KB/token for an 8B GQA model
        return int(w + kv_per_token * c.ctx + 0.7 * GB)
    if m == "image":
        return int(w * 1.15 + 1.2 * GB)
    if m == "video":
        return int(w * 1.3 + 2.5 * GB)
    if m in ("tts", "audio", "stt"):
        return int(w * 1.2 + 0.6 * GB)
    if m == "world":
        return int(w * 1.3 + 3 * GB)
    return int(w * 1.2 + GB)


def _gpu_label(hw: HardwareProfile) -> str:
    return hw.accelerator_label.replace("GPU ", "")


def assess(c: Candidate, hw: HardwareProfile) -> Fit:
    need = runtime_vram(c)
    w = c.weights_bytes or (int(c.params * 2) if c.params else None)
    disk_needed = 0 if c.attached else (int(w * 1.05) if w else None)
    warnings: list[str] = []
    if c.gated and not os.environ.get("HF_TOKEN"):
        warnings.append("Gated: add your Hugging Face token (Access & API → Keys & tokens) and accept the license on HF")
    if c.experimental:
        warnings.append("Experimental on Kaggle GPUs")

    def done(verdict: str, placement: str, reasons: list[str], capacity: int | None = None,
             suggest: dict | None = None) -> Fit:
        f = Fit(verdict, need, capacity, placement, reasons, warnings, disk_needed, suggest=suggest or {},
                unproven=c.experimental and not c.curated)
        f.score = score(c, f)
        return f

    if not c.engine:
        return done("worst", "none", ["No engine in KaggleUI can run this model yet"])

    # Disk check first: a model that can't be downloaded can't run.
    models_disk = hw.models_disk
    if disk_needed and models_disk and disk_needed > models_disk.free:
        return done("worst", "none", [f"Not enough disk: needs {human_bytes(disk_needed)}, "
                                      f"{human_bytes(models_disk.free)} free"])

    if hw.accelerator == "tpu" and hw.tpus:
        if c.engine not in TPU_ENGINES:
            return done("worst", "none", [f"{c.engine or 'This runner'} needs a CUDA GPU; KaggleUI doesn't run "
                                          "it on TPUs. Switch the accelerator to GPU T4 ×2, or pick a text "
                                          "model (vLLM runs on TPU)."])
        if need is None:
            return done("tight", "single", ["Size unknown, so fit can't be estimated"])
        capacity = int(sum(t.vram_free for t in hw.tpus) * GPU_HEADROOM)
        chips = len(hw.tpus)
        spread = [f"Sharded across {chips} TPU chips"] if chips > 1 else []
        pct = f"≈{human_bytes(need)} of {human_bytes(capacity)} HBM"
        placement = "split" if chips > 1 else "single"
        if need <= capacity * COMFORT:
            return done("good", placement, [pct] + spread, capacity)
        if need <= capacity:
            return done("tight", placement, [pct + " (near the limit)"] + spread, capacity)
        return done("worst", "none", [f"Needs ≈{human_bytes(need)} HBM; {human_bytes(capacity)} available on "
                                      f"{hw.accelerator_label}"], capacity)

    if hw.accelerator != "gpu" or not hw.gpus:
        if c.engine in CPU_FRIENDLY and need is not None and need <= hw.ram_available * 0.8:
            return done("tight", "cpu", ["No GPU: runs on CPU, slowly"])
        return done("worst", "none", ["Needs a GPU. Switch the accelerator to GPU T4 ×2"])

    min_sm = c.min_sm or ENGINE_MIN_SM.get(c.engine)
    if min_sm and (hw.min_sm or 0) < min_sm:
        return done("worst", "none", [f"{c.engine} doesn't support {_gpu_label(hw)} (sm_{hw.min_sm})"])

    if c.bf16_only and (hw.min_sm or 0) < 80:
        warnings.append("Made for bf16; runs in fp16 on this GPU (small quality risk)")
    if c.modality in ("video", "world") and (hw.min_sm or 0) < 80:
        warnings.append("Slow on this GPU: expect minutes per clip")

    if need is None:
        return done("tight", "single", ["Size unknown, so fit can't be estimated"])

    free = sorted((g.vram_free for g in hw.gpus), reverse=True)
    single = int(free[0] * GPU_HEADROOM)
    total = int(sum(free) * GPU_HEADROOM)
    in_use = sum(g.vram_used for g in hw.gpus) - 350 * 1024**2 * len(hw.gpus)
    busy_note = [f"{human_bytes(in_use)} VRAM already in use"] if in_use > 1.5 * GB else []
    pct = lambda cap: f"≈{human_bytes(need)} of {human_bytes(cap)} VRAM"  # noqa: E731

    if need <= single * COMFORT:
        return done("good", "single", [pct(single)] + busy_note, single)
    if need <= single:
        return done("tight", "single", [pct(single) + " (near the limit)"] + busy_note, single)
    can_split = c.engine in SPLIT_ENGINES and len(hw.gpus) > 1
    if can_split and need <= total * COMFORT:
        return done("good", "split", [pct(total), f"Split across {len(hw.gpus)} GPUs"] + busy_note, total)
    if can_split and need <= total:
        return done("tight", "split", [pct(total) + " (near the limit)", f"Split across {len(hw.gpus)} GPUs"]
                    + busy_note, total)
    if c.engine in OFFLOAD_ENGINES and w:
        largest = c.largest_component_bytes or int(w * 0.6)
        if largest * 1.15 + 1.2 * GB <= single and w <= hw.ram_available * 0.85:
            return done("tight", "offload", [f"≈{human_bytes(need)} total, more than one GPU holds",
                                             "Uses CPU offload: works, but slower"] + busy_note, single,
                        {"offload": "model"})
        if c.engine in ("diffusers", "diffusers_video") and largest * 0.3 + 1.5 * GB <= single \
                and w * 0.5 <= hw.ram_available * 0.85:
            return done("tight", "offload", [f"≈{human_bytes(need)} at fp16: too big for {_gpu_label(hw)}",
                                             "Runs with 4-bit (nf4) weights + CPU offload; slower, slight quality loss"]
                        + busy_note, single, {"offload": "model", "quantize": "nf4"})
    if c.engine == "ollama":
        cap = total if len(hw.gpus) > 1 else single
        overflow = need - cap
        if cap >= need * 0.8 and overflow <= hw.ram_available * 0.6:
            return done("tight", "offload", [pct(cap), f"{math.ceil(overflow / need * 100)}% runs on CPU: slow"]
                        + busy_note, cap)
    if len(hw.gpus) > 1 and not can_split:
        return done("worst", "none", [f"Needs ≈{human_bytes(need)} VRAM on one GPU; the largest has "
                                      f"{human_bytes(single)} free"] + busy_note, single)
    return done("worst", "none", [f"Needs ≈{human_bytes(need)} VRAM; {human_bytes(total)} available"]
                + busy_note, total)


def score(c: Candidate, f: Fit) -> float:
    if f.verdict == "worst":
        return -1.0
    util = (f.vram_needed or 0) / f.capacity if f.capacity else 0.4
    # Bigger models are usually better, as long as they fit comfortably.
    util_score = max(0.0, 1 - abs(0.62 - util) / 0.62)
    pop = min(math.log10(c.downloads + 1) / 7, 1.0) * 0.7 + min(math.log10(c.likes + 1) / 4, 1.0) * 0.3
    recency = 0.5
    if c.updated:
        age_days = (time.time() - c.updated) / 86400
        recency = max(0.0, 1 - age_days / 720)
    s = 0.45 * util_score + 0.35 * pop + 0.2 * recency
    if c.curated:
        s += 0.25
    if c.attached:
        s += 0.15
    if f.verdict == "tight":
        s *= 0.55
    if f.placement == "offload":
        s *= 0.7
    if c.experimental:
        s *= 0.85
    return round(s, 4)


def rank(fits: list[Fit], top_n: int = 3) -> None:
    """Promote the top-scoring comfortable fits to ``best``, in place (never unproven experimental ones)."""
    good = sorted((f for f in fits if f.verdict == "good" and not f.unproven), key=lambda f: f.score, reverse=True)
    for f in good[:top_n]:
        f.verdict = "best"


def pick_quant(params: float, available: list[str] | None, c_template: Candidate, hw: HardwareProfile
               ) -> tuple[str, Fit]:
    """Largest preferred GGUF quant that fits comfortably; else the best that fits at all."""
    options = [q for q in QUANT_PREFERENCE if not available or q in {a.upper() for a in available}]
    if not options and available:
        options = sorted({a.upper() for a in available}, key=lambda q: -QUANT_BITS.get(q, 4.85))
    if not options:
        options = list(QUANT_PREFERENCE)
    results = []
    for q in options:
        cand = Candidate(**{**asdict(c_template), "weights_bytes": gguf_size(params, q), "params": params})
        results.append((q, assess(cand, hw)))
    for wanted in ("good", "tight"):
        for q, f in results:
            if f.verdict == wanted:
                return q, f
    return results[-1]
