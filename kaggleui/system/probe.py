"""Hardware/session probe: GPUs (via NVML), TPU chips, RAM, disks, Kaggle session timing.

``--fake-hw`` profiles (t4x2, p100, tpu, cpu) let the UI and fit engine be developed off-Kaggle.
"""

from __future__ import annotations

import importlib.util
import os
import random
import re
import shutil
import socket
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import psutil

from ..config import ON_COLAB, ON_KAGGLE, Paths
from ..util import GiB, MiB

GPU_SESSION_LIMIT = 12 * 3600
TPU_SESSION_LIMIT = 9 * 3600

# HBM (GiB) per TPU chip by accelerator version. libtpu doesn't expose per-chip memory,
# so these are the published capacities used for fit estimates.
TPU_HBM_GIB = {"v2": 8, "v3": 16, "v4": 32, "v5e": 16, "v5p": 95, "v6e": 32, "v7x": 32}


@dataclass
class GpuInfo:
    """One memory device. ``kind`` distinguishes CUDA GPUs from TPU chips so the UI and the
    fit gauge can show either accelerator with the same per-device meters."""

    index: int
    name: str
    vram_total: int
    vram_used: int
    util: int | None = None
    temp: int | None = None
    compute_capability: str | None = None
    kind: str = "gpu"  # gpu | tpu

    @property
    def vram_free(self) -> int:
        return max(self.vram_total - self.vram_used, 0)

    @property
    def sm(self) -> int | None:
        if not self.compute_capability:
            return None
        major, _, minor = self.compute_capability.partition(".")
        return int(major) * 10 + int(minor or 0)


@dataclass
class DiskInfo:
    key: str
    label: str
    path: str
    total: int
    used: int
    free: int
    note: str = ""


@dataclass
class HardwareProfile:
    platform: str
    accelerator: str  # gpu | tpu | cpu
    accelerator_label: str
    gpus: list[GpuInfo]
    cpu_count: int
    ram_total: int
    ram_available: int
    disks: list[DiskInfo]
    run_type: str | None
    session_started: float
    session_limit: int
    internet: bool | None
    tpus: list[GpuInfo] = field(default_factory=list)
    fake: str | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def models_disk(self) -> DiskInfo | None:
        return next((d for d in self.disks if d.key == "models"), None)

    @property
    def min_sm(self) -> int | None:
        sms = [g.sm for g in self.gpus if g.sm]
        return min(sms) if sms else None

    def to_dict(self) -> dict:
        d = asdict(self)
        now = time.time()
        elapsed = max(now - self.session_started, 0)
        d["session_elapsed"] = int(elapsed)
        d["session_remaining"] = max(int(self.session_limit - elapsed), 0)
        d["vram_total_all"] = sum(g.vram_total for g in self.gpus)
        d["vram_free_all"] = sum(g.vram_free for g in self.gpus)
        d["hbm_total_all"] = sum(t.vram_total for t in self.tpus)
        d["hbm_free_all"] = sum(t.vram_free for t in self.tpus)
        for device, dd in zip((*self.gpus, *self.tpus), (*d["gpus"], *d["tpus"])):
            dd["vram_free"] = device.vram_free
            dd["sm"] = device.sm
        return d


FAKE_PROFILES: dict[str, dict] = {
    "t4x2": {
        "label": "GPU T4 ×2",
        "gpus": [("Tesla T4", 15360 * MiB, "7.5"), ("Tesla T4", 15360 * MiB, "7.5")],
        "ram": int(31.4 * GiB),
        "accelerator": "gpu",
    },
    "p100": {
        "label": "GPU P100",
        "gpus": [("Tesla P100-PCIE-16GB", 16384 * MiB, "6.0")],
        "ram": int(31.4 * GiB),
        "accelerator": "gpu",
    },
    "tpu": {"label": "TPU VM v5e-8", "gpus": [], "tpus": [("TPU v5e chip", 16 * GiB)] * 8,
            "ram": int(330 * GiB), "accelerator": "tpu"},
    "cpu": {"label": "CPU only", "gpus": [], "ram": int(31.4 * GiB), "accelerator": "cpu"},
}


class Probe:
    """Reads live hardware state. ``reserved`` lets fake GPUs show VRAM used by mock deployments."""

    def __init__(self, paths: Paths, fake: str | None = None):
        self.paths = paths
        self.fake = fake if fake in FAKE_PROFILES else None
        self.reserved: dict[int, int] = {}
        self._nvml = False
        self._started = self._session_start()
        self._internet: bool | None = None
        self._internet_checked = 0.0
        self._snapshot: HardwareProfile | None = None
        self._working_usage = (0.0, 0)
        if not self.fake:
            try:
                import pynvml  # type: ignore

                pynvml.nvmlInit()
                self._nvml = True
            except Exception:
                self._nvml = False
        self.refresh()

    # -- public -------------------------------------------------------------------------------
    def snapshot(self) -> HardwareProfile:
        """Return the last background measurement without blocking the HTTP event loop."""
        if self.fake:
            return self._fake_snapshot()
        assert self._snapshot is not None
        return self._snapshot

    def refresh(self) -> HardwareProfile:
        self._snapshot = self._measure()
        return self._snapshot

    def _measure(self) -> HardwareProfile:
        if self.fake:
            return self._fake_snapshot()
        gpus = self._gpus()
        tpus = [] if gpus else self._tpus()
        vm = psutil.virtual_memory()
        accel = "gpu" if gpus else ("tpu" if tpus else "cpu")
        return HardwareProfile(
            platform="kaggle" if ON_KAGGLE else ("colab" if ON_COLAB else "local"),
            accelerator=accel,
            accelerator_label=self._label(gpus, tpus, accel),
            gpus=gpus,
            tpus=tpus,
            cpu_count=psutil.cpu_count(logical=True) or 1,
            ram_total=vm.total,
            ram_available=vm.available,
            disks=self._disks(),
            run_type=os.environ.get("KAGGLE_KERNEL_RUN_TYPE"),
            session_started=self._started,
            session_limit=TPU_SESSION_LIMIT if accel == "tpu" else GPU_SESSION_LIMIT,
            internet=self.internet(),
            notes=self._notes(gpus, tpus, accel),
        )

    def internet(self) -> bool | None:
        if time.time() - self._internet_checked < 300 and self._internet is not None:
            return self._internet
        self._internet_checked = time.time()
        try:
            with socket.create_connection(("huggingface.co", 443), timeout=3):
                self._internet = True
        except OSError:
            self._internet = False
        return self._internet

    def torch_arch_list(self) -> list[str]:
        """CUDA archs the installed torch was built for (P100/sm_60 may be missing)."""
        try:
            out = subprocess.run(
                ["python", "-c", "import torch;print(' '.join(torch.cuda.get_arch_list()))"],
                capture_output=True, text=True, timeout=60,
            )
            return out.stdout.split()
        except Exception:
            return []

    # -- internals ----------------------------------------------------------------------------
    def _session_start(self) -> float:
        if ON_KAGGLE:
            try:
                return psutil.Process(1).create_time()
            except Exception:
                pass
        return time.time()

    def _gpus(self) -> list[GpuInfo]:
        if self._nvml:
            try:
                return self._gpus_nvml()
            except Exception:
                pass
        return self._gpus_smi()

    def _gpus_nvml(self) -> list[GpuInfo]:
        import pynvml  # type: ignore

        out = []
        for i in range(pynvml.nvmlDeviceGetCount()):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(h)
            if isinstance(name, bytes):
                name = name.decode()
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(h).gpu
            except Exception:
                util = None
            try:
                temp = pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)
            except Exception:
                temp = None
            try:
                major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(h)
                cc = f"{major}.{minor}"
            except Exception:
                cc = None
            out.append(GpuInfo(i, name, int(mem.total), int(mem.used), util, temp, cc))
        return out

    def _gpus_smi(self) -> list[GpuInfo]:
        if not shutil.which("nvidia-smi"):
            return []
        fields = "index,name,memory.total,memory.used,utilization.gpu,temperature.gpu,compute_cap"
        try:
            res = subprocess.run(
                ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10,
            )
        except Exception:
            return []
        gpus = []
        for line in res.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 7:
                continue

            def num(v: str) -> int | None:
                try:
                    return int(float(v))
                except ValueError:
                    return None

            gpus.append(GpuInfo(
                index=num(parts[0]) or 0, name=parts[1],
                vram_total=(num(parts[2]) or 0) * MiB, vram_used=(num(parts[3]) or 0) * MiB,
                util=num(parts[4]), temp=num(parts[5]), compute_capability=parts[6] or None,
            ))
        return gpus

    @staticmethod
    def _tpu_devices() -> list[Path]:
        """Chip device nodes: ``/dev/accel*`` up to v4; v5e and newer expose numbered ``/dev/vfio`` groups."""
        dev = Path("/dev")
        if not dev.is_dir():
            return []
        accel = sorted(dev.glob("accel*"))
        if accel:
            return accel
        # vfio groups can belong to any passthrough device, so only count them on a TPU image (libtpu).
        vfio = sorted(p for p in (dev / "vfio").glob("*") if p.name.isdigit()) if (dev / "vfio").is_dir() else []
        return vfio if vfio and importlib.util.find_spec("libtpu") else []

    def _has_tpu(self) -> bool:
        if any(k.startswith("TPU_") for k in os.environ) or "COLAB_TPU_ADDR" in os.environ:
            return True
        return bool(self._tpu_devices())

    @staticmethod
    def _tpu_type() -> tuple[str | None, int | None]:
        """(version, chip count) from the session's environment, e.g. ``("v5e", 8)``."""
        for var in ("TPU_ACCELERATOR_TYPE", "TPU_NAME"):
            match = re.search(r"v(\d+)(litepod|e|p|x)?(?:-(\d+))?", os.environ.get(var, ""), re.I)
            if match:
                suffix = (match.group(2) or "").lower()
                version = f"v{match.group(1)}" + ("e" if suffix == "litepod" else suffix)
                return version, int(match.group(3)) if match.group(3) else None
        if "COLAB_TPU_ADDR" in os.environ:
            return "v2", 8  # the classic Colab TPU is a single v2-8 slice
        return None, None

    def _tpus(self) -> list[GpuInfo]:
        """TPU chips as memory devices. HBM use isn't exposed by libtpu, so chips read as idle."""
        if not self._has_tpu():
            return []
        version, typed_chips = self._tpu_type()
        devices = self._tpu_devices()
        if not version and devices and devices[0].parent.name == "vfio":
            version = "v5e"  # Kaggle's vfio-based TPU; older v3-8 sessions use /dev/accel*
        chips = len(devices) or typed_chips or (8 if "COLAB_TPU_ADDR" in os.environ else 1)
        hbm = TPU_HBM_GIB.get(version or "", 16) * GiB
        name = f"TPU {version} chip" if version else "TPU chip"
        return [GpuInfo(i, name, hbm, 0, kind="tpu") for i in range(chips)]

    @staticmethod
    def _label(gpus: list[GpuInfo], tpus: list[GpuInfo], accel: str) -> str:
        if gpus:
            names = {g.name.replace("Tesla ", "").replace("NVIDIA ", "").split("-")[0] for g in gpus}
            name = "/".join(sorted(names))
            return f"GPU {name} ×{len(gpus)}" if len(gpus) > 1 else f"GPU {name}"
        if tpus:
            version = tpus[0].name.removeprefix("TPU ").removesuffix(" chip")
            return f"TPU {version} ×{len(tpus)}" if version else f"TPU ×{len(tpus)}"
        return "CPU only"

    def _disks(self) -> list[DiskInfo]:
        disks = []
        self.paths.home.mkdir(parents=True, exist_ok=True)
        candidates = [
            ("models", "Models & cache", self.paths.home, "Ephemeral — wiped when the session ends"),
            ("working", "/kaggle/working", Path("/kaggle/working"), "Saved as notebook output (20 GB cap)"),
            ("input", "/kaggle/input", Path("/kaggle/input"), "Attached datasets & models (read-only)"),
        ]
        for key, label, path, note in candidates:
            if not path.exists():
                continue
            try:
                u = shutil.disk_usage(path)
            except OSError:
                continue
            total, used, free = u.total, u.used, u.free
            if key == "working" and ON_KAGGLE:
                # /kaggle/working shares a big disk but output is capped at 20 GB.
                from ..util import dir_size

                if time.monotonic() - self._working_usage[0] > 30:
                    self._working_usage = (time.monotonic(), dir_size(path))
                used_out = self._working_usage[1]
                total, used, free = 20 * 1000**3, used_out, max(20 * 1000**3 - used_out, 0)
            disks.append(DiskInfo(key, label, str(path), total, used, free, note))
        return disks

    def _notes(self, gpus: list[GpuInfo], tpus: list[GpuInfo], accel: str) -> list[str]:
        notes = []
        if accel == "tpu":
            chips = len(tpus) or 8
            notes.append(f"{chips} TPU chips detected. Text models run through vLLM's TPU backend "
                         "(experimental): the first start compiles XLA graphs and can take 10–30 minutes. "
                         "Ollama, Diffusers, TTS and ASR models need a GPU T4 ×2.")
        elif accel == "cpu":
            notes.append("No accelerator detected. Turn on a GPU (GPU T4 ×2 recommended) or a TPU in notebook settings.")
        if any((g.sm or 0) < 70 for g in gpus):
            notes.append("P100 (sm_60): vLLM is unavailable and some recent PyTorch builds dropped support. "
                         "Ollama/GGUF models work best here.")
        if ON_KAGGLE and self._internet is False:
            notes.append("Internet looks off. Enable it in notebook settings (requires a phone-verified account).")
        return notes

    def _fake_snapshot(self) -> HardwareProfile:
        prof = FAKE_PROFILES[self.fake]  # type: ignore[index]
        gpus = []
        for i, (name, total, cc) in enumerate(prof["gpus"]):
            used = 280 * MiB + self.reserved.get(i, 0)
            util = min(100, random.randint(0, 4) + (60 if self.reserved.get(i) else 0))
            gpus.append(GpuInfo(i, name, total, min(used, total), util, 38 + util // 4, cc))
        tpus = [GpuInfo(i, name, total, 0, kind="tpu")
                for i, (name, total) in enumerate(prof.get("tpus", []))]
        ram_total = prof["ram"]
        reserved_ram = sum(self.reserved.values()) // 8
        disk_total = 73 * 1000**3
        disk_used = 9 * 1000**3 + sum(self.reserved.values())
        disks = [
            DiskInfo("models", "Models & cache", str(self.paths.home), disk_total, disk_used,
                     disk_total - disk_used, "Ephemeral — wiped when the session ends"),
            DiskInfo("working", "/kaggle/working", "/kaggle/working", 20 * 1000**3, 12 * 1000**2,
                     20 * 1000**3 - 12 * 1000**2, "Saved as notebook output (20 GB cap)"),
            DiskInfo("input", "/kaggle/input", "/kaggle/input", 0, 0, 0, "Attached datasets & models (read-only)"),
        ]
        accel = prof["accelerator"]
        return HardwareProfile(
            platform="kaggle (simulated)",
            accelerator=accel,
            accelerator_label=prof["label"],
            gpus=gpus,
            tpus=tpus,
            cpu_count=4,
            ram_total=ram_total,
            ram_available=int(ram_total * 0.86) - reserved_ram,
            disks=disks,
            run_type="Interactive",
            session_started=self._started,
            session_limit=TPU_SESSION_LIMIT if accel == "tpu" else GPU_SESSION_LIMIT,
            internet=True,
            fake=self.fake,
            notes=self._notes(gpus, tpus, accel),
        )
