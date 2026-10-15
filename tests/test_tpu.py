"""TPU detection and fit: Kaggle TPU sessions report chips, memory and a TPU-capable runner."""

from __future__ import annotations

import os

from kaggleui.catalog.fit import TPU_ENGINES, Candidate, assess
from kaggleui.system.probe import Probe
from kaggleui.util import GB, GiB


def _clean_env(monkeypatch) -> None:
    for key in list(os.environ):
        if key.startswith("TPU_") or key == "COLAB_TPU_ADDR":
            monkeypatch.delenv(key)


def test_fake_tpu_profile_reports_chips(paths):
    hw = Probe(paths, fake="tpu").snapshot()
    assert hw.accelerator == "tpu"
    assert hw.accelerator_label == "TPU VM v5e-8"
    assert len(hw.tpus) == 8
    assert all(t.kind == "tpu" and t.vram_total == 16 * GiB for t in hw.tpus)
    assert hw.session_limit == 9 * 3600
    d = hw.to_dict()
    assert d["hbm_total_all"] == d["hbm_free_all"] == 8 * 16 * GiB
    assert d["vram_total_all"] == 0 and d["vram_free_all"] == 0
    assert all(chip["kind"] == "tpu" and chip["vram_free"] == 16 * GiB for chip in d["tpus"])
    assert any("TPU" in note for note in d["notes"])


def test_tpu_detected_from_environment(paths, monkeypatch):
    _clean_env(monkeypatch)
    monkeypatch.setenv("TPU_ACCELERATOR_TYPE", "v5litepod-8")
    hw = Probe(paths).snapshot()
    assert hw.accelerator == "tpu"
    assert hw.accelerator_label == "TPU v5e ×8"
    assert len(hw.tpus) == 8
    assert hw.tpus[0].vram_total == 16 * GiB


def test_tpu_type_parsing(monkeypatch):
    _clean_env(monkeypatch)
    assert Probe._tpu_type() == (None, None)
    monkeypatch.setenv("TPU_ACCELERATOR_TYPE", "v5litepod-8")
    assert Probe._tpu_type() == ("v5e", 8)
    monkeypatch.setenv("TPU_ACCELERATOR_TYPE", "v5e-8")
    assert Probe._tpu_type() == ("v5e", 8)
    monkeypatch.setenv("TPU_ACCELERATOR_TYPE", "v4-8")
    assert Probe._tpu_type() == ("v4", 8)
    monkeypatch.setenv("TPU_ACCELERATOR_TYPE", "v3-8")
    assert Probe._tpu_type() == ("v3", 8)
    monkeypatch.setenv("TPU_ACCELERATOR_TYPE", "v6e-8")
    assert Probe._tpu_type() == ("v6e", 8)
    monkeypatch.delenv("TPU_ACCELERATOR_TYPE")
    monkeypatch.setenv("COLAB_TPU_ADDR", "10.0.0.2:8470")
    assert Probe._tpu_type() == ("v2", 8)


def test_tpu_fit_uses_all_chips(hw):
    f = assess(Candidate(modality="text", engine="vllm", weights_bytes=int(40 * GB), params=40e9), hw("tpu"))
    assert f.verdict in ("good", "tight")
    assert f.placement == "split"
    assert f.capacity == int(8 * 16 * GiB * 0.97)
    assert any("Sharded across 8 TPU chips" in r for r in f.reasons)


def test_tpu_only_for_tpu_engines(hw):
    assert TPU_ENGINES == {"vllm"}
    for engine in ("ollama", "transformers", "diffusers"):
        f = assess(Candidate(modality="text", engine=engine, weights_bytes=int(4 * GB), params=4e9), hw("tpu"))
        assert f.verdict == "worst", engine
        assert "CUDA GPU" in f.reasons[0]


def test_model_too_big_for_tpu_slice(hw):
    f = assess(Candidate(modality="text", engine="vllm", weights_bytes=int(140 * GB), params=120e9,
                         attached=True), hw("tpu"))
    assert f.verdict == "worst"
    assert "HBM" in f.reasons[0]


def test_tpu_detected_from_vfio_devices(paths, monkeypatch):
    # Kaggle's v5e exposes chips as numbered /dev/vfio groups and may not set TPU_* variables.
    _clean_env(monkeypatch)
    from pathlib import Path

    monkeypatch.setattr(Probe, "_gpus", lambda self: [])
    monkeypatch.setattr(Probe, "_tpu_devices", staticmethod(lambda: [Path(f"/dev/vfio/{i}") for i in range(8)]))
    hw = Probe(paths).snapshot()
    assert hw.accelerator == "tpu"
    assert hw.accelerator_label == "TPU v5e ×8"
    assert hw.tpus[0].vram_total == 16 * GiB


def test_tpu_parallel_divides_attention_heads():
    from kaggleui.engines.vllm import tpu_parallel

    assert tpu_parallel(8, {"num_attention_heads": 32}) == 8
    assert tpu_parallel(8, {"num_attention_heads": 28}) == 4  # Qwen2.5-7B
    assert tpu_parallel(8, {"num_attention_heads": 14}) == 2  # Qwen2.5-0.5B
    assert tpu_parallel(8, {"num_attention_heads": 9}) == 1  # SmolLM2-135M (the TPU selftest model)
    assert tpu_parallel(8, {"text_config": {"num_attention_heads": 16}}) == 8
    assert tpu_parallel(8, {}) == 8
