from kaggleui.catalog.fit import Candidate, Fit, assess, gguf_size, pick_quant, rank, weights_from_safetensors
from kaggleui.util import GB


def text(size_gb: float, params_b: float, engine: str = "ollama", **kw) -> Candidate:
    return Candidate(modality="text", engine=engine, weights_bytes=int(size_gb * GB), params=params_b * 1e9, **kw)


def test_small_llm_fits_one_t4(hw):
    f = assess(text(5.2, 8), hw("t4x2"))
    assert f.verdict == "good"
    assert f.placement == "single"


def test_27b_splits_across_both_t4s(hw):
    f = assess(text(17.7, 27), hw("t4x2"))
    assert f.verdict in ("good", "tight")
    assert f.placement == "split"
    assert any("Split across 2 GPUs" in r for r in f.reasons)


def test_27b_on_single_p100_spills_or_fails(hw):
    f = assess(text(17.7, 27), hw("p100"))
    assert f.placement != "split"
    assert f.verdict in ("tight", "worst")


def test_70b_is_worst(hw):
    f = assess(text(42, 70), hw("t4x2"))
    assert f.verdict == "worst"


def test_vllm_not_on_p100(hw):
    f = assess(text(5, 8, engine="vllm"), hw("p100"))
    assert f.verdict == "worst"
    assert "vllm" in f.reasons[0].lower()


def test_sglang_runs_and_splits_on_t4s(hw):
    f = assess(text(5.2, 8, engine="sglang"), hw("t4x2"))
    assert f.verdict in ("good", "best")
    assert f.placement == "single"
    f = assess(text(17.7, 27, engine="sglang"), hw("t4x2"))
    assert f.placement == "split"
    assert any("Split across 2 GPUs" in r for r in f.reasons)


def test_sglang_not_on_older_gpus_or_tpu(hw):
    f = assess(text(5, 8, engine="sglang"), hw("p100"))
    assert f.verdict == "worst"
    assert "sglang" in f.reasons[0].lower()
    f = assess(text(2, 1.5, engine="sglang"), hw("tpu"))
    assert f.verdict == "worst"
    assert "CUDA GPU" in f.reasons[0]


def test_tpu_needs_gpu_for_diffusers(hw):
    f = assess(Candidate(modality="image", engine="diffusers", weights_bytes=int(7 * GB)), hw("tpu"))
    assert f.verdict == "worst"
    assert "GPU" in f.reasons[0]


def test_cpu_can_run_small_ollama_slowly(hw):
    f = assess(text(0.5, 0.6), hw("cpu"))
    assert f.verdict == "tight"
    assert f.placement == "cpu"


def test_flux_needs_4bit_and_offload(hw):
    c = Candidate(modality="image", engine="diffusers", weights_bytes=int(33.7 * GB),
                  largest_component_bytes=int(23.8 * GB))
    f = assess(c, hw("t4x2"))
    assert f.verdict == "tight"
    assert f.suggest == {"offload": "model", "quantize": "nf4"}


def test_sd35_uses_offload(hw):
    c = Candidate(modality="image", engine="diffusers", weights_bytes=int(16.3 * GB),
                  largest_component_bytes=int(9.8 * GB))
    f = assess(c, hw("t4x2"))
    assert f.verdict == "tight"
    assert f.suggest == {"offload": "model"}


def test_non_split_engine_reports_single_gpu_limit(hw):
    c = Candidate(modality="world", engine="world_recipe", weights_bytes=int(21 * GB), min_vram=int(22 * GB))
    f = assess(c, hw("t4x2"))
    assert f.verdict == "worst"
    assert "one GPU" in f.reasons[0]


def test_disk_check(hw):
    f = assess(text(80, 120), hw("t4x2"))
    assert f.verdict == "worst"
    assert "disk" in f.reasons[0].lower()


def test_no_engine_is_worst(hw):
    f = assess(Candidate(modality="tts", engine=None, weights_bytes=GB), hw("t4x2"))
    assert f.verdict == "worst"


def test_gated_warning_without_token(hw, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    f = assess(text(5, 8, gated=True), hw("t4x2"))
    assert any("Hugging Face token" in w for w in f.warnings)


def test_rank_promotes_top_three_and_skips_unproven():
    fits = [Fit("good", 1, 10, "single", score=s) for s in (0.9, 0.8, 0.7, 0.6)]
    fits.append(Fit("good", 1, 10, "single", score=0.99, unproven=True))
    rank(fits)
    assert [f.verdict for f in fits] == ["best", "best", "best", "good", "good"]


def test_pick_quant_prefers_largest_comfortable(hw):
    q, f = pick_quant(8e9, None, text(0, 8), hw("t4x2"))
    assert q == "Q8_0"
    q, f = pick_quant(32e9, None, text(0, 32), hw("t4x2"))
    assert q in ("Q5_K_M", "Q4_K_M", "Q6_K")


def test_size_helpers():
    assert 4.7e9 < gguf_size(8e9, "Q4_K_M") < 5.3e9
    assert weights_from_safetensors({"BF16": 1_000}) == 2_000
    assert weights_from_safetensors({"F32": 1_000}) == 2_000  # loaded as fp16
    assert weights_from_safetensors({"I8": 1_000, "F16": 10}) == 1_020
