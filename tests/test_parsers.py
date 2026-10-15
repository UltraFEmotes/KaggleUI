from pathlib import Path

from kaggleui.capabilities import context_length
from kaggleui.catalog import hf, ollama
from kaggleui.catalog.attached import scan
from kaggleui.catalog.resolve import resolve_hf
from kaggleui.util import parse_params, parse_size

FIX = Path(__file__).parent / "fixtures"


def test_ollama_search_parse():
    items = ollama.parse_search((FIX / "ollama_search.html").read_text())
    assert len(items) == 4
    first = items[0]
    assert first["name"] and first["description"]
    assert "tools" in first["capabilities"]
    assert all(s[-1] in "bm" for s in first["sizes"])
    assert first["downloads"] > 1000


def test_ollama_tags_parse_ignores_digest_hex():
    tags = ollama.parse_tags((FIX / "ollama_tags.html").read_text())
    assert tags[0]["tag"] == "latest"
    sizes = {t["tag"]: t["size"] for t in tags}
    assert all(v and v > 100_000_000 for v in sizes.values()), sizes
    assert tags[0]["context"] == "40K"


def test_cloud_only_models_are_skipped():
    assert not ollama.is_local_model({"name": "x", "sizes": [], "downloads": 0})
    assert ollama.is_local_model({"name": "x", "sizes": [], "downloads": 5})


def test_parse_helpers():
    assert parse_size("abc0b12 • 5.2GB • 40K") == 5_200_000_000
    assert parse_size("e0b9") is None
    assert parse_params("qwen3:8b") == 8e9
    assert parse_params("Qwen3-30B-A3B") == 30e9
    assert parse_params("Kokoro-82M") == 82e6
    assert parse_params("whisper-large-v3") is None


def test_gguf_variants_group_shards_and_skip_mmproj():
    sib = [{"rfilename": "m-Q4_K_M.gguf", "size": 100}, {"rfilename": "m-Q8_0-00001-of-00002.gguf", "size": 60},
           {"rfilename": "m-Q8_0-00002-of-00002.gguf", "size": 60}, {"rfilename": "mmproj-F16.gguf", "size": 5},
           {"rfilename": "README.md", "size": 1}]
    v = {g["quant"]: g for g in hf.gguf_variants(sib)}
    assert set(v) == {"Q4_K_M", "Q8_0"}
    assert v["Q8_0"]["size"] == 120 and len(v["Q8_0"]["files"]) == 2


def test_diffusers_files_prefers_fp16_and_skips_root_checkpoints():
    sib = [{"rfilename": "model_index.json", "size": 1},
           {"rfilename": "sd_xl_base_1.0.safetensors", "size": 7000},
           {"rfilename": "unet/config.json", "size": 1},
           {"rfilename": "unet/diffusion_pytorch_model.safetensors", "size": 1000},
           {"rfilename": "unet/diffusion_pytorch_model.fp16.safetensors", "size": 500},
           {"rfilename": "vae/diffusion_pytorch_model.safetensors", "size": 80},
           {"rfilename": "text_encoder/model.fp16.safetensors", "size": 120}]
    sel = hf.diffusers_files(sib)
    assert "sd_xl_base_1.0.safetensors" not in sel["files"]
    assert "unet/diffusion_pytorch_model.fp16.safetensors" in sel["files"]
    assert "unet/diffusion_pytorch_model.safetensors" not in sel["files"]
    assert "vae/diffusion_pytorch_model.safetensors" in sel["files"]  # no fp16 variant → plain file
    assert sel["variant"] == "fp16" and sel["largest"] == 500


def test_resolver():
    assert resolve_hf("text", "a/b-GGUF", "gguf", [], 75).engine == "ollama"
    assert resolve_hf("text", "a/b", "transformers", ["safetensors"], 75).engine == "vllm"
    assert resolve_hf("text", "a/b", "transformers", ["safetensors"], 60).engine == "transformers"
    assert resolve_hf("text", "a/b", "transformers", ["safetensors"], None, accelerator="tpu").engine == "vllm"
    assert resolve_hf("text", "a/b", "mlx", [], 75).engine is None
    assert resolve_hf("image", "a/b-lora", "diffusers", ["lora"], 75).engine is None
    assert resolve_hf("stt", "openai/whisper-small", "transformers", ["whisper"], 75).engine == "transformers_asr"
    assert resolve_hf("stt", "Systran/faster-whisper-tiny", "ctranslate2", [], 75).engine == "faster_whisper"
    r = resolve_hf("tts", "x/custom-tts", "transformers", [], 75)
    assert r.engine == "transformers_tts" and r.experimental


def test_context_length_reads_real_configs():
    assert context_length(None) is None
    assert context_length({"max_position_embeddings": 32768}) == 32768
    # Qwen2.5-1M: the declared max already is the served window.
    assert context_length({"max_position_embeddings": 1010000}) == 1010000
    # DeepSeek-V3 style: yarn scaling already folded into the declared max; 40× must not apply.
    deepseek = {"max_position_embeddings": 163840,
                "rope_scaling": {"type": "yarn", "factor": 40, "original_max_position_embeddings": 4096}}
    assert context_length(deepseek) == 163840
    # longchat style: base window plus a linear factor, with no original_max to mark the extension.
    longchat = {"max_position_embeddings": 2048, "max_sequence_length": 16384,
                "rope_scaling": {"type": "linear", "factor": 8}}
    assert context_length(longchat) == 16384
    # Multimodal configs keep the window in the text stack.
    assert context_length({"vision_config": {"depth": 24},
                           "text_config": {"max_position_embeddings": 262144}}) == 262144


def test_ollama_context_labels_parse_to_tokens():
    assert ollama.parse_context("128K") == 131072
    assert ollama.parse_context("1M") == 1048576
    assert ollama.parse_context("4096") == 4096
    assert ollama.parse_context(None) is None and ollama.parse_context("x") is None


def test_attached_scan(tmp_path):
    (tmp_path / "llm").mkdir()
    (tmp_path / "llm" / "model-Q4_K_M.gguf").write_bytes(b"x" * 10)
    sd = tmp_path / "sdxl" / "pytorch" / "default" / "1"
    (sd / "unet").mkdir(parents=True)
    (sd / "model_index.json").write_text('{"_class_name": "StableDiffusionXLPipeline"}')
    (sd / "unet" / "diffusion_pytorch_model.safetensors").write_bytes(b"x" * 20)
    wh = tmp_path / "whisper"
    wh.mkdir()
    (wh / "config.json").write_text('{"architectures": ["WhisperForConditionalGeneration"], "model_type": "whisper"}')
    found = {f["modality"]: f for f in scan(tmp_path)}
    assert found["text"]["engine"] == "ollama" and found["text"]["size"] == 10
    assert found["image"]["engine"] == "diffusers" and found["image"]["size"] == 20
    assert found["stt"]["engine"] == "transformers_asr"


def test_ollama_effective_param_sizes():
    from kaggleui.catalog.service import CatalogService

    it = CatalogService(lambda: None)._from_ollama({"name": "gemma4", "sizes": ["e4b", "12b"], "downloads": 1})
    assert [v["params"] for v in it["variants"]] == [4e9, 12e9]


def test_ollama_show_context():
    from kaggleui.engines.ollama import model_context

    assert model_context({"model_info": {"general.architecture": "qwen3", "qwen3.context_length": 40960}}) == 40960
    assert model_context({"model_info": {"llama.context_length": 131072}}) == 131072
    assert model_context({}) is None
