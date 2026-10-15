"""Map Hugging Face metadata to a modality and the KaggleUI engine that can run it."""

from __future__ import annotations

from dataclasses import dataclass

from . import world

MODALITY_PIPELINES: dict[str, list[str]] = {
    "text": ["text-generation", "image-text-to-text"],
    "image": ["text-to-image"],
    "video": ["text-to-video", "image-to-video"],
    "tts": ["text-to-speech", "text-to-audio"],
    "audio": ["text-to-audio"],
    "stt": ["automatic-speech-recognition"],
    "world": [],
    "mesh": ["image-to-3d", "text-to-3d"],
}

PIPELINE_TO_MODALITY = {p: m for m, ps in MODALITY_PIPELINES.items() for p in ps}
PIPELINE_TO_MODALITY.update({"image-to-image": "image", "text2text-generation": "text"})

ENGINE_LABELS = {
    "ollama": "Ollama (llama.cpp)",
    "vllm": "vLLM",
    "sglang": "SGLang",
    "transformers": "Transformers",
    "diffusers": "Diffusers",
    "diffusers_video": "Diffusers (video)",
    "diffusers_world": "Diffusers (world)",
    "kokoro": "Kokoro",
    "chatterbox": "Chatterbox",
    "transformers_tts": "Transformers TTS",
    "transformers_audio": "Transformers audio",
    "faster_whisper": "faster-whisper",
    "transformers_asr": "Transformers ASR",
    "world_recipe": "World recipe",
    "mock": "Mock (dev)",
}

# Engines each modality can use, in order of preference (used for the engine picker in the UI).
MODALITY_ENGINES = {
    "text": ["ollama", "vllm", "sglang", "transformers"],
    "image": ["diffusers"],
    "video": ["diffusers_video"],
    "tts": ["kokoro", "chatterbox", "transformers_tts"],
    "audio": ["transformers_audio"],
    "stt": ["faster_whisper", "transformers_asr"],
    "world": ["world_recipe", "diffusers_world"],
    "mesh": [],
}


# model_type tags the generic transformers pipelines are known to handle.
TTS_TYPES = {"vits", "bark", "speecht5", "seamless_m4t", "seamless_m4t_v2",
             "fastspeech2_conformer", "csm", "dia"}
ASR_TYPES = {"whisper", "wav2vec2", "wav2vec2-bert", "wav2vec2-conformer", "hubert", "wavlm", "moonshine",
             "parakeet_ctc", "speech_to_text", "seamless_m4t", "seamless_m4t_v2", "data2vec-audio",
             "unispeech", "unispeech-sat", "mctct", "granite_speech", "kyutai_speech_to_text", "voxtral"}


@dataclass
class Resolution:
    engine: str | None
    experimental: bool = False
    note: str | None = None


def resolve_hf(modality: str, repo_id: str, library: str | None, tags: list[str] | None,
               min_sm: int | None, accelerator: str = "gpu") -> Resolution:
    rid = repo_id.lower()
    tagset = {t.lower() for t in (tags or [])}
    lib = (library or "").lower()

    if modality == "mesh":
        return Resolution(None, note="3D checkpoint: browse and download; this architecture has no configured KaggleUI runner.")
    if modality == "audio":
        if "midashenglm-gen" in rid:
            return Resolution("transformers_audio", experimental=True,
                              note="Audio scenes, music, effects and speech. Requires Allow model custom code.")
        if "musicgen" in rid and lib in ("transformers", ""):
            return Resolution("transformers_audio", experimental=True, note="Text-conditioned music generation.")
        return Resolution(None, note="Audio generation checkpoint; this architecture needs a dedicated runner.")

    if modality == "world":
        entry = world.profile(repo_id)
        if entry and entry.get("engine"):
            return Resolution(entry["engine"], experimental=True, note=entry.get("blurb"))
        return Resolution(None, note="Published world model; a KaggleUI launch integration is needed")

    if modality == "text":
        if lib == "gguf" or "gguf" in tagset:
            return Resolution("ollama")
        if lib in ("mlx", "onnx", "openvino", "coreml") or "mlx" in tagset:
            return Resolution(None, note=f"{lib or 'This'} format isn't supported on Kaggle GPUs")
        if lib in ("transformers", "") and ("safetensors" in tagset or lib == "transformers"):
            if accelerator == "tpu":
                return Resolution("vllm", experimental=True, note="Runs on the TPU through vLLM's TPU backend")
            if (min_sm or 0) >= 70:
                return Resolution("vllm", experimental=True, note="vLLM on T4 is experimental; Transformers is the fallback")
            return Resolution("transformers")
        return Resolution(None, note=f"Library '{lib or 'unknown'}' isn't supported yet")

    if modality in ("image", "video") and ("lora" in tagset or rid.endswith(("-lora", "_lora"))):
        return Resolution(None, note="LoRA adapter, not a full model")

    if modality == "image":
        if lib == "diffusers" or "diffusers" in tagset:
            return Resolution("diffusers")
        return Resolution(None, note="Only diffusers-format image models are supported")

    if modality == "video":
        if lib == "diffusers" or "diffusers" in tagset:
            return Resolution("diffusers_video")
        return Resolution(None, note="Only diffusers-format video models are supported")

    if modality == "tts":
        if "vibevoice" in rid and not any(kind in rid for kind in ("asr", "streaming", "realtime")):
            return Resolution("transformers_tts", experimental=True,
                              note="Dedicated VibeVoice TTS runner with an isolated compatible environment.")
        if "kokoro" in rid:
            return Resolution("kokoro")
        if "chatterbox" in rid:
            return Resolution("chatterbox", experimental=True)
        if lib == "transformers":
            if tagset & TTS_TYPES:
                return Resolution("transformers_tts")
            return Resolution("transformers_tts", experimental=True,
                              note="Custom architecture: the generic TTS pipeline may not support it")
        return Resolution(None, note=f"TTS library '{lib or 'unknown'}' isn't supported yet")

    if modality == "stt":
        if lib == "ctranslate2" or "faster-whisper" in rid:
            return Resolution("faster_whisper")
        if lib == "transformers":
            if tagset & ASR_TYPES or "whisper" in rid:
                return Resolution("transformers_asr")
            return Resolution("transformers_asr", experimental=True,
                              note="Custom architecture: the generic ASR pipeline may not support it")
        return Resolution(None, note=f"ASR library '{lib or 'unknown'}' isn't supported yet")

    return Resolution(None, note="No supported runtime for this model")
