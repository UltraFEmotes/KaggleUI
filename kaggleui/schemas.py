"""Shared data models: deployment specs, runtime settings and launch presets."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from .catalog.categories import audio_family

Modality = Literal["text", "image", "video", "tts", "audio", "stt", "world", "mesh"]
MODALITIES: tuple[str, ...] = ("text", "image", "video", "tts", "audio", "stt", "world", "mesh")

InterfaceKind = Literal["studio", "gradio", "openwebui", "comfyui", "none"]
TunnelProvider = Literal["cloudflare", "cloudflare_named", "ngrok", "none"]


class ModelRef(BaseModel):
    source: Literal["hf", "ollama", "attached", "curated"] = "hf"
    id: str
    """HF repo id, Ollama model name (``qwen3:8b``), or a local path for attached models."""
    variant: str | None = None
    """GGUF quant (``Q4_K_M``), Ollama tag, or diffusers dtype variant."""
    engine: str
    """Engine that runs it: ollama, vllm, sglang, transformers, diffusers, diffusers_video, diffusers_world,
    kokoro, chatterbox, transformers_tts, faster_whisper, transformers_asr, world_recipe, mock."""


class DeploymentSpec(BaseModel):
    modality: Modality
    model: ModelRef
    name: str | None = None
    """Name clients use in the ``model`` field of API requests. Defaults to a short model name."""
    interface: InterfaceKind = "studio"
    options: dict[str, Any] = Field(default_factory=dict)
    """Engine options, e.g. ``{"ctx": 8192}``, ``{"offload": "model"}``, ``{"dtype": "float16"}``."""
    gpus: list[int] | None = None
    """Pin to specific GPU indices; normally chosen automatically from free VRAM."""

    @model_validator(mode="after")
    def classify_audio(self):
        if self.modality == "tts" and audio_family(self.model.id):
            self.modality = "audio"
            if self.model.engine == "transformers_tts":
                self.model.engine = "transformers_audio"
        return self


class AccessSettings(BaseModel):
    password_enabled: bool = True


class ApiSettings(BaseModel):
    enabled: bool = True
    openai: bool = True
    anthropic: bool = True
    ollama: bool = False
    require_key: bool = True


class ExposureSettings(BaseModel):
    provider: TunnelProvider = "cloudflare"


class Settings(BaseModel):
    access: AccessSettings = Field(default_factory=AccessSettings)
    api: ApiSettings = Field(default_factory=ApiSettings)
    exposure: ExposureSettings = Field(default_factory=ExposureSettings)


class Preset(BaseModel):
    """Everything needed to reproduce a setup without clicking: settings + deployments."""

    version: int = 1
    settings: Settings = Field(default_factory=Settings)
    deployments: list[DeploymentSpec] = Field(default_factory=list)


class SettingsPatch(BaseModel):
    access: dict[str, Any] | None = None
    api: dict[str, Any] | None = None
    exposure: dict[str, Any] | None = None
