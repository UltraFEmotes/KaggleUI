"""Find models already attached to the notebook (Kaggle Models / Datasets under /kaggle/input).

These need no download, which saves time and disk.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from ..capabilities import context_length, local_profile
from ..engines.model_support import model_config

SHARD_RE = re.compile(r"-\d{5}-of-\d{5}\.gguf$", re.I)


def _size(paths: list[Path]) -> int:
    total = 0
    for p in paths:
        try:
            total += p.stat().st_size
        except OSError:
            pass
    return total


def _dir_weight_size(d: Path) -> int:
    total = 0
    for root, _dirs, files in os.walk(d):
        for f in files:
            if f.endswith((".safetensors", ".bin", ".pt", ".pth")):
                try:
                    total += os.path.getsize(os.path.join(root, f))
                except OSError:
                    pass
    return total


def scan(root: Path = Path("/kaggle/input"), max_depth: int = 7) -> list[dict]:
    found: list[dict] = []
    if not root.exists():
        return found
    root_depth = len(root.parts)
    for dirpath, dirnames, filenames in os.walk(root):
        d = Path(dirpath)
        if len(d.parts) - root_depth > max_depth:
            dirnames[:] = []
            continue
        files = set(filenames)
        rel = str(d.relative_to(root))
        label = rel.replace("/", " / ")

        ggufs = [d / f for f in filenames if f.lower().endswith(".gguf") and "mmproj" not in f.lower()]
        if ggufs:
            # Split GGUFs: point at the first shard; llama.cpp loads the rest.
            firsts = [p for p in ggufs if not SHARD_RE.search(p.name) or "-00001-of-" in p.name]
            for p in firsts:
                shards = [q for q in ggufs if SHARD_RE.sub("", q.name) == SHARD_RE.sub("", p.name)] or [p]
                found.append({"path": str(p), "name": f"{label} / {p.name}", "modality": "text",
                              "engine": "ollama", "size": _size(shards)})
            dirnames[:] = []
            continue

        if {"ve.safetensors", "t3_cfg.safetensors", "s3gen.safetensors", "tokenizer.json"} <= files:
            found.append({"path": str(d), "name": label, "modality": "tts", "engine": "chatterbox",
                          "size": _dir_weight_size(d)})
            dirnames[:] = []
            continue

        if "model_index.json" in files:
            modality = "image"
            engine = None
            try:
                cls = json.loads((d / "model_index.json").read_text()).get("_class_name", "")
                if any(kind in cls for kind in ("Audio", "Music")):
                    modality = "audio"
                elif any(kind in cls for kind in ("ShapE", "Hunyuan3D", "Trellis")):
                    modality = "mesh"
                elif cls.startswith("Cosmos"):
                    modality, engine = "world", "diffusers_world"
                elif any(k in cls for k in ("MatrixGame", "LingBot", "HunyuanWorld")):
                    modality = "world"
                elif any(k in cls for k in ("Video", "I2V", "T2V", "Wan", "LTX", "CogVideo", "Hunyuan")):
                    modality = "video"
            except Exception:
                pass
            found.append({"path": str(d), "name": label, "modality": modality,
                          "engine": engine or ("diffusers" if modality == "image" else "diffusers_video" if modality == "video" else None),
                          "size": _dir_weight_size(d)})
            dirnames[:] = []
            continue

        if "config.json" in files:
            try:
                cfg = json.loads((d / "config.json").read_text())
            except Exception:
                cfg = {}
            archs = " ".join(cfg.get("architectures") or [])
            model_type = str(cfg.get("model_type", ""))
            is_ct2 = "model.bin" in files and ("vocabulary.json" in files or "vocabulary.txt" in files)
            if model_type in {"midashenglm-gen", "musicgen", "musicgen_melody"}:
                found.append({"path": str(d), "name": label, "modality": "audio", "engine": "transformers_audio",
                              "size": _dir_weight_size(d), "experimental": True, "custom_code": bool(cfg.get("auto_map"))})
                dirnames[:] = []
            elif cfg.get("plbert") and cfg.get("vocab") and any(f.endswith(".pth") for f in files):
                found.append({"path": str(d), "name": label, "modality": "tts", "engine": "kokoro",
                              "size": _dir_weight_size(d)})
                dirnames[:] = []
            elif model_type in {"vits", "bark", "speecht5", "fastspeech2_conformer", "csm", "dia", "midashenglm-gen", "vibevoice"}:
                found.append({"path": str(d), "name": label, "modality": "tts", "engine": "transformers_tts",
                              "size": _dir_weight_size(d), "experimental": True, "custom_code": bool(cfg.get("auto_map"))})
                dirnames[:] = []
            elif is_ct2 and "whisper" in (rel.lower() + model_type):
                found.append({"path": str(d), "name": label, "modality": "stt", "engine": "faster_whisper",
                              "size": _dir_weight_size(d)})
                dirnames[:] = []
            elif "Whisper" in archs or model_type == "whisper":
                found.append({"path": str(d), "name": label, "modality": "stt", "engine": "transformers_asr",
                              "size": _dir_weight_size(d)})
                dirnames[:] = []
            elif re.search(r"For(CausalLM|ConditionalGeneration)", archs):
                found.append({"path": str(d), "name": label, "modality": "text", "engine": "transformers",
                              "size": _dir_weight_size(d), "params": None})
                dirnames[:] = []
    for item in found:
        config = model_config(item["path"])
        if config.get("auto_map"):
            item["custom_code"] = True
        if item["modality"] == "text":
            item.update(local_profile(item["path"], item["name"]))
            ctx = context_length(config)
            if ctx:
                item["context_length"] = ctx
    return found
