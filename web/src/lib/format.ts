import type { Gpu, System } from "./types";

export function bytes(n: number | null | undefined, digits = 1): string {
  if (n == null || Number.isNaN(n)) return "—";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let v = n;
  let i = 0;
  while (Math.abs(v) >= 1000 && i < units.length - 1) {
    v /= 1000;
    i++;
  }
  return `${v.toFixed(i < 2 ? 0 : digits)} ${units[i]}`;
}

/** Context window in tokens: 131072 → "128K", 1010000 → "1.01M". */
export function tokens(n: number | null | undefined): string {
  if (!n) return "";
  if (n >= 1e6) return `${+(n / 1e6).toPrecision(3)}M`;
  if (n >= 2048) return `${Math.round(n / 1024)}K`;
  return String(n);
}

export interface MemoryPool {
  /** GPUs when the session has them, else TPU chips, else empty. */
  devices: Gpu[];
  tpu: boolean;
  total: number;
  free: number;
}

export function memoryPool(system: System): MemoryPool {
  if (system.gpus.length > 0) {
    return { devices: system.gpus, tpu: false, total: system.vram_total_all, free: system.vram_free_all };
  }
  const tpus = system.tpus ?? [];
  return { devices: tpus, tpu: tpus.length > 0, total: system.hbm_total_all ?? 0, free: system.hbm_free_all ?? 0 };
}

export function params(n: number | null | undefined): string {
  if (!n) return "";
  if (n >= 1e12) return `${(n / 1e12).toFixed(1)}T`;
  if (n >= 1e9) return `${n >= 1e10 ? Math.round(n / 1e9) : (n / 1e9).toFixed(1).replace(/\.0$/, "")}B`;
  return `${Math.round(n / 1e6)}M`;
}

export function compact(n: number | null | undefined): string {
  if (!n) return "0";
  return new Intl.NumberFormat("en", { notation: "compact", maximumFractionDigits: 1 }).format(n);
}

export function duration(seconds: number): string {
  const s = Math.max(0, Math.floor(seconds));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = s % 60;
  return `${h}:${String(m).padStart(2, "0")}:${String(sec).padStart(2, "0")}`;
}

export function ago(ts: number | null | undefined): string {
  if (!ts) return "";
  const d = Date.now() / 1000 - ts;
  if (d < 60) return "just now";
  if (d < 3600) return `${Math.floor(d / 60)}m ago`;
  if (d < 86400) return `${Math.floor(d / 3600)}h ago`;
  if (d < 86400 * 60) return `${Math.floor(d / 86400)}d ago`;
  return `${Math.floor(d / (86400 * 30))}mo ago`;
}

export const MODALITY_LABEL: Record<string, string> = {
  text: "Text",
  image: "Image",
  video: "Video",
  tts: "Text to speech",
  audio: "Audio generation",
  mesh: "3D models",
  stt: "Speech to text",
  world: "World",
};

export const VERDICT_LABEL: Record<string, string> = {
  best: "Best for this machine",
  good: "Fits",
  tight: "Tight",
  worst: "Won't fit",
};

export const SOURCE_LABEL: Record<string, string> = {
  hf: "Hugging Face",
  ollama: "Ollama",
  attached: "Attached",
};

/** Display names for the engines the UI lets you pick between. */
export const ENGINE_LABEL: Record<string, string> = {
  ollama: "Ollama (llama.cpp)",
  vllm: "vLLM",
  sglang: "SGLang",
  transformers: "Transformers",
};

export function cx(...parts: (string | false | null | undefined)[]): string {
  return parts.filter(Boolean).join(" ");
}
