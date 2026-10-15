import brandDefinitions from "../../../kaggleui/catalog/brands.json";
import type { CatalogBrand, Modality } from "./types";

export const CAPABILITY_FILTERS = [
  { id: "vision", label: "Vision" }, { id: "tools", label: "Tools" },
  { id: "reasoning", label: "Reasoning" }, { id: "embedding", label: "Embeddings" },
];
/** Minimum context window presets, in tokens. */
export const CONTEXT_PRESETS = [
  { label: "8K", value: "8192" },
  { label: "32K", value: "32768" },
  { label: "128K", value: "131072" },
  { label: "256K", value: "262144" },
  { label: "1M", value: "1048576" },
];
export const BRAND_DEFINITIONS = brandDefinitions;
const STORAGE_KEY = "kaggleui-catalog-filters";

export interface CatalogFiltersState {
  min: string;
  max: string;
  minCtx: string;
  capabilities: string[];
  match: "all" | "any";
  brand: string;
  engines: Partial<Record<Modality, string>>;
  includeUnknown: boolean;
  hideGated: boolean;
  hideExperimental: boolean;
}

export function emptyFilters(): CatalogFiltersState {
  return { min: "", max: "", minCtx: "", capabilities: [], match: "all", brand: "", engines: {},
    includeUnknown: false, hideGated: false, hideExperimental: false };
}

export function storedFilters(): CatalogFiltersState {
  const base = emptyFilters();
  try {
    const value = JSON.parse(localStorage.getItem(STORAGE_KEY) ?? "null");
    if (!value || typeof value !== "object") return base;
    for (const key of ["min", "max", "brand"] as const) if (typeof value[key] === "string") base[key] = value[key].slice(0, 120);
    if (CONTEXT_PRESETS.some(preset => preset.value === value.minCtx)) base.minCtx = value.minCtx;
    base.capabilities = CAPABILITY_FILTERS.filter(cap => Array.isArray(value.capabilities) && value.capabilities.includes(cap.id)).map(cap => cap.id);
    base.match = value.match === "any" ? "any" : "all";
    for (const key of ["includeUnknown", "hideGated", "hideExperimental"] as const) base[key] = value[key] === true;
    for (const modality of ["text", "image", "video", "tts", "audio", "stt", "world", "mesh"] as const) {
      if (typeof value.engines?.[modality] === "string") base.engines[modality] = value.engines[modality].slice(0, 80);
    }
  } catch { /* storage unavailable */ }
  return base;
}

export function saveFilters(filters: CatalogFiltersState) {
  try { localStorage.setItem(STORAGE_KEY, JSON.stringify(filters)); } catch { /* storage unavailable */ }
}

export function parameterError(filters: CatalogFiltersState): string | null {
  const values = [filters.min, filters.max];
  if (values.some(value => value !== "" && (!Number.isFinite(Number(value) * 1e9) || Number(value) < 0))) return "Enter a valid, non-negative parameter count.";
  if (filters.min !== "" && filters.max !== "" && Number(filters.min) > Number(filters.max)) return "Minimum parameters must be less than or equal to the maximum.";
  return null;
}

export function filterRequest(filters: CatalogFiltersState, modality: Modality) {
  return {
    ...(filters.min !== "" ? { min_params: Number(filters.min) * 1e9 } : {}),
    ...(filters.max !== "" ? { max_params: Number(filters.max) * 1e9 } : {}),
    ...(filters.minCtx !== "" && modality === "text" ? { min_ctx: Number(filters.minCtx) } : {}),
    capabilities: modality === "text" ? filters.capabilities : [],
    capability_match: filters.match, brand: filters.brand, engine: filters.engines[modality] ?? "",
    include_unknown: filters.includeUnknown, hide_gated: filters.hideGated, hide_experimental: filters.hideExperimental,
  };
}

export function inferBrand(id: string, name = "", source = "hf"): CatalogBrand | undefined {
  const owner = source === "hf" && id.includes("/") ? id.split("/")[0].toLowerCase() : "";
  const entry = BRAND_DEFINITIONS.find(brand => brand.owners.includes(owner)) ??
    BRAND_DEFINITIONS.find(brand => new RegExp(brand.pattern, "i").test(`${id} ${name}`));
  return entry ? { id: entry.id, label: entry.label, logo: entry.logo } : undefined;
}
