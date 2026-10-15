import { Brain, Eye, Layers, Wrench, X } from "lucide-react";

import { BRAND_DEFINITIONS, CAPABILITY_FILTERS, CONTEXT_PRESETS, type CatalogFiltersState, parameterError } from "../lib/catalogFilters";
import { cx } from "../lib/format";
import type { CatalogBrand, Modality, SearchResult } from "../lib/types";
import { Input, Select, Segmented, Toggle } from "./ui";

const ICONS = { vision: Eye, tools: Wrench, reasoning: Brain, embedding: Layers };
const RANGES = [
  { label: "Up to 1B", min: "", max: "1" }, { label: "1–8B", min: "1", max: "8" },
  { label: "8–32B", min: "8", max: "32" }, { label: "32–100B", min: "32", max: "100" },
  { label: "100B+", min: "100", max: "" },
];

interface Props {
  value: CatalogFiltersState;
  onChange: (value: CatalogFiltersState) => void;
  modality: Modality;
  facets?: SearchResult["facets"];
}

function brandChoices(facets: Props["facets"], selected: string): CatalogBrand[] {
  const brands = new Map<string, CatalogBrand>(BRAND_DEFINITIONS.map(brand => [brand.id, brand]));
  for (const brand of facets?.brands ?? []) brands.set(brand.id, brand);
  if (selected && !brands.has(selected)) brands.set(selected, { id: selected, label: selected.replace(/^publisher:/, ""), logo: null });
  return [...brands.values()].sort((a, b) => a.label.localeCompare(b.label));
}

export function CatalogFilterPanel({ value, onChange, modality, facets }: Props) {
  const set = (patch: Partial<CatalogFiltersState>) => onChange({ ...value, ...patch });
  const brands = brandChoices(facets, value.brand);
  const error = parameterError(value);
  const engine = value.engines[modality] ?? "";
  const engines = facets?.engines ?? [];
  return <section id="catalog-filter-panel" aria-label="Model filters" className="catalog-filter-panel">
    <div className="catalog-filter-fields">
      <fieldset className="min-w-0">
        <legend className="mb-2 text-[13px] font-medium">Parameter range</legend>
        <div className="flex items-center gap-2">
          <label className="min-w-0 flex-1"><span className="mb-1 block text-[11px] text-fg-3">Minimum (B)</span>
            <Input type="number" min="0" step="any" inputMode="decimal" value={value.min} placeholder="No minimum"
              aria-invalid={!!error} aria-describedby="parameter-filter-help" onChange={e => set({ min: e.target.value })} /></label>
          <span className="mt-5 text-fg-3" aria-hidden="true">–</span>
          <label className="min-w-0 flex-1"><span className="mb-1 block text-[11px] text-fg-3">Maximum (B)</span>
            <Input type="number" min="0" step="any" inputMode="decimal" value={value.max} placeholder="No maximum"
              aria-invalid={!!error} aria-describedby="parameter-filter-help" onChange={e => set({ max: e.target.value })} /></label>
        </div>
        <p id="parameter-filter-help" className={cx("mt-2 text-[11px]", error ? "text-worst" : "text-fg-3")} role={error ? "alert" : undefined}>
          {error ?? "B = billion parameters. 0.5B = 500M. Limits are inclusive."}
        </p>
        <div className="mt-3 flex flex-wrap gap-1.5" aria-label="Parameter presets">{RANGES.map(range => <button key={range.label} type="button"
          className="catalog-filter-preset" aria-pressed={value.min === range.min && value.max === range.max}
          onClick={() => set(value.min === range.min && value.max === range.max ? { min: "", max: "" } : { min: range.min, max: range.max })}>{range.label}</button>)}</div>
        <label className="mt-3 flex items-center gap-2 text-[12px] text-fg-2"><input className="size-3.5 accent-signal" type="checkbox" checked={value.includeUnknown}
          onChange={e => set({ includeUnknown: e.target.checked })} /> Include unknown parameter counts</label>
        {modality === "text" && <div className="mt-4">
          <span className="mb-2 block text-[13px] font-medium">Context window</span>
          <div className="flex flex-wrap gap-1.5" aria-label="Minimum context window">
            <button type="button" className="catalog-filter-preset" aria-pressed={value.minCtx === ""}
              onClick={() => set({ minCtx: "" })}>Any</button>
            {CONTEXT_PRESETS.map(preset => <button key={preset.value} type="button" className="catalog-filter-preset"
              aria-pressed={value.minCtx === preset.value}
              onClick={() => set({ minCtx: value.minCtx === preset.value ? "" : preset.value })}>≥ {preset.label}</button>)}
          </div>
          <p className="mt-2 text-[11px] leading-relaxed text-fg-3">The shortest context window a model must handle. Models with unknown windows are hidden while this is set; open the list to load more metadata.</p>
        </div>}
      </fieldset>
      <fieldset className="min-w-0">
        <legend className="mb-2 text-[13px] font-medium">Capabilities</legend>
        {modality === "text" ? <>
          <div className="flex flex-wrap gap-2">{CAPABILITY_FILTERS.map(cap => {
            const Icon = ICONS[cap.id as keyof typeof ICONS];
            const checked = value.capabilities.includes(cap.id);
            return <button key={cap.id} type="button" className="catalog-capability-filter" aria-pressed={checked}
              onClick={() => set({ capabilities: checked ? value.capabilities.filter(id => id !== cap.id) : [...value.capabilities, cap.id].sort() })}>
              <Icon className="size-3.5" strokeWidth={1.6} />{cap.label}
            </button>;
          })}</div>
          <div className="mt-3"><Segmented size="sm" value={value.match} onChange={match => set({ match })}
            options={[{ value: "all", label: "Match all" }, { value: "any", label: "Match any" }]} /></div>
          <p className="mt-2 text-[11px] leading-relaxed text-fg-3">{value.capabilities.length > 1 ? value.match === "all" ? "Requires every selected capability." : "Requires at least one selected capability." : "Choose one or more capabilities."} Uses detected model tags.</p>
        </> : <p className="text-[12px] leading-relaxed text-fg-3">Vision, tools and reasoning filters are available in the Text catalog.</p>}
      </fieldset>
      <div className="space-y-3">
        <label className="block"><span className="mb-2 block text-[13px] font-medium">Brand / model family</span>
          <Select value={value.brand} onChange={e => set({ brand: e.target.value })}><option value="">All brands</option>
            {brands.map(brand => <option key={brand.id} value={brand.id}>{brand.label}{brand.count ? ` (${brand.count})` : ""}</option>)}
          </Select></label>
        <label className="block"><span className="mb-1.5 block text-[12px] text-fg-2">Runner</span>
          <Select value={engine} onChange={e => set({ engines: { ...value.engines, [modality]: e.target.value } })}><option value="">Any runner</option>
            {engine && !engines.some(entry => entry.id === engine) && <option value={engine}>{engine}</option>}
            {engines.map(entry => <option key={entry.id} value={entry.id}>{entry.label} ({entry.count})</option>)}
          </Select></label>
      </div>
    </div>
    <div className="mt-4 flex flex-wrap gap-x-6 gap-y-3 border-t border-line pt-3">
      <Toggle checked={value.hideGated} onChange={hideGated => set({ hideGated })} label={<span className="text-[12px] font-normal">No gated weights</span>} />
      <Toggle checked={value.hideExperimental} onChange={hideExperimental => set({ hideExperimental })} label={<span className="text-[12px] font-normal">Hide experimental runners</span>} />
    </div>
  </section>;
}

export function activeFilters({ value, modality, facets, onChange }: Props) {
  const set = (patch: Partial<CatalogFiltersState>) => onChange({ ...value, ...patch });
  const chips: { label: string; remove: () => void }[] = [];
  if (value.min !== "" || value.max !== "") chips.push({ label: `${value.min || "0"}–${value.max || "∞"}B parameters${value.includeUnknown ? " + unknown" : ""}`, remove: () => set({ min: "", max: "" }) });
  if (modality === "text" && value.minCtx) {
    const preset = CONTEXT_PRESETS.find(entry => entry.value === value.minCtx);
    chips.push({ label: `≥ ${preset?.label ?? value.minCtx} context`, remove: () => set({ minCtx: "" }) });
  }
  if (modality === "text") for (const cap of CAPABILITY_FILTERS.filter(cap => value.capabilities.includes(cap.id))) {
    chips.push({ label: cap.label, remove: () => set({ capabilities: value.capabilities.filter(id => id !== cap.id) }) });
  }
  if (value.brand) chips.push({ label: brandChoices(facets, value.brand).find(brand => brand.id === value.brand)?.label ?? value.brand, remove: () => set({ brand: "" }) });
  const engine = value.engines[modality];
  if (engine) chips.push({ label: facets?.engines.find(entry => entry.id === engine)?.label ?? engine, remove: () => set({ engines: { ...value.engines, [modality]: "" } }) });
  if (value.hideGated) chips.push({ label: "No gated weights", remove: () => set({ hideGated: false }) });
  if (value.hideExperimental) chips.push({ label: "No experimental runners", remove: () => set({ hideExperimental: false }) });
  return chips;
}

export function ActiveCatalogFilters({ chips }: { chips: ReturnType<typeof activeFilters> }) {
  if (!chips.length) return null;
  return <div className="mb-3 flex flex-wrap items-center gap-2" aria-label="Active model filters">{chips.map(chip => <button key={chip.label} type="button" className="catalog-filter-chip"
    aria-label={`Remove ${chip.label} filter`} onClick={chip.remove}>{chip.label}<X className="size-3" /></button>)}</div>;
}
