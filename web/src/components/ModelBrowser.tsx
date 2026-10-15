import { useQuery } from "@tanstack/react-query";
import { AlertTriangle, Download, HardDrive, Heart, LayoutGrid, List, Search, SlidersHorizontal, Star, X } from "lucide-react";
import { useEffect, useMemo, useState } from "react";

import { api } from "../lib/api";
import { bytes, compact, cx, memoryPool, MODALITY_LABEL, params, tokens } from "../lib/format";
import { useLive } from "../lib/live";
import type { CatalogItem, Gpu, Modality } from "../lib/types";
import { FitGauge } from "./meters";
import { ModelLogo } from "./ModelLogo";
import { CapabilityTags } from "./CapabilityTags";
import { ActiveCatalogFilters, activeFilters, CatalogFilterPanel } from "./CatalogFilters";
import { emptyFilters, filterRequest, parameterError, saveFilters, storedFilters } from "../lib/catalogFilters";
import { Button, Chip, Empty, Input, Segmented, Select, Toggle, VerdictTag } from "./ui";

export const MODALITY_ORDER: Modality[] = ["text", "image", "video", "tts", "audio", "stt", "world", "mesh"];

function useDebounced<T>(value: T, ms: number): T {
  const [v, setV] = useState(value);
  useEffect(() => {
    const t = window.setTimeout(() => setV(value), ms);
    return () => window.clearTimeout(t);
  }, [value, ms]);
  return v;
}

function SourceBadge({ item }: { item: CatalogItem }) {
  const label = item.source === "hf" ? "HF" : item.source === "ollama" ? "Ollama" : item.capabilities.includes("imported") ? "Imported" : "Attached";
  return (
    <span className="inline-flex items-center gap-1.5">
      <span className={cx("rounded-[5px] border px-1.5 font-mono text-[10.5px] leading-[18px]", item.source === "attached" ? "border-info/40 text-info" : "border-line-2 text-fg-3")}>
        {label}
      </span>
      {item.curated && (
        <span className="inline-flex items-center gap-1 text-[11.5px] font-medium text-fg-2" title="Picked and tested on Kaggle GPUs">
          <Star className="size-3 fill-current" />
          Pick
        </span>
      )}
    </span>
  );
}

const VERDICT_FILL: Record<string, string> = { best: "var(--good)", good: "var(--good)", tight: "var(--tight)", worst: "var(--worst)" };

/** Compact fit indicator: estimated VRAM as a share of what this machine can give it. */
function FitBar({ item }: { item: CatalogItem }) {
  const need = item.fit.vram_needed ?? 0;
  const cap = item.fit.capacity ?? 0;
  const frac = need && cap ? Math.min(need / cap, 1) : 0;
  return (
    <div className="mt-1.5 h-[3px] w-full overflow-hidden rounded-full bg-panel-3" aria-hidden>
      <div className="h-full rounded-full" style={{ width: `${Math.max(frac * 100, need ? 4 : 0)}%`, background: VERDICT_FILL[item.fit.verdict] }} />
    </div>
  );
}

export function ModelCard({
  item,
  devices,
  selected,
  onOpen,
}: {
  item: CatalogItem;
  devices: Gpu[];
  selected?: boolean;
  onOpen: () => void;
}) {
  const f = item.fit;
  const unsupported = !item.engine;
  return (
    <button
      type="button"
      onClick={onOpen}
      className={cx(
        "panel group flex h-full w-full min-w-0 flex-col p-5 text-left transition-[border-color,background,transform] duration-150 hover:-translate-y-px hover:border-line-2 hover:bg-panel-2",
        selected && "border-fg ring-1 ring-fg",
      )}
      aria-pressed={!!selected}
    >
      <div className="mb-2.5 flex flex-wrap items-start justify-between gap-2">
        <SourceBadge item={item} />
        {unsupported ? <Chip>Runner unavailable</Chip> : <VerdictTag verdict={f.verdict} size="sm" />}
      </div>
      <div className="mb-1 flex items-baseline gap-2">
        <ModelLogo id={item.id} name={item.name} source={item.source} />
        <h3 className="line-clamp-2 min-w-0 break-words text-[16px] font-semibold leading-snug tracking-[-0.015em] [overflow-wrap:anywhere]">{item.name}</h3>
      </div>
      {item.source === "hf" && item.name !== item.id && <div className="mb-1 truncate font-mono text-[10.5px] text-fg-3">{item.id}</div>}
      <p className="mb-3 line-clamp-2 min-h-[2.6em] text-[13px] leading-snug text-fg-2">{item.description || " "}</p>
      <div className="mb-3 flex flex-wrap gap-1.5">
        {item.variant && <Chip>{item.variant}</Chip>}
        {item.params && params(item.params).toLowerCase() !== (item.variant ?? "").toLowerCase() ? <Chip>{params(item.params)}</Chip> : null}
        {tokens(item.context_length) ? <Chip>{tokens(item.context_length)} ctx</Chip> : null}
        {item.size ? (
          <Chip>
            <HardDrive className="size-3" />
            {bytes(item.size)}
          </Chip>
        ) : null}
        <CapabilityTags capabilities={item.capabilities} />
        {item.experimental && <Chip tone="warn">experimental</Chip>}
      </div>
      <div className="mt-auto">
        {unsupported ? <p className="text-[12px] text-fg-3">View the published model and its requirements.</p> : <FitGauge fit={f} devices={devices} />}
        <div className="mt-2 flex items-start justify-between gap-2">
          <p className="line-clamp-2 text-[12px] text-fg-2">{unsupported ? "This architecture needs a model-specific runner." : f.reasons[0]}</p>
          {f.warnings.length > 0 && (
            <span title={f.warnings.join("\n")} className="mt-0.5 shrink-0 text-tight">
              <AlertTriangle className="size-3.5" />
            </span>
          )}
        </div>
        {(item.downloads > 0 || item.likes > 0) && (
          <div className="mt-2 flex gap-3 font-mono text-[10.5px] text-fg-3">
            {item.downloads > 0 && (
              <span className="flex items-center gap-1">
                <Download className="size-3" />
                {compact(item.downloads)}
              </span>
            )}
            {item.likes > 0 && (
              <span className="flex items-center gap-1">
                <Heart className="size-3" />
                {compact(item.likes)}
              </span>
            )}
          </div>
        )}
      </div>
    </button>
  );
}

function ModelRow({ item, selected, onOpen, showContext }: { item: CatalogItem; selected: boolean; onOpen: () => void; showContext: boolean }) {
  return (
    <button type="button" className={cx("model-row model-columns", showContext && "has-ctx")} onClick={onOpen} aria-pressed={selected}>
      <div className="flex min-w-0 items-start gap-3.5">
        <ModelLogo id={item.id} name={item.name} source={item.source} className="mt-px" />
        <div className="min-w-0">
          <div className="flex flex-wrap items-center gap-x-2.5 gap-y-1">
            <h3 className="min-w-0 break-words text-[14.5px] font-medium tracking-[-0.01em] [overflow-wrap:anywhere]">{item.name}</h3>
            <SourceBadge item={item} />
          </div>
          <p className="mt-0.5 line-clamp-1 text-[13px] text-fg-3">{item.description || item.id}</p>
          {(item.variant || item.capabilities.length > 0 || item.experimental) && (
            <div className="mt-2 flex flex-wrap items-center gap-1.5 text-[11.5px] text-fg-3">
              {item.variant && <span className="font-mono">{item.variant}</span>}
              <CapabilityTags capabilities={item.capabilities} />
              {item.experimental && <span className="text-tight">Experimental</span>}
            </div>
          )}
        </div>
      </div>
      <div className="model-stats">
        <div>
          <span className="block font-mono text-[13px] text-fg">{params(item.params) || "—"}</span>
          <span className="text-[11.5px] text-fg-3 @[760px]:hidden">Parameters</span>
        </div>
        <div>
          <span className="block font-mono text-[13px] text-fg-2">{bytes(item.size)}</span>
          <span className="text-[11.5px] text-fg-3 @[760px]:hidden">Download</span>
        </div>
        {showContext && (
          <div>
            <span className="block font-mono text-[13px] text-fg-2" title={item.context_length ? `${item.context_length.toLocaleString()} tokens` : undefined}>
              {tokens(item.context_length) || "—"}
            </span>
            <span className="text-[11.5px] text-fg-3 @[760px]:hidden">Context</span>
          </div>
        )}
        <div className="model-fit min-w-[150px]">
          {item.engine ? (
            <>
              <div className="flex items-baseline justify-between gap-2">
                <VerdictTag verdict={item.fit.verdict} size="sm" />
                <span className="font-mono text-[11.5px] text-fg-3">{item.fit.vram_needed ? `${bytes(item.fit.vram_needed)}` : "?"}</span>
              </div>
              <FitBar item={item} />
            </>
          ) : (
            <>
              <span className="text-[12px] font-medium text-fg-3">No runner yet</span>
              <span className="mt-0.5 block text-[11.5px] text-fg-3">View model details</span>
            </>
          )}
        </div>
      </div>
    </button>
  );
}

type FitFilter = "all" | "fits";
const FIT_KEY = "kaggleui-fit-filter";
const PAGE = 40;

function storedFit(): FitFilter {
  try {
    return localStorage.getItem(FIT_KEY) === "fits" ? "fits" : "all";
  } catch {
    return "all";
  }
}

const SORTS = [
  { value: "recommended", label: "Best fit" },
  { value: "trending", label: "Trending" },
  { value: "downloads", label: "Downloads" },
  { value: "newest", label: "Newest" },
  { value: "params_asc", label: "Lowest parameters" },
  { value: "params_desc", label: "Highest parameters" },
];

function storedSort(): string | null {
  try {
    const value = localStorage.getItem("kaggleui-catalog-sort");
    return SORTS.some((option) => option.value === value) ? value : null;
  } catch { return null; }
}

function storedView(): "list" | "grid" {
  try { return localStorage.getItem("kaggleui-catalog-view") === "grid" ? "grid" : "list"; }
  catch { return "list"; }
}

export function ModelBrowser({
  modality,
  onModality,
  onOpen,
  selectedKey,
  lockModality,
  initialSource = "all",
}: {
  modality: Modality;
  onModality?: (m: Modality) => void;
  onOpen: (item: CatalogItem) => void;
  selectedKey?: string | null;
  lockModality?: boolean;
  initialSource?: string;
}) {
  const { overview } = useLive();
  const pool = overview ? memoryPool(overview.system) : null;
  const devices = pool?.devices ?? [];
  const [q, setQ] = useState("");
  const [source, setSource] = useState(initialSource);
  const broaderCatalog = ["world", "mesh", "audio"].includes(modality);
  const [generalFit, setGeneralFit] = useState<FitFilter>(() => initialSource === "attached" ? "all" : storedFit());
  const [discoveryFits, setDiscoveryFits] = useState<Partial<Record<Modality, FitFilter>>>({});
  const fitFilter = broaderCatalog ? discoveryFits[modality] ?? "all" : generalFit;
  // "All models" reads best by popularity (big models mixed in); "Fits" by how well they fit.
  const [sortChoice, setSortChoice] = useState<string | null>(storedSort);
  const [view, setView] = useState(storedView);
  const sort = sortChoice ?? (modality === "world" || fitFilter === "fits" ? "recommended" : "trending");
  const [unsupported, setUnsupported] = useState(false);
  const [launchableFilters, setLaunchableFilters] = useState<Partial<Record<Modality, boolean>>>({});
  const launchableOnly = launchableFilters[modality] ?? false;
  const setLaunchableOnly = (value: boolean) => setLaunchableFilters(previous => ({ ...previous, [modality]: value }));
  const showUnsupported = broaderCatalog ? !launchableOnly : unsupported;
  const [filters, setFilters] = useState(storedFilters);
  const [filtersOpen, setFiltersOpen] = useState(false);
  const delayedFilters = useDebounced(filters, 300);
  const requestedFilters = useMemo(() => filterRequest(delayedFilters, modality), [delayedFilters, modality]);
  const filterError = parameterError(filters);
  const dq = useDebounced(q, 350);
  const criteriaKey = JSON.stringify({ modality, dq, source, sort, showUnsupported, fitFilter, ...requestedFilters });
  const [page, setPage] = useState({ key: "", limit: PAGE });
  const limit = page.key === criteriaKey ? page.limit : PAGE;
  useEffect(() => saveFilters(filters), [filters]);

  function changeFit(v: FitFilter) {
    if (broaderCatalog) {
      setDiscoveryFits(previous => ({ ...previous, [modality]: v }));
      return;
    }
    setGeneralFit(v);
    try {
      localStorage.setItem(FIT_KEY, v);
    } catch {
      /* storage unavailable */
    }
  }

  function changeSort(value: string | null) {
    setSortChoice(value);
    try {
      if (value) localStorage.setItem("kaggleui-catalog-sort", value);
      else localStorage.removeItem("kaggleui-catalog-sort");
    } catch { /* storage unavailable */ }
  }

  function changeView(value: "list" | "grid") {
    setView(value);
    try { localStorage.setItem("kaggleui-catalog-view", value); }
    catch { /* storage unavailable */ }
  }

  function resetFilters() {
    setQ("");
    setSource("all");
    setUnsupported(false);
    setLaunchableOnly(false);
    changeFit("all");
    changeSort(null);
    setFilters(emptyFilters());
  }

  const preview = useQuery({
    queryKey: ["search-preview", criteriaKey, limit],
    queryFn: () => api.search({ modality, q: dq, source, sort, show_unsupported: showUnsupported, fit: fitFilter, limit, live: false, ...requestedFilters }),
    enabled: !filterError,
    staleTime: 5 * 60_000,
  });
  const liveQuery = useQuery({
    queryKey: ["search", criteriaKey, limit],
    queryFn: () => api.search({ modality, q: dq, source, sort, show_unsupported: showUnsupported, fit: fitFilter, limit, ...requestedFilters }),
    enabled: !filterError,
    staleTime: 5 * 60_000,
    placeholderData: (prev, previousQuery) => previousQuery?.queryKey[1] === criteriaKey ? prev : undefined,
  });
  const query = { ...liveQuery, data: liveQuery.data ?? preview.data,
    isLoading: liveQuery.isLoading && preview.isLoading, error: liveQuery.data || preview.data ? null : liveQuery.error };
  const loadingMore = query.isFetching && query.isPlaceholderData && (query.data?.items.length ?? 0) < limit;
  const chips = activeFilters({ value: filters, onChange: setFilters, modality, facets: query.data?.facets });

  const sources = useMemo(() => {
    const base = [
      { value: "all", label: "All" },
      { value: "curated", label: "Picks" },
      { value: "hf", label: "Hugging Face" },
    ];
    if (modality === "text") base.push({ value: "ollama", label: "Ollama" });
    base.push({ value: "attached", label: "Notebook attachments" });
    return base;
  }, [modality]);

  const errors = Object.entries(query.data?.sources ?? {}).filter(([, s]) => s.startsWith("error") || s.startsWith("partial"));

  return (
    <div className="@container">
      {!lockModality && onModality && (
        <nav className="catalog-tabs scroll-thin" aria-label="Model types">
          {MODALITY_ORDER.map((m) => (
            <button key={m} type="button" aria-pressed={modality === m} onClick={() => {
              onModality(m);
              if (["world", "mesh", "audio"].includes(m)) setSource("all");
              if (m !== "text" && source === "ollama") setSource("all");
            }}>{MODALITY_LABEL[m]}</button>
          ))}
        </nav>
      )}
      {modality === "world" && (
        <p className="mb-5 max-w-[80ch] text-[13px] text-fg-2">
          Browse published world models from Hugging Face. Models marked “Runner unavailable” can be downloaded; launch requires a runner for their architecture. GPU fit is checked separately.
        </p>
      )}
      {modality === "audio" && <p className="mb-5 max-w-[80ch] text-[13px] text-fg-2">Music, sound effects and mixed audio scenes. Voice-based speech synthesis is in Text to speech.</p>}
      {modality === "mesh" && <p className="mb-5 max-w-[80ch] text-[13px] text-fg-2">Browse text-to-3D and image-to-3D checkpoints, including TRELLIS, Hunyuan3D and TripoSR. Download from model details; deployment requires a configured runner.</p>}
      <div className="mb-4 flex flex-col gap-2.5 @2xl:flex-row @2xl:items-center">
        <div className="relative min-w-0 flex-1">
          <Search className="pointer-events-none absolute left-3 top-1/2 size-4 -translate-y-1/2 text-fg-3" aria-hidden />
          <Input
            aria-label="Search models"
            value={q}
            onChange={(e) => setQ(e.target.value)}
            placeholder={modality === "text" ? "Search models (qwen, gemma, coder…)" : `Search ${MODALITY_LABEL[modality].toLowerCase()} models`}
            className="h-10 pl-9 text-[14px]"
          />
        </div>
        <div className="flex shrink-0 flex-wrap items-center gap-2 @2xl:flex-nowrap">
          <Button className="h-10" variant="outline" onClick={() => setFiltersOpen(open => !open)}
            aria-expanded={filtersOpen} aria-controls="catalog-filter-panel"><SlidersHorizontal className="size-4" /> Filters
            {chips.length > 0 && <span className="catalog-filter-count">{chips.length}</span>}</Button>
          <Select aria-label="Sort models" value={sort} onChange={(e) => changeSort(e.target.value)} className="order-last h-10 w-full @2xl:order-none @2xl:w-52">
            {SORTS.map((option) => <option key={option.value} value={option.value}>{option.label}</option>)}
          </Select>
          <div className="flex h-10 items-center gap-0.5 rounded-lg border border-line bg-bg-2 p-[3px]" role="group" aria-label="Model layout">
            {(["list", "grid"] as const).map((value) => {
              const Icon = value === "list" ? List : LayoutGrid;
              return <button key={value} type="button" onClick={() => changeView(value)} aria-label={`${value === "list" ? "List" : "Grid"} view`} aria-pressed={view === value} className={cx("grid size-8 place-items-center rounded-md transition-colors", view === value ? "bg-panel-3 text-fg shadow-[var(--edge)]" : "text-fg-3 hover:text-fg")}><Icon className="size-4" /></button>;
            })}
          </div>
        </div>
      </div>
      {filtersOpen && <CatalogFilterPanel value={filters} onChange={setFilters} modality={modality} facets={query.data?.facets} />}
      <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
        <Segmented size="sm" value={source} onChange={setSource} options={sources} />
        <div className="flex flex-wrap items-center gap-4">
          <Segmented<FitFilter>
            size="sm"
            value={fitFilter}
            onChange={changeFit}
            options={[
              { value: "all", label: "All models" },
              { value: "fits", label: "Fits this machine" },
            ]}
          />
          <Toggle checked={broaderCatalog ? launchableOnly : unsupported} onChange={broaderCatalog ? setLaunchableOnly : setUnsupported} label={<span className="text-[13px] font-normal text-fg-2">{broaderCatalog ? "Launchable only" : "Show unsupported"}</span>} />
        </div>
      </div>
      <ActiveCatalogFilters chips={chips} />
      <div className="mb-3 flex flex-wrap items-center justify-between gap-2 text-[12px] text-fg-3" role="status" aria-live="polite">
        <span>
          {filterError ? "Check the parameter range." : query.isFetching && !loadingMore ? (query.data ? "Showing available models · refreshing live catalogs…" : "Loading models…") : query.data?.counts ? `${query.data.items.length} of ${query.data.counts.total}${query.data.has_more ? "+" : ""}${chips.length ? " matching" : ""} models${broaderCatalog ? ` · ${query.data.counts.launchable ?? 0} with launch support` : ""} · ${query.data.counts.fits} fit this machine` : query.data ? `${query.data.items.length} models` : ""}
        </span>
        {(q || chips.length > 0 || source !== "all" || (broaderCatalog ? launchableOnly : unsupported) || fitFilter !== "all" || sortChoice) && (
          <button type="button" onClick={resetFilters} className="flex items-center gap-1 text-fg-2 hover:text-fg"><X className="size-3" /> Reset filters</button>
        )}
      </div>
      {liveQuery.error && preview.data && <p role="status" className="mb-4 text-[12px] text-tight">Live catalog unavailable. Showing bundled results. <button className="underline" onClick={() => liveQuery.refetch()}>Retry</button></p>}
      {(sort === "params_asc" || sort === "params_desc") && (
        <p className="mb-4 text-[12px] text-fg-3">Sorted by the shown variant's parameter count. Unknown counts appear last. Load more to include more catalog results.</p>
      )}
      {errors.length > 0 && (
        <div className="mb-4 rounded-lg border border-tight/40 bg-tight-soft px-3 py-2 text-[13px] text-tight">
          {errors.map(([k, v]) => (
            <div key={k}>
              {k}: {v}
            </div>
          ))}
        </div>
      )}
      {filterError ? <Empty title="Check parameter range">{filterError}</Empty> : query.isLoading ? (
        <div className={view === "grid" ? "grid gap-3 @xl:grid-cols-2 @5xl:grid-cols-3" : "panel overflow-hidden"} aria-label="Loading models" aria-busy="true">
          {Array.from({ length: 6 }, (_, i) => (
            <div key={i} className={cx("animate-pulse", view === "grid" ? "panel h-[248px] bg-panel-2" : "flex items-center gap-3.5 border-b border-line px-5 py-4 last:border-0")}>{view === "list" && <><span className="size-9 rounded-[10px] bg-panel-3" /><span className="flex-1 space-y-2"><span className="block h-3 w-40 rounded bg-panel-3" /><span className="block h-2.5 w-72 max-w-full rounded bg-panel-2" /></span><span className="hidden h-3 w-24 rounded bg-panel-3 sm:block" /></>}</div>
          ))}
        </div>
      ) : query.error ? (
        <Empty title="Couldn't load models">{String((query.error as Error).message)}</Empty>
      ) : query.data && query.data.items.length === 0 ? (
        <Empty title={query.isFetching ? "Searching live catalogs…" : chips.length ? "No models match these filters" : "Nothing found"}>
          {chips.length ? "Remove a filter or adjust the range. When available, Load more expands discovery. Capability filters use detected tags; unknown counts are excluded from ranges unless included." : "Try another search, or switch the source filter."}
        </Empty>
      ) : (
        <div className={cx(view === "grid" ? "grid gap-3 @xl:grid-cols-2 @5xl:grid-cols-3" : "panel overflow-hidden", query.isPlaceholderData && "opacity-60")} aria-busy={query.isFetching}>
          {view === "list" && (
            <div className={cx("model-header model-columns border-b border-line px-5 py-2.5 text-[12px] font-medium text-fg-3", modality === "text" && "has-ctx")} aria-hidden="true">
              <span>Model</span><span>Params</span><span>Download</span>{modality === "text" && <span>Context</span>}<span>{modality === "world" ? "Support / GPU fit" : pool?.tpu ? "Fit on this TPU · HBM" : "Fit on this GPU · VRAM"}</span>
            </div>
          )}
          {query.data?.items.map((it) => (
            view === "grid"
              ? <ModelCard key={it.key} item={it} devices={devices} selected={selectedKey === it.key} onOpen={() => onOpen(it)} />
              : <ModelRow key={it.key} item={it} selected={selectedKey === it.key} onOpen={() => onOpen(it)} showContext={modality === "text"} />
          ))}
        </div>
      )}
      {!filterError && query.data?.has_more && (
        <div className="mt-5 flex justify-center">
          <Button busy={query.isFetching} onClick={() => setPage({ key: criteriaKey, limit: Math.min(limit + PAGE, 300) })} disabled={limit >= 300}>
            {limit >= 300 ? "Search to narrow it down" : "Load more"}
          </Button>
        </div>
      )}
    </div>
  );
}
