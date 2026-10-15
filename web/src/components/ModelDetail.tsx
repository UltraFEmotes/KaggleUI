import { useQuery } from "@tanstack/react-query";
import { AlertTriangle, ExternalLink, Loader } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router";

import { api } from "../lib/api";
import { bytes, cx, ENGINE_LABEL, memoryPool, params, tokens } from "../lib/format";
import { useLive } from "../lib/live";
import type { CatalogItem, DeploymentSpec, Variant } from "../lib/types";
import { FitGauge } from "./meters";
import { ModelLogo } from "./ModelLogo";
import { CapabilityTags } from "./CapabilityTags";
import { effortLabel } from "../lib/capabilities";
import { Button, Field, Input, Select, Toggle, VerdictTag } from "./ui";

export interface ModelChoice {
  item: CatalogItem;
  spec: DeploymentSpec;
}

const CTX = [2048, 4096, 8192, 16384, 32768, 65536];

/** Selector options: the standard steps up to this model's window, then the window itself. */
function ctxOptions(max?: number | null): number[] {
  if (!max) return CTX;
  return [...CTX.filter((c) => c < max), max];
}

/** Engine options shown per modality. Values are merged into spec.options. */
function EngineOptions({
  item,
  options,
  onChange,
}: {
  item: Pick<CatalogItem, "modality" | "engine" | "source" | "custom_code" | "context_length">;
  options: Record<string, any>;
  onChange: (o: Record<string, any>) => void;
}) {
  const set = (k: string, v: unknown) => onChange({ ...options, [k]: v === "" ? undefined : v });
  const m = item.modality;
  if (m === "text") {
    const fallback = Math.min(8192, item.context_length ?? 8192);
    return (
      <div className="grid gap-3 sm:grid-cols-2">
        <Field
          label="Context length"
          hint={
            item.context_length
              ? `This model supports up to ${tokens(item.context_length)} tokens. Longer context uses more VRAM for the KV cache.`
              : "Longer context uses more VRAM for the KV cache."
          }
        >
          <Select value={String(options.ctx ?? fallback)} onChange={(e) => set("ctx", Number(e.target.value))}>
            {ctxOptions(item.context_length).map((c) => (
              <option key={c} value={c}>
                {c.toLocaleString()} tokens
              </option>
            ))}
          </Select>
        </Field>
        {item.engine === "vllm" && (
          <Field label="GPU memory use" hint="Fraction of VRAM vLLM reserves.">
            <Select value={String(options.gpu_memory_utilization ?? 0.9)} onChange={(e) => set("gpu_memory_utilization", Number(e.target.value))}>
              {[0.8, 0.85, 0.9, 0.95].map((v) => (
                <option key={v} value={v}>
                  {Math.round(v * 100)}%
                </option>
              ))}
            </Select>
          </Field>
        )}
        {item.engine === "sglang" && (
          <Field label="GPU memory use" hint="Fraction of VRAM SGLang reserves for weights and the KV cache.">
            <Select value={String(options.mem_fraction_static ?? 0.88)} onChange={(e) => set("mem_fraction_static", Number(e.target.value))}>
              {[0.7, 0.8, 0.85, 0.88, 0.9, 0.95].map((v) => (
                <option key={v} value={v}>
                  {Math.round(v * 100)}%
                </option>
              ))}
            </Select>
          </Field>
        )}
      </div>
    );
  }
  if (m === "image" || m === "video" || (m === "world" && item.engine === "diffusers_world")) {
    return (
      <div className="grid gap-3 sm:grid-cols-2">
        <Field label="Memory mode" hint="Offload keeps idle parts in RAM: slower, fits bigger models.">
          <Select value={String(options.offload ?? (m === "video" ? "model" : ""))} onChange={(e) => set("offload", e.target.value)}>
            <option value="">All on GPU (fastest)</option>
            <option value="model">Model offload (balanced)</option>
            <option value="sequential">Sequential offload (smallest, slowest)</option>
          </Select>
        </Field>
        <Field label="Weights" hint="4-bit shrinks big models like FLUX to fit a T4.">
          <Select value={String(options.quantize ?? "")} onChange={(e) => set("quantize", e.target.value)}>
            <option value="">fp16 (full quality)</option>
            <option value="nf4">4-bit nf4 (bitsandbytes)</option>
          </Select>
        </Field>
        <Field label="Default steps">
          <Input type="number" min={1} max={100} value={options.steps ?? ""} placeholder="model default" onChange={(e) => set("steps", e.target.value ? Number(e.target.value) : "")} />
        </Field>
        {m === "image" ? (
          <Field label="Default guidance">
            <Input type="number" step={0.5} min={0} max={20} value={options.guidance ?? ""} placeholder="model default" onChange={(e) => set("guidance", e.target.value ? Number(e.target.value) : "")} />
          </Field>
        ) : (
          <Field label="Default frames">
            <Input type="number" min={5} max={241} value={options.frames ?? ""} placeholder="model default" onChange={(e) => set("frames", e.target.value ? Number(e.target.value) : "")} />
          </Field>
        )}
      </div>
    );
  }
  if (m === "world") {
    if (options.recipe === "hunyuan_world_1" || options.recipe === "hy_world_2") {
      return <p className="text-[13px] text-fg-2">Scene description, panorama input and reconstruction settings are available in Studio after launch.</p>;
    }
    return (
      <div className="grid gap-3 sm:grid-cols-2">
        <Field label="Default frames">
          <Input type="number" min={4} max={600} value={options.frames ?? ""} placeholder="recipe default" onChange={(e) => set("frames", e.target.value ? Number(e.target.value) : "")} />
        </Field>
        <Field label="Default steps">
          <Input type="number" min={1} max={50} value={options.steps ?? ""} placeholder="recipe default" onChange={(e) => set("steps", e.target.value ? Number(e.target.value) : "")} />
        </Field>
      </div>
    );
  }
  return <p className="text-[13px] text-fg-3">No extra settings for this engine.</p>;
}

export function OptionsForm(props: Parameters<typeof EngineOptions>[0]) {
  const { item, options, onChange } = props;
  const customCode = ["vllm", "sglang", "transformers", "transformers_tts", "transformers_audio", "transformers_asr"].includes(item.engine ?? "");
  return <div className="space-y-4">
    <EngineOptions {...props} />
    {customCode && <div className="space-y-3 border-t border-line pt-4">
      <Toggle checked={options.trust_remote_code === true}
        onChange={v => onChange({ ...options, trust_remote_code: v })}
        label="Allow model custom code"
        hint={item.custom_code ? "This checkpoint declares custom code. Enable only if you trust its source; its Python code runs in your notebook." : "For checkpoints that need their own Python code. Enable only for a source you trust."} />
      {item.source === "hf" && <Field label="Model revision" hint="Optional branch, tag or commit SHA. Use a commit SHA to pin the model and its code.">
        <Input value={options.revision ?? ""} placeholder="main" onChange={e => onChange({ ...options, revision: e.target.value || undefined })} />
      </Field>}
    </div>}
  </div>;
}

function VariantRow({ v, active, onPick }: { v: Variant; active: boolean; onPick: () => void }) {
  return (
    <button
      type="button"
      onClick={onPick}
      className={cx(
        "flex w-full items-center gap-3 rounded-lg border px-3 py-2 text-left transition-colors",
        active ? "border-signal bg-signal-soft" : "border-line hover:border-line-2",
      )}
    >
      <span className={cx("size-3 shrink-0 rounded-full border-2", active ? "border-signal bg-signal" : "border-line-2")} />
      <span className="min-w-0 flex-1">
        <span className="block truncate font-mono text-[12.5px]">{v.label}</span>
        {v.fit?.reasons?.[0] && <span className="block truncate text-[11.5px] text-fg-3">{v.fit.reasons[0]}</span>}
      </span>
      <span className="font-mono text-[11.5px] text-fg-2">{bytes(v.size)}</span>
      {v.fit && <VerdictTag verdict={v.fit.verdict} size="sm" />}
    </button>
  );
}

export function ModelDetail({
  item: base,
  actionLabel,
  onAction,
  busy,
  secondary,
  onImport,
}: {
  item: CatalogItem;
  actionLabel: string;
  onAction: (choice: ModelChoice) => void;
  busy?: boolean;
  secondary?: (choice: ModelChoice) => React.ReactNode;
  onImport?: (url: string) => void;
}) {
  const { overview } = useLive();
  const pool = overview ? memoryPool(overview.system) : null;
  const devices = pool?.devices ?? [];
  const canDetail = base.source === "hf" || base.source === "ollama";
  const detail = useQuery({
    queryKey: ["detail", base.key],
    queryFn: () => api.detail(base.key),
    enabled: canDetail,
    staleTime: 10 * 60_000,
  });
  const loaded = detail.data;
  // "Best" is relative to the list it came from; keep it when the detail view agrees the model fits well.
  const item =
    loaded && base.fit.verdict === "best" && loaded.fit.verdict === "good" ? { ...loaded, fit: { ...loaded.fit, verdict: "best" as const } } : (loaded ?? base);
  const [variant, setVariant] = useState<string | null>(base.variant);
  const [engine, setEngine] = useState<string | null>(base.engine);
  const [options, setOptions] = useState<Record<string, any>>({ ...(base.spec?.options ?? {}) });
  const [name, setName] = useState("");

  useEffect(() => {
    if (detail.data) {
      setVariant((v) => v ?? detail.data.variant);
      setOptions((o) => ({ ...(detail.data.spec?.options ?? {}), ...o }));
    }
  }, [detail.data]);

  const engines = useMemo(() => {
    if (item.modality === "text" && item.source === "hf" && item.engine !== "ollama")
      return pool?.tpu ? ["vllm", "transformers"] : ["vllm", "sglang", "transformers"];
    return item.engine ? [item.engine] : [];
  }, [item, pool?.tpu]);

  const picked = item.variants.find((v) => v.id === variant);
  const choice: ModelChoice | null = item.spec
    ? {
        item,
        spec: {
          ...item.spec,
          model: { ...item.spec.model, variant: variant ?? item.spec.model.variant, engine: engine ?? item.spec.model.engine },
          name: name.trim() || null,
          options: Object.fromEntries(
            Object.entries<unknown>({
              ...options,
              // Placement hint for the GPU scheduler; a different variant than the default changes it.
              vram_estimate:
                picked?.size && picked.id !== item.variant ? Math.round(picked.size * 1.12 + 0.8e9) : (item.fit.vram_needed ?? undefined),
            }).filter(([, v]) => v !== undefined && v !== ""),
          ),
        },
      }
    : null;

  return (
    <div className="flex min-h-full flex-col">
      <div className="space-y-6 p-5 sm:p-6">
        <div>
          <div className="mb-2 flex flex-wrap items-center gap-2">
            {item.engine ? <VerdictTag verdict={item.fit.verdict} /> : <span className="text-[12px] text-fg-2">Runner unavailable</span>}
            {item.experimental && <span className="rounded-[5px] bg-tight-soft px-1.5 py-px text-[11px] font-medium text-tight">Experimental</span>}
            {detail.isFetching && <Loader className="size-3.5 animate-spin text-fg-3" />}
          </div>
          <div className="flex items-start gap-4"><ModelLogo id={item.id} name={item.name} source={item.source} className="size-12" /><h2 className="display min-w-0 break-words text-[32px] leading-tight">{item.name}</h2></div>
          <div className="mt-2 flex flex-wrap items-center gap-x-3 gap-y-1 font-mono text-[11.5px] text-fg-2">
            <span>{item.id}</span>
            {item.brand && <span>{item.brand.label}</span>}
            {item.params ? <span>{params(item.params)} params</span> : null}
            {item.context_length ? <span>{tokens(item.context_length)} context</span> : null}
            {item.engine_label && <span>{item.engine_label}</span>}
            {item.url && (
              <a href={item.url} target="_blank" rel="noreferrer" className="inline-flex items-center gap-1 text-fg-2 hover:text-signal">
                page <ExternalLink className="size-3" />
              </a>
            )}
          </div>
          {item.description && <p className="mt-3 text-[14px] text-fg-2">{item.description}</p>}
          <div className="mt-4"><CapabilityTags capabilities={item.capabilities} /></div>
          {item.reasoning?.supported && <p className="mt-2 text-[12px] text-fg-3">Reasoning: {item.reasoning.efforts.length ? ["Auto", ...item.reasoning.efforts.map(effortLabel)].join(" · ") : "Model default"}. {item.reasoning.source === "family" ? "Estimated from the model family; verified by the runner when loaded." : "Detected from model metadata or its chat template."}</p>}
          {!!item.runtime_repos?.length && <div className="mt-3 text-[12px] text-fg-3"><p>The download estimate includes these base checkpoints:</p><div className="mt-1 flex flex-wrap gap-x-3 gap-y-1">{item.runtime_repos.map(repo => <a key={repo} href={`https://huggingface.co/${repo}`} target="_blank" rel="noreferrer" className="text-fg-2 hover:text-signal hover:underline">{repo}</a>)}</div></div>}
        </div>

        {!item.engine ? (
          <div className="rounded-xl border border-line bg-bg-2 p-4">
            <div className="mb-2 text-[14px] font-medium">Runner not available yet</div>
            <p className="text-[13px] text-fg-2">This model is published on Hugging Face. KaggleUI doesn't have a runtime for it yet, so it can't be deployed from this panel.</p>
            {item.url && <a href={item.url} target="_blank" rel="noreferrer" className="mt-3 inline-flex items-center gap-1.5 text-[13px] text-signal hover:underline">Open model documentation <ExternalLink className="size-3.5" /></a>}
            {item.url && onImport && <div className="mt-3"><Button onClick={() => onImport(item.url!)}>Download model files</Button></div>}
          </div>
        ) : <div className="rounded-xl border border-line bg-bg-2 p-4">
          <div className="readout mb-3">Fit on this machine</div>
          <FitGauge fit={item.fit} devices={devices} />
          <ul className="mt-3 space-y-1 text-[13px]">
            {item.fit.reasons.map((r) => (
              <li key={r} className="text-fg-2">
                {r}
              </li>
            ))}
            {item.fit.warnings.map((w) => (
              <li key={w} className="flex items-start gap-1.5 text-tight">
                <AlertTriangle className="mt-0.5 size-3.5 shrink-0" />
                <span>
                  {w}
                  {w.startsWith("Gated") && (
                    <Link to="/access#secrets" className="ml-2 whitespace-nowrap font-medium underline underline-offset-4">
                      Add token →
                    </Link>
                  )}
                </span>
              </li>
            ))}
          </ul>
          {Object.keys(item.fit.suggest ?? {}).length > 0 && (
            <div className="mt-3 font-mono text-[11px] text-fg-3">auto-applied: {Object.entries(item.fit.suggest).map(([k, v]) => `${k}=${v}`).join(", ")}</div>
          )}
        </div>}

        {item.variants.length > 1 && (
          <div>
            <div className="readout mb-2">{item.source === "ollama" ? "Size / tag" : "Quantization"}</div>
            <div className="max-h-[300px] space-y-1.5 overflow-y-auto pr-1 scroll-thin">
              {item.variants.map((v) => (
                <VariantRow key={v.id} v={v} active={v.id === (variant ?? item.variant)} onPick={() => setVariant(v.id)} />
              ))}
            </div>
            {picked?.context && <div className="mt-2 font-mono text-[11px] text-fg-3">context window {picked.context}</div>}
          </div>
        )}

        {engines.length > 1 && (
          <Field label="Engine" hint="vLLM and SGLang are the fast runners (experimental on T4s); Transformers is the safe fallback.">
            <Select value={engine ?? ""} onChange={(e) => setEngine(e.target.value)}>
              {engines.map((e) => (
                <option key={e} value={e}>
                  {ENGINE_LABEL[e] ?? e}
                </option>
              ))}
            </Select>
          </Field>
        )}

        {item.engine && (
          <div>
            <div className="readout mb-2">Settings</div>
            <OptionsForm
              item={{ ...item, engine: engine ?? item.engine, context_length: picked?.context_length ?? item.context_length }}
              options={options}
              onChange={setOptions}
            />
          </div>
        )}

        {item.engine && (
          <Field label="API model name (optional)" hint="What clients send as 'model'. Defaults to a short name.">
            <Input value={name} onChange={(e) => setName(e.target.value)} placeholder={item.source === "ollama" ? `${item.id}:${variant ?? "latest"}` : item.id.split("/").pop()?.toLowerCase()} />
          </Field>
        )}
      </div>

      <div className="sticky bottom-0 mt-auto flex flex-wrap items-center justify-end gap-2 border-t border-line bg-panel/95 px-5 py-4 backdrop-blur sm:px-6">
        {!item.engine ? (
          <span className="mr-auto text-[13px] text-fg-3">This architecture needs a model-specific runner.</span>
        ) : (
          <>
            {choice && secondary?.(choice)}
            <button
              disabled={!choice || busy}
              onClick={() => choice && onAction(choice)}
              className="inline-flex h-11 items-center gap-2 rounded-lg bg-signal px-5 text-[15px] font-semibold text-signal-ink transition hover:brightness-110 disabled:opacity-50"
            >
              {busy && <Loader className="size-4 animate-spin" />}
              {actionLabel}
            </button>
          </>
        )}
      </div>
    </div>
  );
}
