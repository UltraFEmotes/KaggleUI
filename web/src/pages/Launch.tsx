import {
  AppWindow,
  AudioLines,
  Blocks,
  Film,
  Globe,
  Image as ImageIcon,
  LayoutDashboard,
  MessageSquare,
  Mic,
  Plug,
  ArrowRight,
  Check,
  Workflow,
  Box,
  Music2,
} from "lucide-react";
import { useEffect, useState, type ReactNode } from "react";
import { useLocation, useNavigate } from "react-router";

import { FitGauge } from "../components/meters";
import { ModelBrowser } from "../components/ModelBrowser";
import { ModelDetail, type ModelChoice } from "../components/ModelDetail";
import { AccessForm, ApiForm, ExposurePicker, OptionCard, StreamingNote, type ApiState } from "../components/settings";
import { Button, Drawer, SectionTitle, Toggle, useToast } from "../components/ui";
import { api } from "../lib/api";
import { bytes, cx, ENGINE_LABEL, memoryPool, MODALITY_LABEL } from "../lib/format";
import { useLive } from "../lib/live";
import type { CatalogItem, DeploymentSpec, Modality, TunnelProvider } from "../lib/types";

export const MODALITY_ICON: Record<Modality, typeof MessageSquare> = {
  text: MessageSquare,
  image: ImageIcon,
  video: Film,
  tts: AudioLines,
  audio: Music2,
  mesh: Box,
  stt: Mic,
  world: Globe,
};

const TYPES: { m: Modality; blurb: string; engines: string; experimental?: boolean }[] = [
  { m: "text", blurb: "Chat, code, reasoning, embeddings.", engines: "Ollama · vLLM · Transformers" },
  { m: "image", blurb: "Text-to-image diffusion models.", engines: "Diffusers · ComfyUI" },
  { m: "video", blurb: "Text or image to short clips.", engines: "Diffusers", experimental: true },
  { m: "tts", blurb: "Natural voices, some with cloning.", engines: "Kokoro · Chatterbox · Transformers" },
  { m: "audio", blurb: "Generate music, sound effects and audio scenes.", engines: "MusicGen · MiDashengLM-Gen", experimental: true },
  { m: "mesh", blurb: "Discover image and text to 3D checkpoints.", engines: "TRELLIS · Hunyuan3D · TripoSR" },
  { m: "stt", blurb: "Transcribe and translate audio.", engines: "faster-whisper · Transformers" },
  { m: "world", blurb: "Action-conditioned generated worlds.", engines: "Recipes (Oasis, Matrix-Game)", experimental: true },
];

type Iface = NonNullable<DeploymentSpec["interface"]>;

function interfaceOptions(m: Modality): { value: Iface; title: string; desc: string; icon: ReactNode; badge?: string }[] {
  const opts: { value: Iface; title: string; desc: string; icon: ReactNode; badge?: string }[] = [
    { value: "studio", title: "KaggleUI Studio", desc: "Built-in playground. Streams over WebSockets, so it works through any tunnel.", icon: <LayoutDashboard className="size-4" />, badge: "default" },
    { value: "gradio", title: "Gradio", desc: "Auto-generated Gradio app, with an optional public gradio.live link.", icon: <AppWindow className="size-4" /> },
  ];
  if (m === "text") opts.push({ value: "openwebui", title: "Open WebUI", desc: "Full ChatGPT-style app: chat history, RAG, files. First start takes a few minutes.", icon: <Blocks className="size-4" /> });
  if (m === "image" || m === "video") opts.push({ value: "comfyui", title: "ComfyUI", desc: "Node workflows plus ComfyUI-Manager. Replaces the diffusers engine; no /v1/images API.", icon: <Workflow className="size-4" /> });
  opts.push({ value: "none", title: "API only", desc: "No UI, just the endpoints.", icon: <Plug className="size-4" /> });
  return opts;
}

function Step({
  n,
  title,
  summary,
  active,
  done,
  onEdit,
  children,
}: {
  n: number;
  title: string;
  summary?: ReactNode;
  active: boolean;
  done: boolean;
  onEdit: () => void;
  children: ReactNode;
}) {
  return (
    <section className={cx("panel overflow-hidden transition-[border-color]", active ? "border-line-2" : "")}>
      <button
        type="button"
        onClick={onEdit}
        disabled={active}
        className="flex w-full items-center gap-4 px-5 py-4 text-left disabled:cursor-default"
      >
        <span
          className={cx(
            "grid size-7 shrink-0 place-items-center rounded-full border text-[12px] font-medium",
            active ? "border-signal/50 bg-signal-soft text-signal" : done ? "border-line-2 text-signal" : "border-line text-fg-3",
          )}
        >
          {done ? <Check className="size-3.5" /> : n}
        </span>
        <span className="min-w-0 flex-1">
          <span className={cx("block text-[16px] font-semibold", !active && !done && "text-fg-3")}>{title}</span>
          {!active && summary && <span className="mt-0.5 block truncate text-[13px] text-fg-2">{summary}</span>}
        </span>
        {!active && done && <span className="text-[12.5px] text-fg-3 hover:text-fg">Change</span>}
      </button>
      {active && <div className="border-t border-line px-5 pb-5 pt-5 animate-[rise_.3s_ease_both]">{children}</div>}
    </section>
  );
}

function SheetRow({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="flex items-baseline justify-between gap-4 border-b border-line py-2.5 last:border-0">
      <span className="readout shrink-0">{label}</span>
      <span className="min-w-0 truncate text-right text-[13.5px]">{children}</span>
    </div>
  );
}

export default function Launch() {
  const { overview, dispatch } = useLive();
  const nav = useNavigate();
  const loc = useLocation();
  const toast = useToast();
  const incoming = (loc.state as { choice?: ModelChoice } | null)?.choice;

  const [step, setStep] = useState(incoming ? 2 : 0);
  const [modality, setModality] = useState<Modality>(incoming?.item.modality ?? "text");
  const [choice, setChoice] = useState<ModelChoice | null>(incoming ?? null);
  const [open, setOpen] = useState<CatalogItem | null>(null);
  const [iface, setIface] = useState<Iface>("studio");
  const [share, setShare] = useState(false);
  const [exposure, setExposure] = useState<TunnelProvider>("cloudflare");
  const [locked, setLocked] = useState(true);
  const [apiState, setApiState] = useState<ApiState>({ enabled: true, openai: true, anthropic: true, ollama: false, require_key: true });
  const [busy, setBusy] = useState(false);
  const [seeded, setSeeded] = useState(false);

  useEffect(() => {
    if (overview && !seeded) {
      setExposure(overview.settings.exposure.provider);
      setLocked(overview.settings.access.password_enabled);
      setApiState(overview.settings.api);
      setSeeded(true);
    }
  }, [overview, seeded]);

  useEffect(() => {
    if (!interfaceOptions(modality).some((o) => o.value === iface)) setIface("studio");
  }, [modality, iface]);

  const devices = overview ? memoryPool(overview.system).devices : [];
  const ifaceLabel = interfaceOptions(modality).find((o) => o.value === iface)?.title ?? iface;
  const surfaces = (["openai", "anthropic", "ollama"] as const).filter((k) => apiState[k]);

  async function launch() {
    if (!choice || !overview) return;
    setBusy(true);
    try {
      const s = overview.settings;
      const patch: Record<string, unknown> = {};
      if (s.access.password_enabled !== locked) patch.access = { password_enabled: locked };
      if (JSON.stringify(s.api) !== JSON.stringify(apiState)) patch.api = apiState;
      if (s.exposure.provider !== exposure) patch.exposure = { provider: exposure };
      if (Object.keys(patch).length) {
        const settings = await api.updateSettings(patch);
        dispatch({ type: "settings", data: settings });
      }
      const spec: DeploymentSpec = {
        ...choice.spec,
        interface: iface,
        options: { ...(choice.spec.options ?? {}), ...(iface === "gradio" && share ? { gradio_share: true } : {}) },
      };
      const dep = await api.deploy(spec);
      toast(`Launching ${dep.name}…`);
      nav(`/deployments#${dep.id}`);
    } catch (e) {
      toast(e instanceof Error ? e.message : "Launch failed", "error");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div>
      <SectionTitle
        title="New deployment"
        description="Pick a model and how you want to reach it. KaggleUI checks it against this machine, then installs, downloads and starts it."
      />

      <div className="grid items-start gap-6 xl:grid-cols-[minmax(0,1fr)_288px]">
        <div className="space-y-3">
          <Step n={1} title="Model type" summary={MODALITY_LABEL[modality]} active={step === 0} done={step > 0} onEdit={() => setStep(0)}>
            <div className="grid gap-2 sm:grid-cols-2">
              {TYPES.map(({ m, blurb, engines, experimental }) => {
                const Icon = MODALITY_ICON[m];
                const active = m === modality;
                return (
                  <button
                    key={m}
                    type="button"
                    onClick={() => {
                      if (m !== modality) setChoice(null);
                      setModality(m);
                      setStep(1);
                    }}
                    className={cx(
                      "group relative flex flex-col items-start rounded-lg border px-4 py-3.5 text-left transition-[border-color,background]",
                      active ? "border-signal/50 bg-signal-soft" : "border-line hover:border-line-2 hover:bg-panel-2",
                    )}
                  >
                    <span className="mb-1.5 flex items-center gap-2.5">
                      <Icon className={cx("size-4", active ? "text-signal" : "text-fg-2")} strokeWidth={1.6} />
                      <span className="text-[14px] font-semibold">{MODALITY_LABEL[m]}</span>
                    </span>
                    <span className="text-[12.5px] text-fg-2">{blurb}</span>
                    <span className="mt-2 text-[11px] text-fg-3">{engines}</span>
                    {experimental && (
                      <span className="absolute right-3 top-3 text-[10px] text-tight">
                        Experimental
                      </span>
                    )}
                  </button>
                );
              })}
            </div>
          </Step>

          <Step
            n={2}
            title="Pick a model"
            summary={choice ? `${choice.item.name}${choice.spec.model.variant ? ` · ${choice.spec.model.variant}` : ""}` : undefined}
            active={step === 1}
            done={!!choice && step > 1}
            onEdit={() => setStep(1)}
          >
            <ModelBrowser modality={modality} lockModality onOpen={setOpen} selectedKey={choice?.item.key} />
          </Step>

          <Step
            n={3}
            title="Interface & exposure"
            summary={`${ifaceLabel} · ${overview?.tunnels.find((t) => t.provider === exposure)?.label ?? exposure.replace("_", " ")}`}
            active={step === 2}
            done={step > 2}
            onEdit={() => choice && setStep(2)}
          >
            <div className="readout mb-2">Interface</div>
            <div className="grid gap-2 md:grid-cols-2">
              {interfaceOptions(modality).map((o) => (
                <OptionCard
                  key={o.value}
                  active={iface === o.value}
                  onClick={() => setIface(o.value)}
                  icon={o.icon}
                  title={o.title}
                  badge={o.badge ? <span className="rounded-[5px] bg-panel-3 px-1.5 py-px text-[11px] font-medium text-fg-3">{o.badge}</span> : undefined}
                >
                  {o.desc}
                </OptionCard>
              ))}
            </div>
            {iface === "gradio" && (
              <div className="mt-3">
                <Toggle checked={share} onChange={setShare} label="Also create a gradio.live share link" hint="Gradio's own public relay. It streams well; the link expires after 72h." />
              </div>
            )}
            <div className="readout mb-2 mt-6">Public URL</div>
            <ExposurePicker value={exposure} onChange={setExposure} />
            <div className="mt-5 flex justify-end">
              <Button variant="primary" onClick={() => setStep(3)}>
                Continue
              </Button>
            </div>
          </Step>

          <Step n={4} title="Access" summary={locked ? "Password-locked" : "Open to anyone with the link"} active={step === 3} done={step > 3} onEdit={() => choice && setStep(3)}>
            <AccessForm enabled={locked} onChange={setLocked} />
            <div className="mt-5 flex justify-end">
              <Button variant="primary" onClick={() => setStep(4)}>
                Continue
              </Button>
            </div>
          </Step>

          <Step
            n={5}
            title="API"
            summary={apiState.enabled ? `${surfaces.join(" + ") || "no surfaces"}${apiState.require_key ? " · key required" : " · no key"}` : "Off"}
            active={step === 4}
            done={step > 4}
            onEdit={() => choice && setStep(4)}
          >
            <ApiForm value={apiState} onChange={setApiState} />
            <div className="mt-4">
              <StreamingNote />
            </div>
          </Step>
        </div>

        {/* Build sheet */}
        <aside className="panel sticky top-6 p-5">
          <div className="mb-3 flex items-center justify-between">
            <span className="text-[14px] font-semibold">Deployment summary</span>
            <span className="font-mono text-[10px] text-fg-3">{overview?.system.accelerator_label}</span>
          </div>
          <SheetRow label="Type">{MODALITY_LABEL[modality]}</SheetRow>
          <SheetRow label="Model">{choice ? choice.item.name : <span className="text-fg-3">not chosen</span>}</SheetRow>
          {choice?.spec.model.variant && <SheetRow label="Variant">{choice.spec.model.variant}</SheetRow>}
          {choice && <SheetRow label="Engine">{ENGINE_LABEL[choice.spec.model.engine ?? ""] ?? choice.item.engine_label ?? choice.spec.model.engine}</SheetRow>}
          {choice && (
            <div className="border-b border-dashed border-line py-3">
              <div className="mb-2 flex items-baseline justify-between">
                <span className="readout">Footprint</span>
                <span className="font-mono text-[11px] text-fg-2">{bytes(choice.item.size)} download</span>
              </div>
              <FitGauge fit={choice.item.fit} devices={devices} />
            </div>
          )}
          <SheetRow label="Interface">{ifaceLabel}</SheetRow>
          <SheetRow label="URL">{exposure === "cloudflare" ? "Quick tunnel" : exposure === "cloudflare_named" ? "Named tunnel" : exposure === "ngrok" ? "ngrok" : "Local only"}</SheetRow>
          <SheetRow label="Password">{locked ? "On" : "Off"}</SheetRow>
          <SheetRow label="API">{apiState.enabled ? surfaces.join(", ") || "none" : "Off"}</SheetRow>
          <Button variant="primary" size="lg" className="mt-5 w-full" disabled={!choice} busy={busy} onClick={launch}>
            Launch model <ArrowRight className="size-4" />
          </Button>
          {!choice && <p className="mt-2 text-center text-[12px] text-fg-3">Pick a model to launch.</p>}
        </aside>
      </div>

      <Drawer open={!!open} onClose={() => setOpen(null)} title={open ? `${MODALITY_LABEL[open.modality]} model` : ""}>
        {open && (
          <ModelDetail
            key={open.key}
            item={open}
            actionLabel="Use this model"
            onAction={(c) => {
              setChoice(c);
              setOpen(null);
              setStep(2);
            }}
          />
        )}
      </Drawer>
    </div>
  );
}
