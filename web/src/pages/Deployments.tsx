import { Check, ExternalLink, FileCode2, Plus, RotateCcw, ScrollText, SlidersHorizontal, Sparkles, Square } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { Link, useLocation } from "react-router";

import { ProgressBar } from "../components/meters";
import { Button, CopyButton, Drawer, Empty, SectionTitle, useToast } from "../components/ui";
import { api } from "../lib/api";
import { bytes, cx, duration, ENGINE_LABEL, MODALITY_LABEL } from "../lib/format";
import { useLive, useLogs } from "../lib/live";
import { ModelLogo } from "../components/ModelLogo";
import { OptionsForm } from "../components/ModelDetail";
import type { Deployment } from "../lib/types";

const STAGES = ["installing", "downloading", "loading", "ready"] as const;
const STAGE_LABEL: Record<string, string> = { installing: "Install", downloading: "Download", loading: "Load", ready: "Ready" };

function Pipeline({ d }: { d: Deployment }) {
  const idx = d.status === "queued" ? -1 : STAGES.indexOf(d.status as (typeof STAGES)[number]);
  const failed = d.status === "error";
  return (
    <ol className="flex flex-wrap items-center gap-x-1.5 gap-y-1 text-[12.5px]" aria-label="Deployment progress">
      {STAGES.map((s, i) => {
        const done = idx > i || d.status === "ready";
        const current = idx === i && d.status !== "ready";
        return (
          <li key={s} className="flex items-center gap-1.5">
            <span
              className={cx(
                "inline-flex items-center gap-1.5",
                done && "text-fg-2",
                current && !failed && "font-medium text-fg",
                current && failed && "font-medium text-worst",
                !done && !current && "text-fg-3",
              )}
            >
              {done ? (
                <Check className="size-3.5 text-good" strokeWidth={2.5} />
              ) : (
                <span className={cx("dot", current ? (failed ? "dot-bad" : "dot-warn") : "")} />
              )}
              {STAGE_LABEL[s]}
            </span>
            {i < STAGES.length - 1 && <span className={cx("h-px w-5", done ? "bg-line-2" : "bg-line")} aria-hidden />}
          </li>
        );
      })}
    </ol>
  );
}

function Logs({ id }: { id: string }) {
  const { dispatch } = useLive();
  const logs = useLogs();
  const box = useRef<HTMLDivElement>(null);
  const stick = useRef(true);
  useEffect(() => {
    api.logs(id).then((r) => dispatch({ type: "logs", id, lines: r.lines })).catch(() => {});
  }, [id, dispatch]);
  const lines = logs[id] ?? [];
  useEffect(() => {
    const el = box.current;
    if (el && stick.current) el.scrollTop = el.scrollHeight;
  }, [lines.length]);
  return (
    <div
      ref={box}
      onScroll={(e) => {
        const el = e.currentTarget;
        stick.current = el.scrollHeight - el.scrollTop - el.clientHeight < 40;
      }}
      className="max-h-[340px] overflow-auto rounded-lg border border-line bg-bg px-3 py-2.5 font-mono text-[11px] leading-[1.7] text-fg-2 scroll-thin"
    >
      {lines.length === 0 ? (
        <span className="text-fg-3">No output yet.</span>
      ) : (
        lines.map((l, i) => (
          <div key={i} className={cx("whitespace-pre-wrap break-all", /error|traceback|failed/i.test(l) && "text-worst")}>
            {l}
          </div>
        ))
      )}
    </div>
  );
}

function DeploymentCard({ d, focus }: { d: Deployment; focus: boolean }) {
  const [showLogs, setShowLogs] = useState(d.status === "error" || focus);
  const [busy, setBusy] = useState<string | null>(null);
  const [configure, setConfigure] = useState(false);
  const [options, setOptions] = useState<Record<string, unknown>>(d.spec.options ?? {});
  const toast = useToast();
  const ref = useRef<HTMLDivElement>(null);
  const [, tick] = useState(0);
  useEffect(() => {
    if (focus) ref.current?.scrollIntoView({ behavior: "smooth", block: "start" });
  }, [focus]);
  useEffect(() => {
    if (d.status !== "ready") return;
    const t = window.setInterval(() => tick((x) => x + 1), 1000);
    return () => window.clearInterval(t);
  }, [d.status]);

  const active = !["ready", "error", "stopped"].includes(d.status);
  const iface = d.interface ?? {};
  const ifaceName = iface.kind === "openwebui" ? "Open WebUI" : iface.kind === "comfyui" ? "ComfyUI" : iface.kind === "gradio" ? "Gradio" : null;

  async function act(kind: "stop" | "restart") {
    setBusy(kind);
    try {
      if (kind === "stop") await api.stop(d.id);
      else await api.restart(d.id);
    } catch (e) {
      toast(e instanceof Error ? e.message : "Failed", "error");
    } finally {
      setBusy(null);
    }
  }

  async function retry() {
    setBusy("retry");
    try {
      await api.deploy({ ...d.spec, options });
      setConfigure(false);
      toast(`Launching ${d.name} with updated settings…`);
    } catch (e) {
      toast(e instanceof Error ? e.message : "Launch failed", "error");
    } finally { setBusy(null); }
  }

  return (
    <div ref={ref} className={cx("panel scroll-mt-24 p-5 sm:p-6", focus && "border-fg-3")}>
      <div className="flex flex-wrap items-start gap-4">
        <ModelLogo id={d.spec.model.id} name={d.name} source={d.spec.model.source} className="size-10" />
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-1">
            <h3 className="text-[17px] font-semibold tracking-[-0.02em]">{d.name}</h3>
            <CopyButton text={d.name} label="Copy API model name" className="!px-1.5 [&>span]:sr-only" />
          </div>
          <div className="mt-0.5 flex flex-wrap gap-x-2 gap-y-1 text-[12.5px] text-fg-3">
            <span className="font-mono">{d.spec.model.id}</span>
            {d.spec.model.variant && <span className="font-mono">· {d.spec.model.variant}</span>}
            <span>· {MODALITY_LABEL[d.modality]} on {ENGINE_LABEL[d.spec.model.engine] ?? d.spec.model.engine}</span>
          </div>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          {d.status === "ready" && d.info?.runtime !== "comfyui" && (
            <Link to={`/studio/${d.id}`}>
              <Button size="sm" variant="primary">
                <Sparkles className="size-3.5" /> Studio
              </Button>
            </Link>
          )}
          <Button size="sm" variant="ghost" onClick={() => setShowLogs(!showLogs)}>
            <ScrollText className="size-3.5" /> Logs
          </Button>
          {d.status === "error" && <Button size="sm" variant="ghost" onClick={() => { setOptions(d.spec.options ?? {}); setConfigure(true); }}>
            <SlidersHorizontal className="size-3.5" /> Configure & retry
          </Button>}
          <Button size="sm" variant="ghost" busy={busy === "restart"} onClick={() => act("restart")}>
            <RotateCcw className="size-3.5" /> Restart
          </Button>
          <Button size="sm" variant="danger" busy={busy === "stop"} onClick={() => act("stop")}>
            <Square className="size-3.5" /> {active ? "Cancel" : "Stop"}
          </Button>
        </div>
      </div>

      <div className="mt-5 flex flex-wrap items-center justify-between gap-3 border-t border-line pt-4">
        <Pipeline d={d} />
        <div className="flex flex-wrap gap-x-4 gap-y-1 font-mono text-[11.5px] text-fg-3">
          <span>
            {d.info?.device === "tpu"
              ? `TPU ×${d.info.tensor_parallel ?? ""}`
              : d.gpus.length
                ? d.gpus.map((g) => `GPU${g}`).join(" + ")
                : "CPU"}
          </span>
          {d.vram_estimate > 0 && <span>≈{bytes(d.vram_estimate)} {d.info?.device === "tpu" ? "HBM" : "VRAM"}</span>}
          {d.ready_at && <span>up {duration(Date.now() / 1000 - d.ready_at)}</span>}
        </div>
      </div>

      {active && (
        <div className="mt-4">
          <ProgressBar value={d.progress} />
          <div className="mt-1.5 text-[13px] text-fg-2">{d.message}</div>
        </div>
      )}
      {d.status === "error" && (
        <div className="mt-4 rounded-lg border border-worst/30 bg-worst-soft px-3.5 py-2.5 text-[13.5px] leading-relaxed text-fg">
          <span className="font-medium text-worst">Failed. </span>
          {d.error}
        </div>
      )}
      <Drawer open={configure} onClose={() => setConfigure(false)} title={`Retry ${d.name}`}>
        <div className="space-y-5 p-5">
          <p className="text-[13px] text-fg-2">Adjust this model’s settings and start a new deployment using its cached download.</p>
          <OptionsForm item={{ modality: d.modality, source: d.spec.model.source === "curated" ? "hf" : d.spec.model.source,
            engine: d.spec.model.engine, custom_code: /trust_remote_code|custom code|custom Python/i.test(d.error ?? "") }}
            options={options} onChange={setOptions} />
          <Button variant="primary" busy={busy === "retry"} onClick={retry}>Deploy with these settings</Button>
        </div>
      </Drawer>

      {ifaceName && (
        <div className="mt-4 flex flex-wrap items-center gap-3 rounded-lg border border-line bg-bg-2 px-3.5 py-2">
          <span className="text-[12.5px] font-medium text-fg-2">{ifaceName}</span>
          {iface.status === "up" && iface.url ? (
            <>
              <a href={iface.url} target="_blank" rel="noreferrer" className="min-w-0 flex-1 truncate font-mono text-[12px] text-fg underline decoration-line-2 underline-offset-4 hover:decoration-fg">
                {iface.url}
              </a>
              <CopyButton text={iface.url} />
              <a href={iface.url} target="_blank" rel="noreferrer" className="text-fg-2 hover:text-fg">
                <ExternalLink className="size-4" />
              </a>
            </>
          ) : iface.status === "error" ? (
            <span className="text-[13px] text-worst">{iface.error}</span>
          ) : (
            <span className="text-[13px] text-fg-2">{iface.message ?? (d.status === "ready" ? "Starting…" : "Starts once the model is ready")}</span>
          )}
          {iface.share_url && (
            <span className="flex w-full items-center gap-2 font-mono text-[12px]">
              <span className="readout">share</span>
              <a href={iface.share_url} target="_blank" rel="noreferrer" className="truncate text-fg underline decoration-line-2 underline-offset-4 hover:decoration-fg">
                {iface.share_url}
              </a>
              <CopyButton text={iface.share_url} />
            </span>
          )}
        </div>
      )}

      {showLogs && (
        <div className="mt-4">
          <Logs id={d.id} />
        </div>
      )}
    </div>
  );
}

export default function Deployments() {
  const { overview } = useLive();
  const loc = useLocation();
  const focus = loc.hash.replace("#", "");
  const [preset, setPreset] = useState<string | null>(null);
  const deps = [...(overview?.deployments ?? [])].sort((a, b) => b.created_at - a.created_at);

  return (
    <div>
      <SectionTitle title="Deployments" description="Everything installed, loading or running on this machine, with live logs.">
        <div className="flex flex-wrap gap-2">
          <Button onClick={() => api.preset().then((p) => setPreset(p.cell))} disabled={!deps.length}>
            <FileCode2 className="size-4" /> Export launch cell
          </Button>
          <Link to="/deploy/new">
            <Button variant="primary">
              <Plus className="size-4" /> New
            </Button>
          </Link>
        </div>
      </SectionTitle>
      {deps.length === 0 ? (
        <Empty icon={<Sparkles className="size-8" />} title="Nothing running yet">
          Launch a model and it shows up here with live install, download and load progress.
          <div className="mt-5">
            <Link to="/deploy/new">
              <Button variant="primary">Launch a model</Button>
            </Link>
          </div>
        </Empty>
      ) : (
        <div className="space-y-3 stagger">
          {deps.map((d) => (
            <DeploymentCard key={d.id} d={d} focus={focus === d.id} />
          ))}
        </div>
      )}
      <Drawer open={preset !== null} onClose={() => setPreset(null)} title="Launch cell" width={720}>
        <div className="space-y-4 p-5">
          <p className="text-[14px] text-fg-2">
            Paste this into a Kaggle notebook cell to recreate this exact setup without clicking. It's ideal for <b>Save Version → Save & Run All</b> runs that keep going for up to 12 hours while your browser is closed.
          </p>
          <div className="relative">
            <pre className="max-h-[60vh] overflow-auto rounded-lg border border-line bg-bg p-4 font-mono text-[11.5px] leading-relaxed scroll-thin">{preset}</pre>
            <div className="absolute right-2 top-2">
              <CopyButton text={preset ?? ""} label="Copy" />
            </div>
          </div>
          <p className="text-[13px] text-fg-3">
            Add a <code className="font-mono">KAGGLEUI_NOTIFY_URL</code> secret (an ntfy.sh topic or Discord webhook) to get the new link pushed to your phone when it starts.
          </p>
        </div>
      </Drawer>
    </div>
  );
}
