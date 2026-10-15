import { Brain, ChevronDown, ImagePlus, Send, SlidersHorizontal, Square, Trash2, X } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";

import { Button, Field, Select, Slider, Textarea } from "../../components/ui";
import { cx, tokens } from "../../lib/format";
import { renderMarkdown } from "../../lib/markdown";
import type { StudioModel } from "../../lib/types";
import { effortLabel } from "../../lib/capabilities";

interface Msg {
  role: "user" | "assistant";
  content: string;
  images?: string[];
  reasoning?: string;
  stats?: { tps: number | null; ttft: number; tokens?: number };
  error?: string;
  pending?: boolean;
}

type Handler = (msg: any) => void;

type Effort = string;

/** One WebSocket per Studio, multiplexed by request id. */
function useStudioSocket() {
  const ws = useRef<WebSocket | null>(null);
  const handlers = useRef(new Map<string, Handler>());
  const [ready, setReady] = useState(false);
  const requests = useRef(new Map<string, AbortController>());

  useEffect(() => {
    let closed = false;
    let timer: number | undefined;
    const deltas = new Map<string, any>();
    const flush = (id: string) => {
      const delta = deltas.get(id);
      if (delta) { handlers.current.get(id)?.(delta); deltas.delete(id); }
    };
    const frame = window.setInterval(() => { for (const id of deltas.keys()) flush(id); }, 40);
    const connect = () => {
      const proto = location.protocol === "https:" ? "wss" : "ws";
      const sock = new WebSocket(`${proto}://${location.host}/_kui/studio/ws`);
      ws.current = sock;
      sock.onopen = () => setReady(true);
      sock.onmessage = (ev) => {
        const msg = JSON.parse(ev.data);
        if (msg.op === "delta") {
          const pending = deltas.get(msg.id) ?? { op: "delta", id: msg.id, content: "", reasoning: "" };
          pending.content += msg.content ?? ""; pending.reasoning += msg.reasoning ?? "";
          deltas.set(msg.id, pending);
        } else { flush(msg.id); handlers.current.get(msg.id)?.(msg); }
      };
      sock.onclose = (event) => {
        setReady(false);
        for (const [id, h] of handlers.current) {
          if (!requests.current.has(id)) { h({ op: "error", message: "Live connection lost. Send your message again to use HTTP mode." }); handlers.current.delete(id); }
        }
        if (!closed && event.code !== 4401) timer = window.setTimeout(connect, 5000);
      };
    };
    connect();
    const ping = window.setInterval(() => { if (ws.current?.readyState === WebSocket.OPEN) ws.current.send(JSON.stringify({ op: "ping" })); }, 20_000);
    return () => {
      closed = true;
      window.clearTimeout(timer);
      window.clearInterval(ping);
      window.clearInterval(frame);
      for (const controller of requests.current.values()) controller.abort();
      requests.current.clear();
      ws.current?.close();
    };
  }, []);

  const send = useCallback((payload: { model: string; messages: unknown[]; params: object; op: string }, onMsg: Handler) => {
    const id = Math.random().toString(36).slice(2);
    handlers.current.set(id, onMsg);
    if (ws.current?.readyState === WebSocket.OPEN) ws.current.send(JSON.stringify({ ...payload, id }));
    else {
      const controller = new AbortController();
      requests.current.set(id, controller);
      const started = performance.now();
      void (async () => {
        try {
          const response = await fetch("/v1/chat/completions", { method: "POST", credentials: "same-origin",
            headers: { "content-type": "application/json", "x-kui-studio": "1" }, signal: controller.signal,
            body: JSON.stringify({ model: payload.model, messages: payload.messages, ...payload.params, stream: false }) });
          const data = await response.json();
          if (!response.ok || data.error) throw new Error(data.error?.message || data.detail || "Model request failed");
          const message = data.choices?.[0]?.message;
          onMsg({ op: "delta", content: message?.content ?? "", reasoning: message?.reasoning_content ?? message?.reasoning });
          onMsg({ op: "done", usage: data.usage, ttft: ((performance.now() - started) / 1000).toFixed(2), tps: null });
        } catch (error) {
          onMsg(controller.signal.aborted ? { op: "done" } : { op: "error", message: error instanceof Error ? error.message : "Request failed" });
        } finally { requests.current.delete(id); }
      })();
    }
    return id;
  }, []);
  const cancel = useCallback((id: string) => {
    if (requests.current.has(id)) requests.current.get(id)?.abort();
    else if (ws.current?.readyState === WebSocket.OPEN) ws.current.send(JSON.stringify({ op: "cancel", id }));
  }, []);
  const done = useCallback((id: string) => handlers.current.delete(id), []);
  return { ready, send, cancel, done };
}

function toDataUrl(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const r = new FileReader();
    r.onload = () => resolve(String(r.result));
    r.onerror = reject;
    r.readAsDataURL(file);
  });
}

function Reasoning({ text, live }: { text: string; live: boolean }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="mb-2 rounded-lg border border-line bg-bg-2">
      <button onClick={() => setOpen(!open)} className="flex w-full items-center gap-2 px-3 py-2 text-[12.5px] text-fg-2 hover:text-fg">
        <Brain className={cx("size-3.5", live && "animate-pulse text-signal")} />
        {live ? "Thinking…" : "Thought process"}
        <ChevronDown className={cx("ml-auto size-3.5 transition-transform", open && "rotate-180")} />
      </button>
      {open && <div className="whitespace-pre-wrap border-t border-line px-3 py-2 font-mono text-[11.5px] leading-relaxed text-fg-3">{text}</div>}
    </div>
  );
}

export default function ChatStudio({ model }: { model: StudioModel }) {
  const { ready, send, cancel, done } = useStudioSocket();
  const [messages, setMessages] = useState<Msg[]>([]);
  const [input, setInput] = useState("");
  const [images, setImages] = useState<string[]>([]);
  const [current, setCurrent] = useState<string | null>(null);
  const [showParams, setShowParams] = useState(false);
  const [system, setSystem] = useState("");
  const [temperature, setTemperature] = useState(0.7);
  const [maxTokens, setMaxTokens] = useState(2048);
  const [effort, setEffort] = useState<Effort>("default");
  // The reply cap can go all the way to the context the model was launched with; prompt and
  // history live in that same window, so the engines reject requests where both don't fit.
  const ctxCap = Math.max(256, Math.round(Number(model.options?.ctx) || 8192));
  const tokenStep = ctxCap <= 32768 ? 64 : ctxCap <= 131072 ? 256 : 1024;
  const scroller = useRef<HTMLDivElement>(null);
  const vision = model.capabilities.includes("vision");
  const thinking = model.reasoning?.supported || model.capabilities.some(cap => ["reasoning", "thinking"].includes(cap));
  const effortOptions = [{ value: "default", label: "Auto" }, ...(model.reasoning?.supported === false ? [] : model.reasoning?.efforts ?? []).map(value => ({ value, label: effortLabel(value) }))];
  const activeEffort = effortOptions.some(option => option.value === effort) ? effort : "default";

  useEffect(() => {
    scroller.current?.scrollTo({ top: scroller.current.scrollHeight });
  }, [messages]);

  // Switching to a model with a smaller window pulls the reply cap down with it.
  useEffect(() => {
    setMaxTokens((value) => Math.min(value, ctxCap));
  }, [ctxCap]);

  function submit() {
    const text = input.trim();
    if ((!text && !images.length) || current) return;
    const user: Msg = { role: "user", content: text, images };
    const history = [...messages.filter((m) => !m.error), user];
    setMessages([...history, { role: "assistant", content: "", pending: true }]);
    setInput("");
    setImages([]);
    const apiMessages = [
      ...(system.trim() ? [{ role: "system", content: system.trim() }] : []),
      ...history.map((m) =>
        m.images?.length
          ? { role: m.role, content: [{ type: "text", text: m.content }, ...m.images.map((url) => ({ type: "image_url", image_url: { url } }))] }
          : { role: m.role, content: m.content },
      ),
    ];
    const id = send(
      {
        op: "chat",
        model: model.name,
        messages: apiMessages,
        params: { temperature, max_tokens: maxTokens, ...(activeEffort !== "default" ? { reasoning_effort: activeEffort } : {}) },
      },
      (ev) => {
        setMessages((prev) => {
          const next = [...prev];
          const last = { ...next[next.length - 1] };
          if (ev.op === "delta") {
            if (ev.content) last.content += ev.content;
            if (ev.reasoning) last.reasoning = (last.reasoning ?? "") + ev.reasoning;
          } else if (ev.op === "done") {
            last.pending = false;
            last.stats = { tps: ev.tps, ttft: ev.ttft, tokens: ev.usage?.completion_tokens };
          } else if (ev.op === "error") {
            last.pending = false;
            last.error = ev.message;
          }
          next[next.length - 1] = last;
          return next;
        });
        if (ev.op === "done" || ev.op === "error") {
          setCurrent(null);
          done(id);
        }
      },
    );
    setCurrent(id);
  }

  return (
    <div className="panel studio-chat flex h-[calc(100dvh-320px)] min-h-[520px] flex-col overflow-hidden">
      <div className="flex items-center justify-between gap-3 border-b border-line px-4 py-2.5">
        <div className="flex items-center gap-2 text-[12.5px] text-fg-2">
          <span className={cx("dot", ready ? "dot-live" : "dot-warn")} />
          {ready ? "Live replies" : "HTTP mode · replies appear when complete"}
        </div>
        <div className="flex flex-wrap items-center justify-end gap-1">
          {thinking && (
            <div className="mr-1 flex items-center gap-2" title="How much the model reasons before answering">
              <span className="readout hidden sm:inline">Reasoning</span>
              <Select aria-label="Reasoning effort" value={activeEffort} onChange={event => setEffort(event.target.value)} className="h-8 max-w-32 text-[12px]" disabled={effortOptions.length < 2}>{effortOptions.map(option => <option key={option.value} value={option.value}>{option.label}</option>)}</Select>
            </div>
          )}
          <Button size="sm" variant="ghost" onClick={() => setShowParams(!showParams)}>
            <SlidersHorizontal className="size-3.5" /> Settings
          </Button>
          <Button size="sm" variant="ghost" onClick={() => setMessages([])} disabled={!messages.length || !!current}>
            <Trash2 className="size-3.5" /> Clear
          </Button>
        </div>
      </div>
      {showParams && (
        <div className="grid gap-4 border-b border-line bg-bg-2 px-4 py-4 md:grid-cols-[2fr_1fr_1fr_1fr]">
          <Field label="System prompt">
            <Textarea rows={2} value={system} onChange={(e) => setSystem(e.target.value)} placeholder="You are a helpful assistant." />
          </Field>
          <Slider label="Temperature" value={temperature} min={0} max={2} step={0.05} onChange={setTemperature} format={(v) => v.toFixed(2)} />
          <div>
            <Slider label="Max tokens" value={maxTokens} min={64} max={ctxCap} step={tokenStep} onChange={setMaxTokens} format={tokens} />
            <p className="mt-1 text-[11px] leading-relaxed text-fg-3">
              Reply length cap, up to the {tokens(ctxCap)} context window this model was launched with.
              Prompt and history share that window, so leave room for them on long conversations.
            </p>
          </div>
          <Field label="Reasoning effort" hint={thinking ? model.reasoning?.efforts.length ? "Available controls detected for this model and runner. Auto keeps the model default." : "This model provides no verified effort controls. Its default reasoning behavior is used." : "No reasoning controls detected for this model."}>
            <Select value={activeEffort} onChange={(e) => setEffort(e.target.value)} disabled={effortOptions.length < 2}>
              {effortOptions.map(option => <option key={option.value} value={option.value}>{option.label}</option>)}
            </Select>
          </Field>
        </div>
      )}

      <div ref={scroller} className="min-h-0 flex-1 space-y-5 overflow-y-auto px-4 py-5 scroll-thin sm:px-6">
        {messages.length === 0 && (
          <div className="flex h-full flex-col items-center justify-center text-center">
            <div className="display max-w-xl text-[32px] text-fg">{model.name}</div>
            <p className="mt-3 max-w-sm text-[14px] text-fg-3">What would you like to work on?</p>
            <div className="mt-6 flex flex-wrap justify-center gap-2">{["Explain something simply", "Help me write some code", "Brainstorm an idea"].map(prompt => <button key={prompt} onClick={() => setInput(prompt)} className="rounded-lg border border-line px-3 py-2 text-[12px] text-fg-2 transition-colors hover:border-line-2 hover:bg-panel-2">{prompt}</button>)}</div>
          </div>
        )}
        {messages.map((m, i) =>
          m.role === "user" ? (
            <div key={i} className="flex justify-end">
              <div className="max-w-[85%] rounded-2xl rounded-br-md border border-line bg-panel-3 px-4 py-2.5 text-[14.5px] whitespace-pre-wrap">
                {m.images?.map((src) => <img key={src.slice(-24)} src={src} alt="" className="mb-2 max-h-48 rounded-lg" />)}
                {m.content}
              </div>
            </div>
          ) : (
            <div key={i} className="max-w-[92%]">
              {m.reasoning && <Reasoning text={m.reasoning} live={!!m.pending && !m.content} />}
              {m.content ? (
                <div className="prose-chat text-[14.5px] leading-relaxed" dangerouslySetInnerHTML={{ __html: renderMarkdown(m.content) }} />
              ) : m.pending && !m.reasoning ? (
                <span className="inline-block h-4 w-1.5 rounded-sm bg-fg animate-blink" />
              ) : null}
              {m.error && <div className="mt-2 rounded-lg border border-worst/40 bg-worst-soft px-3 py-2 text-[13px] text-worst">{m.error}</div>}
              {m.stats && (
                <div className="mt-2 flex gap-3 font-mono text-[10.5px] text-fg-3">
                  {m.stats.tokens ? <span>{m.stats.tokens} tok</span> : null}
                  {m.stats.tps ? <span>{m.stats.tps} tok/s</span> : null}
                  <span>first token {m.stats.ttft}s</span>
                </div>
              )}
            </div>
          ),
        )}
      </div>

      <div className="border-t border-line p-3">
        {images.length > 0 && (
          <div className="mb-2 flex gap-2">
            {images.map((src, i) => (
              <div key={i} className="relative">
                <img src={src} alt="" className="size-14 rounded-md object-cover" />
                <button onClick={() => setImages(images.filter((_, j) => j !== i))} className="absolute -right-1.5 -top-1.5 rounded-full bg-panel-3 p-0.5">
                  <X className="size-3" />
                </button>
              </div>
            ))}
          </div>
        )}
        <div className="flex items-end gap-2">
          {vision && (
            <label className="grid size-11 shrink-0 cursor-pointer place-items-center rounded-lg border border-line text-fg-2 hover:border-line-2 hover:text-fg" title="Attach image">
              <ImagePlus className="size-4" />
              <input
                type="file"
                accept="image/*"
                className="hidden"
                onChange={async (e) => {
                  const f = e.target.files?.[0];
                  if (f) setImages([...images, await toDataUrl(f)]);
                  e.target.value = "";
                }}
              />
            </label>
          )}
          <Textarea
            rows={1}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                submit();
              }
            }}
            placeholder={`Message ${model.name}…`}
            className="max-h-48 min-h-11 resize-none"
          />
          {current ? (
            <Button variant="outline" aria-label="Stop generating" className="h-11 shrink-0" onClick={() => cancel(current)}>
              <Square className="size-4" />
            </Button>
          ) : (
            <Button variant="primary" aria-label="Send message" className="h-11 shrink-0" onClick={submit} disabled={!input.trim() && !images.length}>
              <Send className="size-4" />
            </Button>
          )}
        </div>
      </div>
    </div>
  );
}
