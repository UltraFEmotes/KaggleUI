import type {
  CatalogItem,
  CatalogMeta,
  Deployment,
  DeploymentSpec,
  Modality,
  ModelImport,
  Overview,
  SearchResult,
  SecretDetails,
  Session,
  Settings,
  StudioModel,
  UpdateStatus,
} from "./types";

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers = new Headers(init.headers);
  if (init.body && !(init.body instanceof FormData) && !headers.has("content-type")) {
    headers.set("content-type", "application/json");
  }
  const res = await fetch(path, { credentials: "same-origin", cache: "no-store", signal: AbortSignal.timeout(30_000), ...init, headers });
  if (!res.ok) {
    let msg = res.statusText;
    try {
      const data = await res.json();
      msg = data?.detail ?? data?.error?.message ?? data?.error ?? msg;
      if (typeof msg !== "string") msg = JSON.stringify(msg);
    } catch {
      /* not json */
    }
    if (res.status === 405 || (res.status === 404 && path.startsWith("/_kui/api/") && msg === "Not Found")) {
      // An endpoint this page knows about but the running server doesn't: the server is outdated.
      msg = "The KaggleUI server is older than this page. In the notebook, re-run the KaggleUI cell (it restarts the server).";
    }
    throw new ApiError(res.status, msg);
  }
  const type = res.headers.get("content-type") ?? "";
  return (type.includes("application/json") ? res.json() : res.text()) as Promise<T>;
}

const json = (body: unknown) => JSON.stringify(body);

export const api = {
  session: () => request<Session>("/_kui/api/session"),
  login: (token: string) => request("/_kui/api/login", { method: "POST", body: json({ token }) }),
  accessLogin: (password: string) => request("/_kui/api/access-login", { method: "POST", body: json({ password }) }),
  logout: () => request("/_kui/api/logout", { method: "POST" }),
  overview: () => request<Overview>("/_kui/api/overview"),
  updates: (check = false) => request<UpdateStatus>(`/_kui/api/updates?check=${check}`),
  installUpdate: () => request<UpdateStatus>("/_kui/api/updates/install", { method: "POST" }),
  updateLogs: () => request<{ lines: string[] }>("/_kui/api/updates/logs"),
  updateSettings: (patch: Partial<{ [K in keyof Settings]: Partial<Settings[K]> }>) =>
    request<Settings>("/_kui/api/settings", { method: "PUT", body: json(patch) }),
  regenerate: (which: "password" | "api_key") =>
    request<Overview["credentials"]>("/_kui/api/credentials/regenerate", { method: "POST", body: json({ which }) }),
  meta: () => request<CatalogMeta>("/_kui/api/catalog/meta"),
  modelImports: () => request<ModelImport[]>("/_kui/api/catalog/imports"),
  importModel: (body: { url: string; kind?: string; name?: string; revision?: string }) =>
    request<ModelImport>("/_kui/api/catalog/imports", { method: "POST", body: json(body) }),
  cancelImport: (id: string) => request<ModelImport>(`/_kui/api/catalog/imports/${id}`, { method: "DELETE" }),
  customizeImport: (id: string, index: number) => request<CatalogItem>(`/_kui/api/catalog/imports/${id}/models/${index}`),
  search: (p: {
    modality: Modality;
    q?: string;
    source?: string;
    sort?: string;
    show_unsupported?: boolean;
    fit?: "all" | "fits";
    limit?: number;
    live?: boolean;
    min_params?: number;
    max_params?: number;
    min_ctx?: number;
    capabilities?: string[];
    capability_match?: "all" | "any";
    brand?: string;
    engine?: string;
    include_unknown?: boolean;
    hide_gated?: boolean;
    hide_experimental?: boolean;
  }) => {
    const qs = new URLSearchParams({
      modality: p.modality,
      q: p.q ?? "",
      source: p.source ?? "all",
      sort: p.sort ?? "recommended",
      show_unsupported: String(Boolean(p.show_unsupported)),
      fit: p.fit ?? "all",
      limit: String(p.limit ?? 40),
      live: String(p.live ?? true),
    });
    for (const key of ["min_params", "max_params", "min_ctx"] as const) if (p[key] !== undefined) qs.set(key, String(p[key]));
    if (p.capabilities?.length) qs.set("capabilities", p.capabilities.join(","));
    if (p.capability_match) qs.set("capability_match", p.capability_match);
    if (p.brand) qs.set("brand", p.brand);
    if (p.engine) qs.set("engine", p.engine);
    for (const key of ["include_unknown", "hide_gated", "hide_experimental"] as const) if (p[key]) qs.set(key, "true");
    return request<SearchResult>(`/_kui/api/catalog/search?${qs}`);
  },
  detail: (key: string) => request<CatalogItem>(`/_kui/api/catalog/detail?key=${encodeURIComponent(key)}`),
  deploy: (spec: DeploymentSpec) => request<Deployment>("/_kui/api/deployments", { method: "POST", body: json(spec) }),
  stop: (id: string) => request(`/_kui/api/deployments/${id}`, { method: "DELETE" }),
  restart: (id: string) => request<Deployment>(`/_kui/api/deployments/${id}/restart`, { method: "POST" }),
  logs: (id: string) => request<{ lines: string[] }>(`/_kui/api/deployments/${id}/logs`),
  restartTunnel: () => request("/_kui/api/tunnels/main/restart", { method: "POST" }),
  preset: () => request<{ preset: unknown; cell: string }>("/_kui/api/preset"),
  studioModels: () => request<StudioModel[]>("/_kui/api/studio/models"),
  secrets: () => request<SecretDetails>("/_kui/api/secrets"),
  setSecret: (name: string, value: string) =>
    request<SecretDetails>(`/_kui/api/secrets/${name}`, { method: "PUT", body: json({ value }) }),
  clearSecret: (name: string) => request<SecretDetails>(`/_kui/api/secrets/${name}`, { method: "DELETE" }),
  reloadSecrets: () => request<SecretDetails>("/_kui/api/secrets/reload", { method: "POST" }),
};

/** Calls to the public gateway from the Studio (cookie auth + studio header). */
export const gateway = {
  async post(path: string, body: unknown): Promise<Response> {
    const res = await fetch(path, {
      method: "POST",
      credentials: "same-origin",
      headers: { "content-type": "application/json", "x-kui-studio": "1" },
      body: JSON.stringify(body),
    });
    if (!res.ok) throw new ApiError(res.status, await errorText(res));
    return res;
  },
  async form(path: string, form: FormData): Promise<Response> {
    const res = await fetch(path, { method: "POST", credentials: "same-origin", headers: { "x-kui-studio": "1" }, body: form });
    if (!res.ok) throw new ApiError(res.status, await errorText(res));
    return res;
  },
  async get(path: string): Promise<Response> {
    const res = await fetch(path, { credentials: "same-origin", headers: { "x-kui-studio": "1" } });
    if (!res.ok) throw new ApiError(res.status, await errorText(res));
    return res;
  },
};

async function errorText(res: Response): Promise<string> {
  try {
    const data = await res.json();
    return data?.error?.message ?? data?.detail ?? data?.error ?? res.statusText;
  } catch {
    return res.statusText;
  }
}
