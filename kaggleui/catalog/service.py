"""Merge curated picks, live Hugging Face / Ollama results and attached models; tag each for this machine."""

from __future__ import annotations

import asyncio
import fnmatch
import os
import re
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import yaml

from ..capabilities import context_length, profile
from ..system.probe import HardwareProfile
from ..util import parse_params
from . import attached, brands, hf, ollama, world
from .categories import MESH_ENTRIES, audio_category
from .filters import CatalogFilters
from .fit import Candidate, Fit, assess, gguf_size, pick_quant, rank, weights_from_safetensors
from .resolve import ENGINE_LABELS, MODALITY_PIPELINES, resolve_hf

CURATED_PATH = Path(__file__).with_name("curated.yaml")
VERDICT_ORDER = {"best": 0, "good": 1, "tight": 2, "worst": 3}


class CatalogService:
    def __init__(self, probe: Callable[[], HardwareProfile], imports_root: Path | None = None):
        self.probe = probe
        self.imports_root = imports_root
        self.client = httpx.AsyncClient(timeout=20, headers={"User-Agent": "kaggleui/0.1"})
        self.curated: dict[str, list[dict]] = yaml.safe_load(CURATED_PATH.read_text()) or {}
        self._cache: dict[str, tuple[float, Any]] = {}
        self._pending: dict[str, asyncio.Task] = {}
        self._sem = asyncio.Semaphore(8)

    async def close(self) -> None:
        for task in self._pending.values():
            task.cancel()
        await asyncio.gather(*self._pending.values(), return_exceptions=True)
        await self.client.aclose()

    async def _cached(self, key: str, ttl: float, factory: Callable[[], Any]) -> Any:
        hit = self._cache.get(key)
        if hit and time.time() - hit[0] < ttl:
            return hit[1]
        async def fetch():
            try:
                value = await factory()
                self._cache[key] = (time.time(), value)
                return value
            finally:
                self._pending.pop(key, None)
        if key not in self._pending:
            self._pending[key] = asyncio.create_task(fetch())
        return await asyncio.shield(self._pending[key])

    # =========================================================================================
    # Search
    # =========================================================================================
    async def search(self, modality: str, q: str = "", source: str = "all", sort: str = "recommended",
                     limit: int = 40, show_unsupported: bool | None = None, fit: str = "all", live: bool = True,
                     filters: CatalogFilters | None = None) -> dict:
        """``fit="all"`` returns every model (tagged); ``fit="fits"`` drops the ones that won't run here."""
        hw = self.probe()
        filters = filters or CatalogFilters()
        q = q.strip()
        if show_unsupported is None:
            show_unsupported = modality in ("world", "mesh", "audio")
        more = {"v": False}  # set by sources whose last fetched page was full
        statuses: dict[str, str] = {}
        items: dict[str, dict] = {}

        def add(item: dict) -> None:
            existing = items.get(item["key"])
            if existing is None:
                items[item["key"]] = item
            elif (existing.get("curated") and not item.get("curated")) or modality == "world":
                # Keep curated settings but take live popularity numbers.
                for k in ("downloads", "likes", "updated", "capabilities", "params", "library", "pipeline_tag",
                          "context_length"):
                    if k == "capabilities":
                        existing[k] = sorted(set(existing.get(k, [])) | set(item.get(k, [])))
                        continue
                    if item.get(k) and not existing.get(k):
                        existing[k] = item[k]

        if source in ("all", "curated"):
            for entry in self.curated.get(modality, []):
                if q and q.lower() not in (entry["id"] + " " + entry.get("name", "")).lower():
                    continue
                add(self._from_curated(entry, modality))
            statuses["curated"] = "ok"

        jobs = []
        if modality == "mesh" and source in ("all", "hf"):
            for entry in MESH_ENTRIES:
                if q and q.lower() not in (entry["id"] + " " + entry["name"]).lower():
                    continue
                item = self._from_curated({"source": "hf", "engine": None, **entry}, "mesh")
                item["curated"] = False
                add(item)
        if source in ("all", "hf"):
            if modality == "world":
                # Seed published families first so a temporary Hub outage never
                # reduces World browsing back to the two built-in recipes.
                for entry in [*self.curated.get("world", []), *world.ENTRIES]:
                    if q and q.lower() not in (entry["id"] + " " + entry.get("name", "")).lower():
                        continue
                    add(self._world_item(entry, hw))
                if live:
                    jobs.append(("hf", self._world_items(q, sort, hw, limit, more)))
            elif MODALITY_PIPELINES.get(modality) and live:
                jobs.append(("hf", self._hf_items(modality, q, sort, hw, limit, more, filters)))
        if source in ("all", "ollama") and modality == "text":
            if live:
                terms = brands.search_terms(filters.brand) if filters.brand and not q else []
                jobs.append(("ollama", self._ollama_items(q, sort, limit, more, terms)))
            else:
                for raw in ollama.snapshot(q):
                    add(self._from_ollama(raw))
                statuses["ollama"] = "bundled"
        if source in ("all", "attached") and live:
            jobs.append(("attached", self._attached_items(modality)))
        results = await asyncio.gather(*(j for _, j in jobs), return_exceptions=True)
        for (name, _), res in zip(jobs, results):
            if isinstance(res, BaseException):
                statuses[name] = f"error: {type(res).__name__}: {res}"[:200]
                if name == "ollama":
                    for raw in ollama.snapshot(q):
                        add(self._from_ollama(raw))
                    statuses[name] += " (showing bundled snapshot)"
                continue
            statuses[name] = "ok"
            if name == "hf" and more.get("world_errors"):
                statuses[name] = f"partial: {more['world_errors']} World searches failed; showing available results"
            elif name == "hf" and more.get("hf_errors"):
                statuses[name] = f"partial: {more['hf_errors']} searches failed; showing available results"
            for it in res:
                if name == "attached" and q and q.lower() not in (it["id"] + " " + it["name"]).lower():
                    continue
                add(it)

        all_items = list(items.values())
        if live:
            await self._enrich(all_items)
        fits = [self._apply_fit(it, hw) for it in all_items]
        rank(fits)
        for it, f in zip(all_items, fits):
            it["fit"] = f.to_dict()
            it["brand"] = brands.describe(it["id"], it["name"], it["source"])

        if not show_unsupported:
            all_items = [it for it in all_items if it.get("engine")]
        filter_facets = {"brands": brands.facets(all_items),
                         "engines": [{"id": engine, "label": ENGINE_LABELS.get(engine, engine),
                                      "count": sum(it.get("engine") == engine for it in all_items)}
                                     for engine in sorted({it["engine"] for it in all_items if it.get("engine")})],
                         "capabilities": {cap: sum(cap in it.get("capabilities", []) for it in all_items)
                                          for cap in ("vision", "tools", "reasoning", "embedding")}}
        discovered = len(all_items)
        all_items = [it for it in all_items if filters.matches(it)]
        fitting = sum(1 for it in all_items if it["fit"]["verdict"] != "worst")
        launchable = sum(1 for it in all_items if it.get("engine"))
        total = len(all_items)
        if fit == "fits":
            all_items = [it for it in all_items if it["fit"]["verdict"] != "worst"]
        all_items = self._sort(all_items, sort)
        discover_more = more["v"]
        if modality == "world" and (not show_unsupported or fit == "fits"):
            discover_more = False  # all configured World runtimes are already seeded
        has_more = len(all_items) > limit or discover_more
        return {"modality": modality, "items": all_items[:limit], "sources": statuses, "hardware": _hw_summary(hw),
                "counts": {"total": total, "fits": fitting, "launchable": launchable, "shown": min(len(all_items), limit),
                           "discovered": discovered}, "facets": filter_facets, "filters_active": filters.active,
                "has_more": has_more}

    @staticmethod
    def _sort(items: list[dict], sort: str) -> list[dict]:
        if sort in ("params_asc", "params_desc"):
            direction = 1 if sort == "params_asc" else -1
            # Missing counts belong at the end in both directions. Sort the selected
            # variant, after fit assessment, before taking the requested page.
            return sorted(items, key=lambda i: (
                not bool(i.get("params")), direction * (i.get("params") or 0), i["key"],
            ))
        if sort == "downloads":
            return sorted(items, key=lambda i: -(i.get("downloads") or 0))
        if sort == "newest":
            return sorted(items, key=lambda i: -(i.get("updated") or 0))
        if sort == "trending":
            return sorted(items, key=lambda i: i.get("_order", 1e9))
        return sorted(items, key=lambda i: (VERDICT_ORDER.get(i["fit"]["verdict"], 9), -i["fit"]["score"]))

    # -- item builders ------------------------------------------------------------------------
    def _world_item(self, entry: dict, hw: HardwareProfile, raw: dict | None = None, order: int = 0) -> dict:
        pinned = next((e for e in self.curated.get("world", []) if e["id"] == entry["id"]), None)
        it = self._from_curated({"source": "hf", **entry}, "world")
        it["curated"] = pinned is not None
        it["_order"] = order
        if raw:
            live = self._from_hf(raw, "world", hw, order)
            for key in ("downloads", "likes", "updated", "library", "pipeline_tag"):
                it[key] = live.get(key)
            it["params"] = live.get("params") or it.get("params")
            it["gated"] = live.get("gated", it.get("gated"))
        return it

    async def _world_items(self, q: str, sort: str, hw: HardwareProfile, want: int,
                           more: dict) -> list[dict]:
        searches = world.queries(q, want)

        async def run(params: dict) -> list[dict]:
            async with self._sem:
                return await self._cached(f"world-search:{params}:{sort}", 900,
                                          lambda: hf.search(self.client, sort=sort, **params))

        batches = await asyncio.gather(*(run(params) for params in searches), return_exceptions=True)
        if all(isinstance(batch, BaseException) for batch in batches):
            raise RuntimeError("World discovery is unavailable; showing the bundled catalog")
        more["world_errors"] = sum(isinstance(batch, BaseException) for batch in batches)
        entries = {e["id"].lower(): e for e in [*world.ENTRIES, *self.curated.get("world", [])]}
        out, seen = [], set()
        for params, batch in zip(searches, batches):
            if isinstance(batch, BaseException):
                continue
            if len(batch) >= params["limit"]:
                more["v"] = True
            for raw in batch:
                key = raw["id"].lower()
                if key in seen or not world.matches(raw):
                    continue
                seen.add(key)
                entry = entries.get(key)
                out.append(self._world_item(entry, hw, raw, len(out)) if entry
                           else self._from_hf(raw, "world", hw, len(out)))
        return out

    def _from_curated(self, e: dict, modality: str) -> dict:
        size = int(float(e.get("size_gb", 0)) * 1e9) or None
        params = float(e["params_b"]) * 1e9 if e.get("params_b") else None
        base = {
            "curated": True, "selftest": bool(e.get("selftest")), "description": e.get("blurb", ""),
            "experimental": bool(e.get("experimental")), "gated": bool(e.get("gated")),
            "options": dict(e.get("options") or {}), "recipe": e.get("recipe"),
            "min_sm": e.get("min_sm"),
            "min_vram": int(float(e["min_vram_gb"]) * 1e9) if e.get("min_vram_gb") else None,
            "largest": int(float(e["largest_gb"]) * 1e9) if e.get("largest_gb") else None,
            "capabilities": e.get("tags", []), "downloads": 0, "likes": 0, "updated": None,
        }
        if e["source"] == "ollama":
            name, tag = ollama.split_ref(e["id"])
            return {
                **base, "key": f"ollama:{name}", "source": "ollama", "id": name, "name": e.get("name", name),
                "modality": "text", "engine": "ollama", "params": params, "size": size, "variant": tag,
                "variants": [{"id": tag, "ref": f"{name}:{tag}", "label": tag, "size": size, "params": params}],
                "url": f"https://ollama.com/library/{name}",
                **profile(e["id"], tags=e.get("tags", [])),
            }
        return {
            **base, "key": f"hf:{e['id']}", "source": "hf", "id": e["id"], "name": e.get("name", e["id"]),
            "modality": modality, "engine": e.get("engine"), "params": params, "size": size,
            "variant": e.get("variant"), "variants": [], "url": f"https://huggingface.co/{e['id']}",
            **(profile(e["id"], tags=e.get("tags", [])) if modality == "text" else {}),
        }

    def _from_hf(self, raw: dict, modality: str, hw: HardwareProfile, order: int) -> dict:
        rid = raw["id"]
        res = resolve_hf(modality, rid, raw.get("library_name"), raw.get("tags"), hw.min_sm,
                         accelerator=hw.accelerator)
        st = raw.get("safetensors") or {}
        gg = raw.get("gguf") or {}
        params = gg.get("total") or st.get("total")
        is_gguf = res.engine == "ollama"
        size = None
        if not is_gguf and modality == "text":
            size = weights_from_safetensors(st.get("parameters"))
        dtypes = set((st.get("parameters") or {}).keys())
        windows = [ctx for ctx in (gg.get("context_length"),
                                   context_length(raw.get("config")) if modality == "text" else None)
                   if isinstance(ctx, int) and ctx > 0]
        return {
            "key": f"hf:{rid}", "source": "hf", "id": rid, "name": rid.split("/")[-1], "modality": modality,
            "engine": res.engine, "description": res.note or "", "params": params, "size": size,
            "variant": None, "variants": [], "downloads": raw.get("downloads") or 0,
            "likes": raw.get("likes") or 0, "updated": hf.parse_time(raw.get("lastModified")),
            "gated": hf.is_gated(raw.get("gated")), "library": raw.get("library_name"),
            "pipeline_tag": raw.get("pipeline_tag"), "capabilities": [], "experimental": res.experimental,
            "curated": False, "options": {}, "url": f"https://huggingface.co/{rid}",
            "bf16_only": dtypes == {"BF16"} and modality in ("image", "video"),
            "context_length": max(windows) if windows else None, "_order": order,
            **(profile(rid, tags=raw.get("tags", []), config=raw.get("config"), pipeline=raw.get("pipeline_tag")) if modality == "text" else {}),
        }

    def _from_ollama(self, raw: dict, order: int = 0) -> dict:
        name = raw["name"]
        variants = []
        for s in raw.get("sizes") or []:
            p = parse_params(re.sub(r"^e(?=\d)", "", s))  # "e4b" = effective params (Gemma)
            variants.append({"id": s, "ref": f"{name}:{s}", "label": s.upper(), "params": p,
                             "size": gguf_size(p, "Q4_K_M") if p else None})
        if not variants:
            variants.append({"id": "latest", "ref": f"{name}:latest", "label": "latest", "params": None,
                             "size": None})
        return {
            "key": f"ollama:{name}", "source": "ollama", "id": name, "name": name, "modality": "text",
            "engine": "ollama", "description": raw.get("description", ""), "params": variants[0]["params"],
            "size": variants[0]["size"], "variant": variants[0]["id"], "variants": variants,
            "downloads": raw.get("downloads", 0), "likes": 0, "updated": None, "gated": False,
            "capabilities": raw.get("capabilities", []), "experimental": False, "curated": False,
            "options": {}, "url": f"https://ollama.com/library/{name}", "_order": order,
            **profile(name, tags=raw.get("capabilities", [])),
        }

    async def _hf_items(self, modality: str, q: str, sort: str, hw: HardwareProfile, want: int = 40,
                        more: dict | None = None, filters: CatalogFilters | None = None) -> list[dict]:
        scale = max(1, -(-want // 40))  # fetch deeper as the user asks for more

        def n(base: int) -> int:
            return min(base * scale, 300)

        queries: list[dict] = []
        if modality == "text":
            queries = [{"pipeline_tag": "text-generation", "library": "gguf", "limit": n(30)},
                       {"pipeline_tag": "text-generation", "limit": n(20)},
                       {"pipeline_tag": "image-text-to-text", "library": "gguf", "limit": n(10)},
                       {"pipeline_tag": "image-text-to-text", "limit": n(10)}]
            if filters and "vision" in filters.capabilities and (filters.capability_match == "all" or len(filters.capabilities) == 1):
                queries = [entry for entry in queries if entry["pipeline_tag"] == "image-text-to-text"]
        elif modality in ("image", "video"):
            queries = [{"pipeline_tag": p, "library": "diffusers", "limit": n(30)} for p in MODALITY_PIPELINES[modality]]
        else:
            queries = [{"pipeline_tag": p, "limit": n(40)} for p in MODALITY_PIPELINES[modality]]

        terms = brands.hf_search_terms(filters.brand) if filters and filters.brand and not q else []
        searches = [(params, term) for params in queries for term in (terms or [q])]

        async def run(params: dict, term: str) -> list[dict]:
            key = f"hf-search:{params}:{term}:{sort}"
            async def fetch():
                async with self._sem:
                    return await hf.search(self.client, query=term, sort=sort, **params)
            return await self._cached(key, 900, fetch)

        results = await asyncio.gather(*(run(p, term) for p, term in searches), return_exceptions=True)
        errors = [result for result in results if isinstance(result, BaseException)]
        if len(errors) == len(results):
            raise errors[0]
        if more is not None and errors:
            more["hf_errors"] = len(errors)
        batches = [result if isinstance(result, list) else [] for result in results]
        if more is not None and any(len(b) >= p["limit"] for b, (p, _) in zip(batches, searches)):
            more["v"] = True
        out, seen = [], set()
        order = 0
        for batch in batches:
            for raw in batch:
                if modality in ("tts", "audio") and audio_category(raw) != modality:
                    continue
                if raw["id"] in seen:
                    continue
                seen.add(raw["id"])
                out.append(self._from_hf(raw, modality, hw, order))
                order += 1
        return out

    async def _ollama_items(self, q: str, sort: str = "recommended", want: int = 40,
                            more: dict | None = None, terms: list[str] | None = None) -> list[dict]:
        pages = min(max(1, -(-want // 20)), 8)
        order = "newest" if sort == "newest" else "popular"

        async def fetch() -> dict:
            calls = [ollama.search(self.client, term, "local", order, page)
                     for term in (terms or [q]) for page in range(1, pages + 1)]
            if not q and not terms:
                calls.append(ollama.search(self.client, "", "embedding"))
            res = await asyncio.gather(*calls)
            last_full = any(len(res[index]) >= 20 for index in range(pages - 1, pages * len(terms or [q]), pages))
            merged, seen = [], set()
            for batch in res:
                for r in batch:
                    if r["name"] not in seen:
                        seen.add(r["name"])
                        merged.append(r)
            if not merged:
                raise RuntimeError("no results parsed")
            return {"items": merged, "last_full": last_full}

        res = await self._cached(f"ollama-search:{q}:{terms}:{order}:{pages}", 1800, fetch)
        if more is not None and res["last_full"]:
            more["v"] = True
        return [self._from_ollama(r, i) for i, r in enumerate(res["items"])]

    async def _attached_items(self, modality: str) -> list[dict]:
        root = Path(os.environ.get("KAGGLEUI_INPUT_DIR", "/kaggle/input"))
        found = await self._cached(f"attached:{root}", 30, lambda: asyncio.to_thread(attached.scan, root))
        out = []
        for f in found:
            if f["modality"] != modality:
                continue
            out.append({
                "key": f"attached:{f['path']}", "source": "attached", "id": f["path"], "name": f["name"],
                "modality": modality, "engine": f["engine"], "description": "Downloaded from a custom URL" if f.get("imported") else "Attached to this notebook: no download",
                "params": f.get("params"), "size": f["size"], "variant": None, "variants": [],
                "downloads": 0, "likes": 0, "updated": None, "gated": False, "capabilities": ["imported" if f.get("imported") else "attached"],
                "experimental": bool(f.get("experimental")), "curated": False, "options": f.get("options", {}),
                "recipe": f.get("recipe"), "url": f.get("url"), "attached": True,
                "min_sm": f.get("min_sm"), "min_vram": f.get("min_vram"),
                **{key: f[key] for key in ("capabilities", "reasoning", "custom_code", "context_length") if key in f},
            })
        return out

    # -- enrichment ---------------------------------------------------------------------------
    async def _enrich(self, items: list[dict]) -> None:
        """Fill in exact sizes (Ollama manifests, HF file listings) and context windows (HF config.json).

        Context reads are one tiny cached request per model, so catalog rows can show and filter by
        the true maximum window without opening every model's detail page.
        """
        jobs: list = []  # (callable, arg) so nothing beyond the cap is ever created
        ctx: list = []
        for it in items:
            if it["source"] == "ollama":
                v = next((v for v in it["variants"] if v["id"] == it["variant"]), None)
                if v and not v.get("exact"):
                    jobs.append((self._ollama_variant_size, v))
                if not it.get("context_length"):
                    ctx.append((self._ollama_context, it))
            elif (it["source"] == "hf" and it.get("engine") and it.get("size") is None
                  and it["modality"] != "text"):
                jobs.append((self._hf_sizes, it))
            elif (it["source"] == "hf" and it.get("engine") and it["modality"] == "text"
                  and not it.get("context_length")):
                ctx.append((self._hf_context, it))
        for batch, cap in ((jobs, 16), (ctx, 64)):
            if not batch:
                continue
            try:
                await asyncio.wait_for(asyncio.gather(*(fn(arg) for fn, arg in batch[:cap]), return_exceptions=True),
                                       timeout=6)
            except asyncio.TimeoutError:
                pass  # keep estimates; exact metadata remains cached for the next request
        for it in items:
            if it["source"] == "ollama" and it["variants"]:
                v = next((v for v in it["variants"] if v["id"] == it["variant"]), it["variants"][0])
                it["size"], it["params"] = v.get("size"), v.get("params") or it.get("params")

    async def _hf_context(self, it: dict) -> None:
        revision = (it.get("options") or {}).get("revision")

        async def fetch() -> dict:
            async with self._sem:
                return await hf.config_json(self.client, it["id"], revision=revision)

        config = await self._cached(f"hf-config:{it['id']}:{revision or 'main'}", 6 * 3600, fetch)
        ctx = context_length(config)
        if ctx:
            it["context_length"] = ctx

    async def _ollama_context(self, it: dict) -> None:
        async def fetch() -> list[dict]:
            async with self._sem:
                return await ollama.tags(self.client, it["id"])

        # Same cache key as the detail view, so opening a model warms the list (and vice versa).
        tags = await self._cached(f"ollama-tags:{it['id']}", 3600, fetch)
        windows = [w for w in (ollama.parse_context(t.get("context")) for t in tags) if w]
        if windows:
            it["context_length"] = max(windows)

    async def _ollama_variant_size(self, v: dict) -> None:
        name, tag = ollama.split_ref(v["ref"])

        async def fetch() -> int | None:
            async with self._sem:
                return await ollama.manifest_size(self.client, name, tag)

        size = await self._cached(f"ollama-manifest:{v['ref']}", 86400, fetch)
        if size:
            v["size"] = size
            v["exact"] = True
            if not v.get("params"):
                v["params"] = size / 0.62  # ≈ Q4_K_M bytes per param

    async def _hf_sizes(self, it: dict) -> None:
        async def fetch() -> dict:
            async with self._sem:
                return await hf.model_info(self.client, it["id"], revision=(it.get("options") or {}).get("revision"))

        revision = (it.get("options") or {}).get("revision")
        info = await self._cached(f"hf-info:{it['id']}:{revision or 'main'}", 6 * 3600, fetch)
        sib = info.get("siblings") or []
        if it["engine"] in ("diffusers", "diffusers_video", "diffusers_world"):
            sel = hf.diffusers_files(sib)
            has_index = any(s.get("rfilename") == "model_index.json" for s in sib)
            if not has_index or sel["size"] == 0:
                it["engine"] = None
                it["description"] = "Not a full diffusers pipeline (single-file checkpoint, LoRA or adapter)"
                return
            it["size"], it["largest"], it["variant"] = sel["size"], sel["largest"], sel["variant"]
        elif it["engine"] in ("transformers_asr", "transformers_tts", "transformers_audio", "transformers"):
            it["size"] = hf.transformers_files(sib)["weights"] or None
        elif it["engine"] == "world_recipe" and it.get("recipe"):
            from ..engines.world_engine import list_recipes

            recipe = next(r for r in list_recipes() if r["name"] == it["recipe"])
            it["size"] = sum(int(s.get("size") or 0) for s in sib
                             if any(fnmatch.fnmatch(s["rfilename"], p) for p in recipe.get("allow", ["*"]))) or None
            if recipe.get("extra_models"):
                primary_size = it["size"] or 0
                it["size"] = None  # partial metadata must not be mistaken for a full pipeline estimate
                it["runtime_repos"] = [model["repo"] for model in recipe["extra_models"]]
                extra = 0
                repos = []
                for model in recipe["extra_models"]:
                    repo = model["repo"]
                    async def fetch_extra(repo=repo):
                        async with self._sem:
                            return await hf.model_info(self.client, repo)
                    details = await self._cached(f"hf-info:{repo}:main", 6 * 3600, fetch_extra)
                    files = details.get("siblings") or []
                    if model.get("mode") == "diffusers":
                        amount = hf.diffusers_files(files)["size"]
                    elif model.get("mode") == "transformers":
                        amount = hf.transformers_files(files)["size"]
                    else:
                        amount = sum(int(s.get("size") or 0) for s in files if not model.get("allow") or
                                     any(fnmatch.fnmatch(s["rfilename"], p) for p in model["allow"]))
                    extra += amount
                    repos.append(repo)
                    if hf.is_gated(details.get("gated")):
                        it["runtime_gated"] = True
                it["extra_size"] = extra
                it["runtime_repos"] = repos
                it["size"] = primary_size + extra
        else:
            it["size"] = hf.repo_size(sib) or None

    # -- fit ----------------------------------------------------------------------------------
    def _planned_ctx(self, it: dict) -> int:
        """Serving context for this deployment: the chosen length, capped by the model's window."""
        ctx = int((it.get("options") or {}).get("ctx") or 8192)
        limit = it.get("context_length")
        return min(ctx, int(limit)) if limit else ctx

    def _candidate(self, it: dict, size: int | None = None, params: float | None = None) -> Candidate:
        return Candidate(
            modality=it["modality"], engine=it.get("engine"), weights_bytes=size if size is not None else it.get("size"),
            params=params if params is not None else it.get("params"), largest_component_bytes=it.get("largest"),
            ctx=self._planned_ctx(it), min_vram=it.get("min_vram"),
            gated=bool(it.get("gated") or it.get("runtime_gated")), bf16_only=bool(it.get("bf16_only")),
            experimental=bool(it.get("experimental")), curated=bool(it.get("curated")),
            downloads=int(it.get("downloads") or 0), likes=int(it.get("likes") or 0), updated=it.get("updated"),
            attached=bool(it.get("attached")),
            min_sm=it.get("min_sm"),
        )

    def _apply_fit(self, it: dict, hw: HardwareProfile) -> Fit:
        # GGUF repos on HF: choose the best quant for this machine.
        if it["source"] == "hf" and it.get("engine") == "ollama" and it.get("params"):
            quant, f = pick_quant(float(it["params"]), None, self._candidate(it), hw)
            it["variant"] = quant
            it["size"] = gguf_size(float(it["params"]), quant)
            it["spec"] = self._spec(it, f)
            return f
        # Ollama models with several sizes: pick the biggest one that fits comfortably.
        if it["source"] == "ollama" and len(it["variants"]) > 1:
            best = None
            for v in it["variants"]:
                f = assess(self._candidate(it, v.get("size"), v.get("params")), hw)
                v["fit"] = {"verdict": f.verdict, "reasons": f.reasons}
                key = (VERDICT_ORDER[f.verdict] if f.verdict != "worst" else 9, -(v.get("size") or 0))
                if best is None or key < best[0]:
                    best = (key, v, f)
            assert best
            _, v, f = best
            if it.get("curated") and any(x["id"] == it["variant"] for x in it["variants"]):
                pass  # curated entries keep their pinned tag
            else:
                it["variant"], it["size"], it["params"] = v["id"], v.get("size"), v.get("params")
            f = assess(self._candidate(it), hw)
            it["spec"] = self._spec(it, f)
            return f
        f = assess(self._candidate(it), hw)
        it["spec"] = self._spec(it, f)
        return f

    @staticmethod
    def _spec(it: dict, f: Fit) -> dict | None:
        if not it.get("engine"):
            return None
        options = {**(it.get("options") or {}), **f.suggest}
        if it.get("recipe"):
            options["recipe"] = it["recipe"]
        limit = it.get("context_length")
        ctx = int(options.get("ctx") or 8192)
        if limit:
            ctx = min(ctx, int(limit))
        if ctx != 8192 or "ctx" in options:
            options["ctx"] = ctx  # pin the effective window so engines don't fall back to 8192
        return {
            "modality": it["modality"],
            "model": {"source": it["source"], "id": it["id"], "variant": it.get("variant"), "engine": it["engine"]},
            "options": options,
        }

    # =========================================================================================
    # Detail
    # =========================================================================================
    async def detail(self, key: str) -> dict:
        hw = self.probe()
        source, _, ident = key.partition(":")
        if source == "ollama":
            return await self._ollama_detail(ident, hw)
        if source == "hf":
            return await self._hf_detail(ident, hw)
        raise KeyError(key)

    async def _ollama_detail(self, name: str, hw: HardwareProfile) -> dict:
        tags = await self._cached(f"ollama-tags:{name}", 3600, lambda: ollama.tags(self.client, name))
        curated = next((self._from_curated(e, "text") for e in self.curated.get("text", [])
                        if e["source"] == "ollama" and ollama.split_ref(e["id"])[0] == name), None)
        it = curated or self._from_ollama({"name": name, "sizes": [], "downloads": 0})
        variants = []
        for t in tags:
            if t["tag"] == "latest" or not t.get("size"):
                continue
            p = parse_params(t["tag"]) or (t["size"] / 0.62)
            f = assess(self._candidate(it, t["size"], p), hw)
            variants.append({"id": t["tag"], "ref": f"{name}:{t['tag']}", "label": t["tag"], "size": t["size"],
                             "params": p, "context": t.get("context"),
                             "context_length": ollama.parse_context(t.get("context")), "input": t.get("input"),
                             "fit": {"verdict": f.verdict, "reasons": f.reasons, "warnings": f.warnings}})
        it["variants"] = variants or it["variants"]
        windows = [v["context_length"] for v in it["variants"] if v.get("context_length")]
        if windows:
            it["context_length"] = max(windows)
        return self._finish_detail(it, hw)

    async def _hf_detail(self, repo: str, hw: HardwareProfile) -> dict:
        entry = world.profile(repo)
        revision = (entry.get("options") or {}).get("revision") if entry else None
        info = await self._cached(f"hf-info:{repo}:{revision or 'main'}", 6 * 3600,
                                  lambda: hf.model_info(self.client, repo, revision=revision))
        modality = None
        for m, entries in self.curated.items():
            if any(e["source"] == "hf" and e["id"] == repo for e in entries):
                modality = m
        if not modality and (entry or world.matches(info)):
            modality = "world"
        if not modality and any(model["id"].lower() == repo.lower() for model in MESH_ENTRIES):
            modality = "mesh"
        if not modality:
            from .resolve import PIPELINE_TO_MODALITY
            modality = PIPELINE_TO_MODALITY.get(info.get("pipeline_tag") or "", "text")
            if info.get("pipeline_tag") in ("text-to-audio", "text-to-speech"):
                modality = audio_category(info)
        curated = next((self._from_curated(e, modality) for e in self.curated.get(modality, [])
                        if e["source"] == "hf" and e["id"] == repo), None)
        it = curated or (self._world_item(entry, hw, info) if entry else self._from_hf(info, modality, hw, 0))
        sib = info.get("siblings") or []
        it["card"] = (info.get("cardData") or {}).get("license")
        if it.get("engine") == "ollama":
            variants = []
            for g in hf.gguf_variants(sib):
                p = it.get("params") or g["size"] / 0.62
                f = assess(self._candidate(it, g["size"], p), hw)
                variants.append({"id": g["quant"], "ref": f"hf.co/{repo}:{g['quant']}", "label": g["quant"],
                                 "size": g["size"], "params": p,
                                 "fit": {"verdict": f.verdict, "reasons": f.reasons, "warnings": f.warnings}})
            it["variants"] = variants
            ok = [v for v in variants if v["fit"]["verdict"] in ("good",)]
            pick = max(ok, key=lambda v: v["size"]) if ok else (variants[0] if variants else None)
            if pick:
                it["variant"], it["size"] = pick["id"], pick["size"]
        elif it.get("engine") in ("diffusers", "diffusers_video", "diffusers_world"):
            sel = hf.diffusers_files(sib)
            it["size"], it["largest"], it["variant"] = sel["size"], sel["largest"], sel["variant"]
        elif it.get("engine") in ("vllm", "transformers", "transformers_asr", "transformers_tts", "transformers_audio"):
            it["size"] = hf.transformers_files(sib)["weights"] or it.get("size")
        else:
            it["size"] = hf.repo_size(sib) or it.get("size")
        it["files"] = len(sib)
        if modality == "text" or it.get("engine") in ("transformers_tts", "transformers_audio", "transformers_asr"):
            data = await self._cached(f"hf-capabilities:{repo}:{revision or 'main'}", 6 * 3600,
                                      lambda: hf.capability_files(self.client, repo, revision))
            it["custom_code"] = bool(data["config"].get("auto_map"))
            if modality == "text":
                it.update(profile(repo, tags=info.get("tags", []), config=data["config"], template=data["template"],
                                  pipeline=info.get("pipeline_tag")))
                ctx = context_length(data["config"])
                if ctx and ctx > (it.get("context_length") or 0):
                    it["context_length"] = ctx
        return self._finish_detail(it, hw)

    def _finish_detail(self, it: dict, hw: HardwareProfile) -> dict:
        it["brand"] = brands.describe(it["id"], it["name"], it["source"])
        f = assess(self._candidate(it), hw)
        it["fit"] = f.to_dict()
        it["spec"] = self._spec(it, f)
        it["engine_label"] = ENGINE_LABELS.get(it.get("engine") or "", it.get("engine"))
        it["hardware"] = _hw_summary(hw)
        return it

    def imported(self, model: dict) -> dict:
        """An import can be customized directly without registering it in the catalog."""
        item = {"key": f"import:{model['path']}", "source": "attached", "id": model["path"],
                "name": model["name"], "modality": model["modality"], "engine": model.get("engine"),
                "description": "Custom import · configure this checkpoint before deploying.",
                "params": model.get("params"), "size": model.get("size"), "variant": None, "variants": [],
                "downloads": 0, "likes": 0, "updated": None, "gated": False, "capabilities": ["imported"],
                "experimental": bool(model.get("experimental")), "curated": False, "attached": True,
                "options": model.get("options", {}), "recipe": model.get("recipe"), "url": model.get("url"),
                "min_sm": model.get("min_sm"), "min_vram": model.get("min_vram")}
        item.update({key: model[key] for key in ("capabilities", "reasoning", "custom_code") if key in model})
        return self._finish_detail(item, self.probe())

    def selftest_specs(self) -> list[dict]:
        """Smallest curated model per modality, used by ``kaggleui selftest``."""
        from .fit import TPU_ENGINES

        hw = self.probe()
        out = []
        for modality, entries in self.curated.items():
            for e in entries:
                if e.get("selftest"):
                    it = self._from_curated(e, modality)
                    if hw.accelerator == "tpu" and it.get("engine") not in TPU_ENGINES:
                        continue  # selftest models on a TPU session must use the TPU runner
                    f = assess(self._candidate(it), hw)
                    out.append(self._spec(it, f))
        if hw.accelerator == "tpu" and hw.tpus and not any(s and s["modality"] == "text" for s in out):
            # None of the curated selftest models run on a TPU; use a tiny text model to exercise the TPU runner.
            out.append({"modality": "text", "options": {"ctx": 2048},
                        "model": {"source": "hf", "id": "HuggingFaceTB/SmolLM2-135M-Instruct",
                                  "variant": None, "engine": "vllm"}})
        return [s for s in out if s]


def _hw_summary(hw: HardwareProfile) -> dict:
    return {"accelerator": hw.accelerator_label, "gpus": len(hw.gpus), "tpus": len(hw.tpus),
            "vram_free": sum(g.vram_free for g in hw.gpus), "hbm_free": sum(t.vram_free for t in hw.tpus),
            "ram_available": hw.ram_available,
            "disk_free": hw.models_disk.free if hw.models_disk else None, "fake": hw.fake}
