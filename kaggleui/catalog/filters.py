"""Validate and apply catalog filters before pagination."""

from __future__ import annotations

import math
from dataclasses import dataclass

CAPABILITIES = {"vision", "tools", "reasoning", "embedding"}


@dataclass(frozen=True)
class CatalogFilters:
    min_params: float | None = None
    max_params: float | None = None
    min_ctx: int | None = None
    capabilities: tuple[str, ...] = ()
    capability_match: str = "all"
    brand: str = ""
    engine: str = ""
    include_unknown: bool = False
    hide_gated: bool = False
    hide_experimental: bool = False

    def __post_init__(self):
        for value in (self.min_params, self.max_params):
            if value is not None and (not math.isfinite(value) or value < 0):
                raise ValueError("Parameter limits must be finite, non-negative numbers.")
        if self.min_params is not None and self.max_params is not None and self.min_params > self.max_params:
            raise ValueError("Minimum parameters cannot exceed maximum parameters.")
        if self.min_ctx is not None and (not math.isfinite(self.min_ctx) or self.min_ctx < 0):
            raise ValueError("The context minimum must be a non-negative number of tokens.")
        if set(self.capabilities) - CAPABILITIES:
            raise ValueError("Unknown capability filter. Choose vision, tools, reasoning or embedding.")
        if self.capability_match not in ("all", "any"):
            raise ValueError("capability_match must be 'all' or 'any'.")

    @property
    def active(self) -> bool:
        return any((self.min_params is not None, self.max_params is not None, self.min_ctx is not None,
                    self.capabilities, self.brand, self.engine, self.hide_gated, self.hide_experimental))

    def matches(self, item: dict) -> bool:
        if self.brand and item["brand"]["id"] != self.brand:
            return False
        if self.engine and item.get("engine") != self.engine:
            return False
        if self.hide_gated and (item.get("gated") or item.get("runtime_gated")):
            return False
        if self.hide_experimental and item.get("experimental"):
            return False
        if self.min_ctx is not None:
            value = item.get("context_length")
            if not isinstance(value, (int, float)) or value < self.min_ctx:
                return False
        if self.min_params is not None or self.max_params is not None:
            value = item.get("params")
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                if not self.include_unknown:
                    return False
            elif (self.min_params is not None and value < self.min_params) or (self.max_params is not None and value > self.max_params):
                return False
        if self.capabilities:
            caps = set(item.get("capabilities") or [])
            selected = set(self.capabilities)
            if not (selected <= caps if self.capability_match == "all" else selected & caps):
                return False
        return True
