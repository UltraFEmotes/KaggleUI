"""Catalog filter validation: parameter ranges, context-window minimums and capability matching."""

from __future__ import annotations

import pytest

from kaggleui.catalog.filters import CatalogFilters


def item(**kw) -> dict:
    return {"brand": {"id": "qwen"}, "engine": "vllm", "params": 8e9, "capabilities": [], **kw}


def test_min_ctx_keeps_only_models_with_a_long_enough_window():
    f = CatalogFilters(min_ctx=131072)
    assert f.active
    assert f.matches(item(context_length=131072))
    assert f.matches(item(context_length=1010000))
    assert not f.matches(item(context_length=32768))
    # Unknown windows are hidden while the minimum is set.
    assert not f.matches(item(context_length=None))
    assert not f.matches(item())


def test_min_ctx_validation():
    assert not CatalogFilters().active
    with pytest.raises(ValueError):
        CatalogFilters(min_ctx=-1)


def test_filters_combine():
    f = CatalogFilters(min_ctx=32768, min_params=4e9, max_params=10e9, capabilities=("tools",))
    assert f.matches(item(context_length=32768, capabilities=["tools"]))
    assert not f.matches(item(context_length=32768, capabilities=[]))
    assert not f.matches(item(context_length=32768, capabilities=["tools"], params=20e9))
