"""ZeroSignal provider profile for Hermes Agent.

ZeroSignal is a local proxy (``zs-proxy``) onto a pay-per-request, privacy-preserving
inference network. The proxy speaks OpenAI-compatible chat completions on
``http://127.0.0.1:9376/v1``; the wallet behind the proxy is the credential, so the
``Authorization`` header is ignored. Hermes still needs *some* value in
``ZEROSIGNAL_API_KEY`` to light the provider in its pickers — any non-empty string works.

This module is a standalone model-provider plugin: drop it under
``$HERMES_HOME/plugins/model-providers/zerosignal/`` (or install it with
``hermes plugins install``) and Hermes discovers it through ``providers/__init__.py``.
It never touches Hermes core, never reads ``os.environ`` for credentials (core resolves
them from ``env_vars``), and sends no attribution headers.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import urllib.request
from typing import Any, Optional

from agent.reasoning_effort import (
    EFFORT_LADDER,
    OPENAI_COMPAT_WIRE_EFFORTS,
    clamp_effort,
    requested_effort,
)
from providers import register_provider
from providers.base import ProviderProfile, _profile_user_agent

logger = logging.getLogger(__name__)

DEFAULT_ZEROSIGNAL_BASE_URL = "http://127.0.0.1:9376/v1"

# ── Per-model reasoning vocabulary, from the proxy's own catalog ────────────────────────
#
# ``GET /v1/models`` on the proxy returns, per model, ``reasoning.supported`` and (when the
# serving operator declared them) ``reasoning.allowed_efforts``. The backend behind an
# operator enforces that list: sending ``medium`` (Hermes' default) or ``none`` to a GLM 5.3
# node that declares ``low/high/max`` is rejected outright. So the requested effort is
# clamped onto the model's declared levels before it goes on the wire. The list is still the
# operator's declaration rather than a measured surface — a clamp can only ever narrow, never
# escalate, so the cost of a wrong declaration is a level the user cannot reach, not a 400.
#
# Cache shape: base_url -> {model id -> tuple of levels}; ``()`` means the model accepts no
# reasoning parameters at all; a model absent from the map is unknown (pass-through on the
# OpenAI-compatible vocabulary). ``fetch_models`` seeds it from the same payload the picker
# uses. On a cold cache the request path seeds it synchronously ONCE with a short timeout:
# the proxy is on loopback and answers in milliseconds, and if it is down the chat request
# that follows fails anyway. After that, only stale entries are refreshed, at most once per
# TTL, and a failed refresh keeps serving the stale map.
_CATALOG_TTL_SECONDS = 300.0
_SEED_TIMEOUT_SECONDS = 2.0
_SEED_RETRY_SECONDS = 30.0  # after a failed cold seed (proxy down), stay pass-through this long
_catalog_lock = threading.Lock()
_catalog_by_base: dict[str, tuple[float, dict[str, tuple[str, ...]]]] = {}
_seed_failed_at: dict[str, float] = {}


def _norm_base(base_url: Optional[str]) -> str:
    return (base_url or DEFAULT_ZEROSIGNAL_BASE_URL).strip().rstrip("/")


def parse_catalog_efforts(items: Any) -> Optional[dict[str, tuple[str, ...]]]:
    """``/v1/models`` ``data`` array → ``{model id: allowed levels}``; None if unusable.

    ``reasoning.supported: false`` → ``()`` (omit every reasoning field). Declared levels
    are filtered to Hermes' effort ladder so an unrecognized vendor tier cannot pass through
    unclamped. ``supported: true`` with no recognized level leaves the model out (unknown).
    """
    if not isinstance(items, list):
        return None
    efforts: dict[str, tuple[str, ...]] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        mid = str(item.get("id") or "").strip()
        reasoning = item.get("reasoning")
        if not mid or not isinstance(reasoning, dict):
            continue
        if reasoning.get("supported") is False:
            efforts[mid] = ()
            continue
        declared = reasoning.get("allowed_efforts") or []
        levels = tuple(
            lvl for lvl in (str(e).strip().lower() for e in declared if isinstance(e, str))
            if lvl in EFFORT_LADDER
        )
        if levels:
            efforts[mid] = levels
    return efforts or None


def _fetch_catalog_items(base_url: str, api_key: Optional[str], timeout: float) -> Optional[list]:
    """Raw ``data`` array from ``{base_url}/models``; None on any failure (never raises)."""
    from hermes_cli.urllib_security import open_credentialed_url

    req = urllib.request.Request(base_url.rstrip("/") + "/models")
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")
    req.add_header("Accept", "application/json")
    req.add_header("User-Agent", _profile_user_agent())  # same header the base fetch_models sends
    try:
        with open_credentialed_url(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode())
    except Exception as exc:
        logger.debug("zerosignal: catalog fetch failed: %s", exc)
        return None
    items = data if isinstance(data, list) else data.get("data", []) if isinstance(data, dict) else None
    return items if isinstance(items, list) else None


def _seed_catalog(base_url: str, items: Any) -> Optional[dict[str, tuple[str, ...]]]:
    parsed = parse_catalog_efforts(items)
    if parsed is not None:
        with _catalog_lock:
            _catalog_by_base[_norm_base(base_url)] = (time.monotonic(), parsed)
    return parsed


def _catalog_efforts(base_url: str, api_key: Optional[str]) -> Optional[dict[str, tuple[str, ...]]]:
    """The efforts map for ``base_url``: cached, else one bounded seed; stale → one refresh per TTL.

    A failed cold seed is not retried for ``_SEED_RETRY_SECONDS`` so a down proxy costs one
    short timeout per window, not one per request.
    """
    key = _norm_base(base_url)
    now = time.monotonic()
    with _catalog_lock:
        entry = _catalog_by_base.get(key)
        if entry is None:
            failed_at = _seed_failed_at.get(key)
            if failed_at is not None and now - failed_at < _SEED_RETRY_SECONDS:
                return None
            _seed_failed_at[key] = now  # claim the attempt; cleared on success
        elif now - entry[0] > _CATALOG_TTL_SECONDS:
            # Serve stale while bumping the stamp so only one caller per TTL pays the refresh.
            _catalog_by_base[key] = (now, entry[1])
        else:
            return entry[1]
    stale = entry[1] if entry is not None else None
    items = _fetch_catalog_items(key, api_key, _SEED_TIMEOUT_SECONDS)
    refreshed = _seed_catalog(key, items) if items is not None else None
    if refreshed is None:
        return stale
    with _catalog_lock:
        _seed_failed_at.pop(key, None)
    return refreshed


def reset_catalog_cache() -> None:
    """Forget every cached catalog (tests, or after switching proxies)."""
    with _catalog_lock:
        _catalog_by_base.clear()
        _seed_failed_at.clear()


class ZeroSignalProfile(ProviderProfile):
    """ZeroSignal: local ``zs-proxy`` in front of the ZeroSignal network."""

    def fetch_models(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 8.0,
    ) -> Optional[list[str]]:
        """Live catalog ids (``{"data": [{"id": ...}]}``); the same payload seeds the
        per-model reasoning vocabulary. None, never an exception, when the proxy is down.

        The proxy's catalog endpoint is unauthenticated; the placeholder key is forwarded as
        a Bearer token only because the base implementation does the same, and it is ignored.
        """
        effective = _norm_base(base_url or self.base_url)
        items = _fetch_catalog_items(effective, api_key, timeout)
        if items is None:
            return None
        _seed_catalog(effective, items)
        ids = [str(m["id"]) for m in items if isinstance(m, dict) and m.get("id")]
        return list(dict.fromkeys(ids))

    def supported_reasoning_efforts(self, model: Optional[str]) -> Optional[tuple[str, ...]]:
        """Catalog-declared levels for *model* (cache only — never network on this hook).

        Tri-state per the base contract: ``None`` unknown (pass-through), ``()`` the model
        takes no reasoning fields, else the levels to clamp onto. This answers from whatever
        ``fetch_models`` or the request path has already cached for the default proxy URL.
        """
        mid = str(model or "").strip()
        if not mid:
            return None
        with _catalog_lock:
            entry = _catalog_by_base.get(_norm_base(self.base_url))
        if entry is None:
            return None
        return entry[1].get(mid)

    def build_api_kwargs_extras(
        self, *, reasoning_config: dict | None = None, **context: Any
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Map Hermes' reasoning controls to a top-level ``reasoning_effort``.

        The proxy forwards ``reasoning_effort`` to the serving node unchanged, so the field
        goes on the wire verbatim, clamped onto the model's catalog-declared levels when the
        catalog knows the model and onto the OpenAI-compatible vocabulary otherwise
        (Hermes-internal ``ultra`` becomes ``max``). Nothing requested → nothing sent, so each
        model keeps its own default. Hermes' internal disable (``enabled: False``, used for
        auxiliary calls) sends ``none`` only where the catalog says the model accepts it;
        elsewhere the field is omitted rather than rejected by a model that always thinks.

        Deliberately NOT gated on ``context["supports_reasoning"]``: the core allowlist in
        ``agent/reasoning_params.py`` is keyed by upstream hostnames and never includes
        ``127.0.0.1``, so the transport always passes ``False`` here. Gating on it would make
        this method a permanent no-op (the DeepInfra profile documents the same trap in
        hermes-agent #111872). Hermes' nested ``reasoning`` dict form is never emitted.
        """
        if not isinstance(reasoning_config, dict):
            return {}, {}
        model = str(context.get("model") or "").strip()
        base_url = _norm_base(context.get("base_url") or self.base_url)
        catalog = _catalog_efforts(base_url, context.get("api_key")) if model else None
        allowed = catalog.get(model) if catalog else None  # None = unknown model

        if reasoning_config.get("enabled") is False:
            if allowed and "none" in allowed:
                return {}, {"reasoning_effort": "none"}
            return {}, {}
        effort = requested_effort(reasoning_config)
        if not effort:
            return {}, {}
        if allowed == ():
            return {}, {}  # the catalog says this model takes no reasoning parameters
        vocabulary = allowed or OPENAI_COMPAT_WIRE_EFFORTS
        clamped = clamp_effort(effort, vocabulary)
        if clamped in vocabulary:
            return {}, {"reasoning_effort": clamped}
        return {}, {}


zerosignal = ZeroSignalProfile(
    name="zerosignal",
    aliases=("zs", "zero-signal"),
    display_name="ZeroSignal",
    description=(
        "ZeroSignal — local proxy onto a pay-per-request, private inference network; "
        "no API key, your wallet is the credential"
    ),
    signup_url="https://docs.zerosignal.ai/using-the-proxy/guides/hermes",
    # ZEROSIGNAL_API_KEY: any non-empty value (the proxy ignores it; Hermes needs it set to
    # treat the provider as configured). ZEROSIGNAL_BASE_URL: override for a non-default port.
    env_vars=("ZEROSIGNAL_API_KEY", "ZEROSIGNAL_BASE_URL"),
    base_url=DEFAULT_ZEROSIGNAL_BASE_URL,
    auth_type="api_key",
    api_mode="chat_completions",
    supports_health_check=True,
    supports_model_listing=True,
    default_aux_model="google/gemini-3.7-flash",
    # Curated picker list, shown when no key is set or the proxy is unreachable.
    # Mirrors the ZeroSignal entry in models.dev; all are tool-calling chat models.
    fallback_models=(
        "glm-5.3",
        "glm-5.3-flash",
        "glm-5.2",
        "kimi-k3",
        "grok-4.6",
        "grok-4.5",
        "grok-4.3",
        "google/gemini-3.7-flash",
        "google/gemini-3.8-flash",
        "openai/gpt-5.6-luna",
        "openai/gpt-5.6-terra",
        "openai/gpt-6-astra",
    ),
)

register_provider(zerosignal)
