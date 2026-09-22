"""ZeroSignal provider profile for Hermes Agent.

ZeroSignal is a local proxy (``zs-proxy``) onto a pay-per-request, privacy-preserving
inference network. The proxy speaks the OpenAI Responses API and OpenAI-compatible chat
completions on ``http://127.0.0.1:9376/v1``; the wallet behind the proxy is the credential,
so the ``Authorization`` header is ignored. Hermes still needs *some* value in
``ZEROSIGNAL_API_KEY`` to light the provider in its pickers — any non-empty string works.

This module is a standalone model-provider plugin: drop it under
``$HERMES_HOME/plugins/model-providers/zerosignal/`` (or install it with
``hermes plugins install``) and Hermes discovers it through ``providers/__init__.py``.
It never touches Hermes core, never reads ``os.environ`` for credentials (core resolves
them from ``env_vars``), and sends no attribution headers.

WIRE PROTOCOL. Requests ride ``/v1/responses`` (``api_mode="codex_responses"``), which is
ZeroSignal's preferred wire. Reasoning effort goes native as ``reasoning.effort``, clamped
through ``supported_reasoning_efforts`` — the one profile hook that transport reads. The chat
wire remains reachable per-model via ``model.api_mode: chat_completions`` in ``config.yaml``,
which is what ``build_api_kwargs_extras`` below still serves.

Two things had to land before this wire was safe to default to, and both now have:

    ``prompt_cache_key``. Hermes' Responses transport sets it unconditionally on a generic
    host (``agent/transports/codex.py``), and no profile can opt out — ``supports_prompt_cache_key``
    governs the chat path only. It is a digest of (session id, system prompt, tool schemas),
    so it is constant for a whole session and identical whichever operator serves a turn.
    Scope it honestly: target selection is deterministic and affinity *pins* one operator
    across tool rounds, an operator decrypts the full history anyway, and it reads
    ``payer_addr`` off the verified payment — so operators can already join on prefix or
    payer. What this token added was a join that is short, opaque, cheap to index, and
    survives truncation and compaction, precisely where prefix matching stops working, for a
    field nothing downstream needs. ``inject.DropPromptCacheKey`` now strips it pre-seal at
    the proxy, alongside ``DropEmptyTools`` and ``DropRoutingPreferences``. See
    ``proto/SPEC.md`` §3f.

    Responses is assumed of every node but was not mechanically guaranteed. A node emulates
    the route when its provider is ``local`` or ``vertexai``, or when the operator set
    ``llm.openai.translate_responses_to_chat`` (defaults off); otherwise it forwards to an
    upstream that may not implement it. Nothing in the catalog advertises the capability and
    nothing in ``proto/go/selection`` filters on it, so a misconfigured operator was a 404 the
    client could not route around. ``llm.VerifyResponsesRoute`` now probes at node startup and
    refuses to boot when the route is definitively absent.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import urllib.request
from typing import Any, NamedTuple, Optional

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


class ModelInfo(NamedTuple):
    """What one ``/v1/models`` entry tells us, reduced to the fields the profile acts on.

    ``efforts`` is tri-state per the ``supported_reasoning_efforts`` contract: ``None``
    unknown, ``()`` the model takes no reasoning parameters, else the declared levels.
    ``output_price`` is USD per completion token. The proxy drops the whole ``pricing`` block
    when either token rate is zero OR while its fee-rate cache is still cold, so ``None``
    means "no usable price right now" — which covers image-only routes, genuinely free
    models, and a catalog fetched in the first moments after the proxy started. Treating all
    three as unpriced keeps them out of the auxiliary and vision picks, at the cost of
    passing over a real free model; picking blind on an unknown price is the worse trade.
    """

    efforts: Optional[tuple[str, ...]]
    vision: bool
    text_out: bool
    output_price: Optional[float]


# ── Per-model catalog facts, from the proxy's own ``GET /v1/models`` ────────────────────
#
# The catalog carries, per model, ``reasoning.supported`` / ``reasoning.allowed_efforts``,
# ``architecture.input_modalities`` / ``output_modalities`` and ``pricing``. The backend
# behind an operator enforces the declared effort list: sending ``medium`` to a GLM 5.3 node
# that declares ``low/high/max`` is rejected outright. ``create_client`` seeds the cache once
# at client construction, which happens before the first request is built; the per-request
# hooks stay cache-only, as their contract requires.
#
# TWO LIMITS ON THE CLAMP, both worth knowing before trusting it:
#
# * ``allowed_efforts`` here is a CROSS-OPERATOR UNION, not any one node's list. The proxy
#   ORs ``supported`` and unions the levels across every operator serving the model
#   (proxy/internal/hayai/registry.go ``aggregateReasoning``), and nothing in
#   ``proto/go/selection`` filters candidates on the requested effort. So for a model served
#   by two operators with different vocabularies, a level inside the union can still be
#   rejected by the operator that actually gets the request. The clamp narrows the failure
#   rate; it does not eliminate it. Closing it needs per-operator effort data on this
#   endpoint, or an effort filter in selection — both proxy-side.
# * A clamp does NOT only narrow. ``clamp_effort`` falls back to the weakest supported level
#   when nothing weaker than the request exists, so an under-declared vocabulary escalates:
#   ``medium`` against a declared ``high/max`` becomes ``high``. Reasoning tokens are billed
#   output tokens, so a wrong declaration can cost the payer more, not just deny a level.
#
# Cache shape: base_url -> (stamp, {model id -> ModelInfo}). A model absent from the map is
# unknown and every hook falls through to its own default. Entries past the TTL are refreshed
# by the next caller that can do I/O, and a failed refresh keeps serving the stale map.
_CATALOG_TTL_SECONDS = 300.0
_SEED_TIMEOUT_SECONDS = 2.0
_SEED_RETRY_SECONDS = 30.0  # after a failed cold seed (proxy down), stay pass-through this long
_catalog_lock = threading.Lock()
_catalog_by_base: dict[str, tuple[float, dict[str, ModelInfo]]] = {}
_seed_failed_at: dict[str, float] = {}
_warm_in_flight: set[str] = set()
# Which proxy the per-request hooks should read. They receive no base_url of their own, and
# ``self.base_url`` is the compiled-in default even when the user set ZEROSIGNAL_BASE_URL
# (core resolves that override into the runtime, never back onto the profile object), so
# keying those lookups on self.base_url alone would miss the cache for every user who moved
# the proxy off :9376.
#
# Two slots, because the two writers are not equally trustworthy. ``_client_base`` is the
# base the runtime actually built an inference client against — that is where requests go,
# so it wins outright. ``_active_base`` is the last base a catalog fetch succeeded against,
# which is a good guess but is set by the model picker too, and the picker probes whatever
# base core hands it; letting that steal the lookup would silently make the hooks describe a
# proxy no request is being sent to.
_client_base: Optional[str] = None
_active_base: Optional[str] = None


def _norm_base(base_url: Optional[str]) -> str:
    return (base_url or DEFAULT_ZEROSIGNAL_BASE_URL).strip().rstrip("/")


def _remember_active_base(base_url: str, *, authoritative: bool = False) -> None:
    """Record ``base_url`` as a candidate for the hooks' cache lookups.

    ``authoritative`` marks the base an inference client was built against, which outranks
    any base a catalog fetch merely succeeded against.
    """
    global _active_base, _client_base
    with _catalog_lock:
        if authoritative:
            _client_base = base_url
        else:
            _active_base = base_url


def _parse_efforts(reasoning: Any) -> Optional[tuple[str, ...]]:
    """Declared levels for one entry's ``reasoning`` block, tri-state.

    Declared levels are filtered to Hermes' effort ladder so an unrecognized vendor tier
    cannot pass through unclamped. ``supported: true`` with no recognized level reads as
    unknown, not as "no levels".
    """
    if not isinstance(reasoning, dict):
        return None
    if reasoning.get("supported") is False:
        return ()
    declared = reasoning.get("allowed_efforts") or []
    levels = tuple(
        lvl for lvl in (str(e).strip().lower() for e in declared if isinstance(e, str))
        if lvl in EFFORT_LADDER
    )
    return levels or None


def _parse_price(pricing: Any) -> Optional[float]:
    """``pricing.completion`` as USD per token. The proxy emits decimal strings."""
    if not isinstance(pricing, dict):
        return None
    try:
        price = float(pricing["completion"])
    except (KeyError, TypeError, ValueError):
        return None
    return price if price > 0 else None


def parse_catalog(items: Any) -> Optional[dict[str, ModelInfo]]:
    """``/v1/models`` ``data`` array → ``{model id: ModelInfo}``; None if unusable."""
    if not isinstance(items, list):
        return None
    catalog: dict[str, ModelInfo] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        mid = str(item.get("id") or "").strip()
        if not mid:
            continue
        architecture = item.get("architecture")
        modalities = architecture if isinstance(architecture, dict) else {}
        inputs = modalities.get("input_modalities") or []
        outputs = modalities.get("output_modalities") or []
        catalog[mid] = ModelInfo(
            efforts=_parse_efforts(item.get("reasoning")),
            vision=isinstance(inputs, list) and "image" in inputs,
            # An entry that declares no modalities at all is a text model by omission: the
            # proxy only spells them out when it has an operator declaration to report.
            text_out=not isinstance(outputs, list) or not outputs or "text" in outputs,
            output_price=_parse_price(item.get("pricing")),
        )
    return catalog or None


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


def _seed_catalog(base_url: str, items: Any) -> Optional[dict[str, ModelInfo]]:
    parsed = parse_catalog(items)
    if parsed is not None:
        with _catalog_lock:
            _catalog_by_base[_norm_base(base_url)] = (time.monotonic(), parsed)
    return parsed


def _catalog(base_url: str, api_key: Optional[str]) -> Optional[dict[str, ModelInfo]]:
    """The catalog for ``base_url``: cached, else one bounded fetch; stale → one refresh per TTL.

    A failed cold fetch is not retried for ``_SEED_RETRY_SECONDS`` so a down proxy costs one
    short timeout per window, not one per caller.
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


def _warm_async(base_url: str) -> None:
    """Refresh ``base_url`` off the request path, at most one thread per base at a time.

    The per-request hooks may not block, so this is the only way a cache that was cold or
    stale when they ran ever becomes current. Without it a proxy that was down at client
    construction stays unknown for the life of the process, and a long-lived session clamps
    forever against the vocabulary it happened to fetch at startup.

    Skipped under pytest: a background fetch mid-test makes cache state timing-dependent.
    """
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return
    key = _norm_base(base_url)
    with _catalog_lock:
        if key in _warm_in_flight:
            return
        failed_at = _seed_failed_at.get(key)
        if failed_at is not None and time.monotonic() - failed_at < _SEED_RETRY_SECONDS:
            return  # a down proxy costs one attempt per window, not one per request
        _warm_in_flight.add(key)

    def _refresh() -> None:
        try:
            _catalog(key, None)  # the catalog endpoint is unauthenticated
        finally:
            with _catalog_lock:
                _warm_in_flight.discard(key)

    try:
        threading.Thread(target=_refresh, name="zerosignal-catalog-warm", daemon=True).start()
    except Exception as exc:
        with _catalog_lock:
            _warm_in_flight.discard(key)
        logger.debug("zerosignal: catalog warmer failed to start: %s", exc)


def _cached_catalog(base_url: str) -> Optional[dict[str, ModelInfo]]:
    """Cache-only read for ``base_url`` — never blocks, safe on the request hot path.

    A cold or stale answer schedules a background refresh and is still returned as-is, so a
    caller never waits and a stale map keeps serving until a better one lands.
    """
    key = _norm_base(base_url)
    with _catalog_lock:
        entry = _catalog_by_base.get(key)
        stale = entry is not None and time.monotonic() - entry[0] > _CATALOG_TTL_SECONDS
    if entry is None or stale:
        _warm_async(key)
    return entry[1] if entry is not None else None


def _cached_for(profile_base: str) -> Optional[dict[str, ModelInfo]]:
    """Cache-only lookup for the proxy actually in use.

    Tried in order: the base an inference client was built against, the last base a catalog
    fetch succeeded against, the profile default, then a sole cached entry. The last rung
    covers the case where nothing has recorded a base yet but exactly one proxy is cached,
    where there is no ambiguity about which one it is.
    """
    for candidate in (_client_base, _active_base, profile_base):
        if candidate:
            found = _cached_catalog(candidate)
            if found is not None:
                return found
    with _catalog_lock:
        entries = list(_catalog_by_base.values())
    return entries[0][1] if len(entries) == 1 else None


def reset_catalog_cache() -> None:
    """Forget every cached catalog (tests, or after switching proxies)."""
    global _active_base, _client_base
    with _catalog_lock:
        _catalog_by_base.clear()
        _seed_failed_at.clear()
        _warm_in_flight.clear()
        _active_base = _client_base = None


def _cheapest(catalog: dict[str, ModelInfo], *, vision: bool) -> str:
    """Lowest per-output-token model id that can answer in text, or ``""``.

    Ties keep catalog order, which is the proxy's own presentation order.
    """
    best_id, best_price = "", None
    for mid, info in catalog.items():
        if info.output_price is None or not info.text_out or (vision and not info.vision):
            continue
        if best_price is None or info.output_price < best_price:
            best_id, best_price = mid, info.output_price
    return best_id


class ZeroSignalProfile(ProviderProfile):
    """ZeroSignal: local ``zs-proxy`` in front of the ZeroSignal network."""

    def create_client(self, **client_kwargs: Any) -> None:
        """Always ``None`` — ZeroSignal speaks OpenAI-over-HTTP, so the shared client is right.

        The override exists for its side effect: this is the one seam the core offers that
        runs once per client, before any request is built, and that is allowed to do I/O.
        Seeding here gives the per-request hooks an answer while leaving them cache-only, as
        their contract requires. It matters most on the Responses wire, where the transport
        sends ``reasoning.effort`` on every request whether or not the user asked for one, so
        an unseeded cache is an unclamped level and a 400 rather than a soft degradation.

        Client construction is not the only refresh: ``_cached_catalog`` schedules a
        background warm whenever it is asked for a cold or stale map, so a proxy that was
        down at startup heals without a restart.

        Unknown ``client_kwargs`` are tolerated per the base contract; a fetch failure is
        already swallowed downstream, so this never raises.
        """
        base_url = _norm_base(client_kwargs.get("base_url") or self.base_url)
        _remember_active_base(base_url, authoritative=True)
        _catalog(base_url, client_kwargs.get("api_key"))
        return None

    def fetch_models(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 8.0,
    ) -> Optional[list[str]]:
        """Live catalog ids (``{"data": [{"id": ...}]}``); the same payload seeds the
        per-model facts every other hook reads. None, never an exception, when the proxy
        is down.

        The proxy's catalog endpoint is unauthenticated; the placeholder key is forwarded as
        a Bearer token only because the base implementation does the same, and it is ignored.
        """
        effective = _norm_base(base_url or self.base_url)
        items = _fetch_catalog_items(effective, api_key, timeout)
        if items is None:
            return None
        # Only adopt this base as the active one once it actually yielded a usable catalog:
        # the picker probes whatever base core hands it, and a second proxy answering
        # ``{"data": []}`` must not redirect the hooks away from a working cache entry.
        if _seed_catalog(effective, items) is not None:
            _remember_active_base(effective)
        ids = [str(m["id"]) for m in items if isinstance(m, dict) and m.get("id")]
        return list(dict.fromkeys(ids))

    def _info(self, model: Optional[str]) -> Optional[ModelInfo]:
        """Cached facts for *model* on the live proxy, or None while the cache is cold."""
        mid = str(model or "").strip()
        if not mid:
            return None
        catalog = _cached_for(self.base_url)
        return catalog.get(mid) if catalog else None

    def supported_reasoning_efforts(self, model: Optional[str]) -> Optional[tuple[str, ...]]:
        """Catalog-declared levels for *model* (cache only — never network on this hook).

        Tri-state per the base contract: ``None`` unknown (transport keeps its default
        vocabulary), ``()`` the model takes no reasoning fields, else the levels to clamp
        onto. The Responses transport reads this to clamp ``reasoning.effort``; it is the
        only profile hook that transport consults.
        """
        info = self._info(model)
        return info.efforts if info is not None else None

    def default_vision_model(self) -> Optional[str]:
        """Cheapest catalog model that accepts image input, or None while the cache is cold."""
        catalog = _cached_for(self.base_url)
        if not catalog:
            return None
        return _cheapest(catalog, vision=True) or None

    def resolve_aux_model(self, *, vision: bool = False) -> str:
        """Cheapest live model for auxiliary work (compression, titles, summaries), or ``""``.

        ``default_aux_model`` is a hardcoded id and rots the moment the network stops serving
        it — every auxiliary call then spends a round-trip 404ing. The catalog already
        carries per-token pricing, so pick from it and keep the constant as the cold-cache
        fallback. Cache-only and never raises, per the base contract.
        """
        catalog = _cached_for(self.base_url)
        return _cheapest(catalog, vision=vision) if catalog else ""

    def build_api_kwargs_extras(
        self, *, reasoning_config: dict | None = None, **context: Any
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Map Hermes' reasoning controls to a top-level ``reasoning_effort``.

        This is the clamp for the chat-completions wire, which is no longer the default. The
        Responses transport never calls this — it clamps through
        ``supported_reasoning_efforts`` instead — so on the default wire this method is
        dormant, not wrong. **Do not delete it.** Chat stays reachable per-model through
        ``model.api_mode: chat_completions`` in ``config.yaml``, and there an unclamped level
        is a 400, so removing this would break the wire it is the only clamp for.

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
        catalog = _catalog(base_url, context.get("api_key")) if model else None
        info = catalog.get(model) if catalog else None
        allowed = info.efforts if info is not None else None  # None = unknown model

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
    # Responses is ZeroSignal's preferred wire. supported_reasoning_efforts is the hook that
    # transport clamps through, and the tests drive the real ResponsesApiTransport. The chat
    # wire stays reachable per-model through ``model.api_mode: chat_completions``.
    api_mode="codex_responses",
    supports_health_check=True,
    supports_model_listing=True,
    # Image input rides the ordinary message parts; the proxy passes them through and prices
    # them. Which models accept them is a per-model fact — default_vision_model() reads it
    # from the catalog rather than pinning an id here. Note the base class scopes this flag
    # to image content inside TOOL-RESULT messages; whether a given node forwards list-type
    # ``role: tool`` content upstream is a per-backend question nobody has measured, and
    # supports_vision_tool_messages (default True) is the carve-out if one ever needs it.
    supports_vision=True,
    # The primary aux id, not just a fallback: Hermes consults resolve_aux_model() only on
    # the prefer_fast rung (title generation, itself off by default), so on a stock config
    # this constant is what every auxiliary call gets. The hook narrows the rot; it does not
    # remove the need to keep this current.
    default_aux_model="google/gemini-3.7-flash",
    # NOT SET, deliberately: default_max_tokens / get_max_tokens(). Hermes omits max_tokens
    # unless something supplies one, and that is the right call here — with the field absent
    # the proxy sizes each reservation from the chosen operator's own declared capacity
    # (selection.DeriveMaxOutput). A flat profile cap would replace that per-operator
    # derivation with one guessed number and drop operators whose window cannot honour it.
    # (Only the chat transport consults get_max_tokens(); the Responses transport reads
    # agent.max_tokens directly. Leaving both unset is what makes the outcome wire-neutral.)
    # Curated picker list, shown when no key is set or the proxy is unreachable.
    # Mirrors the ZeroSignal entry in models.dev — note that once that entry lands,
    # hermes_cli.providers.get_provider() takes its models.dev branch, whose default overlay
    # says transport="openai_chat"; an api_mode set here stops being honoured at that point
    # and needs a HERMES_OVERLAYS entry upstream instead. All are tool-calling chat models.
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
