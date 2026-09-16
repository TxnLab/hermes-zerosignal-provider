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

from typing import Any

from agent.reasoning_effort import (
    OPENAI_COMPAT_WIRE_EFFORTS,
    clamp_effort,
    requested_effort,
)
from providers import register_provider
from providers.base import ProviderProfile

DEFAULT_ZEROSIGNAL_BASE_URL = "http://127.0.0.1:9376/v1"


class ZeroSignalProfile(ProviderProfile):
    """ZeroSignal: local ``zs-proxy`` in front of the ZeroSignal network."""

    def build_api_kwargs_extras(
        self, *, reasoning_config: dict | None = None, **context: Any
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Map Hermes' reasoning controls to a top-level ``reasoning_effort``.

        The proxy forwards ``reasoning_effort`` to the serving node unchanged, so the
        field goes on the wire verbatim, clamped onto the OpenAI-compatible vocabulary
        (Hermes-internal ``ultra`` becomes ``max``). Nothing requested → nothing sent,
        so each model keeps its own default. An explicit disable sends ``none``, the
        only off switch for models that think by default.

        Deliberately NOT gated on ``context["supports_reasoning"]``: the core allowlist
        in ``agent/reasoning_params.py`` is keyed by upstream hostnames and never
        includes ``127.0.0.1``, so the transport always passes ``False`` here. Gating on
        it would make this method a permanent no-op (the DeepInfra profile documents
        the same trap in hermes-agent #111872). Hermes' nested ``reasoning`` dict form
        is never emitted.
        """
        if isinstance(reasoning_config, dict) and reasoning_config.get("enabled") is False:
            return {}, {"reasoning_effort": "none"}
        effort = requested_effort(reasoning_config)
        clamped = clamp_effort(effort, OPENAI_COMPAT_WIRE_EFFORTS)
        if clamped in OPENAI_COMPAT_WIRE_EFFORTS:
            return {}, {"reasoning_effort": clamped}
        return {}, {}

    def supported_reasoning_efforts(self, model: str | None) -> tuple[str, ...] | None:
        """Pass-through: no per-model clamp is declared.

        The proxy's ``/v1/models`` entries carry ``reasoning.allowed_efforts``, but that
        list is whatever the serving operator typed into their node config — an
        operator declaration, not a measured surface (levels have been observed to be
        byte-identical on some models). Until the network publishes measured
        vocabularies, the requested effort is forwarded as-is and the backend decides.

        If a catalog-derived clamp is added later, follow the pattern in
        hermes-agent ``plugins/model-providers/router/__init__.py``: seed a cache from
        ``fetch_models`` and answer from memory/disk only — this hook runs on the
        per-request hot path and must never block on network I/O.
        """
        return None


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
    # The proxy's /v1/models is an unauthenticated GET returning {"data": [{"id": ...}]},
    # so the base fetch_models() works as-is: it returns None (never raises) when the
    # proxy is not running, and the picker falls back to the curated list below.
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
