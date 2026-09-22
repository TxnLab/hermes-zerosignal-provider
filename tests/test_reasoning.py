"""Reasoning-effort translation on both wires.

The default wire is Responses, driven in the first half of this file by the real
``ResponsesApiTransport``. There the transport builds ``reasoning: {effort, summary}`` itself
and asks the profile only for the model's declared vocabulary, through
``supported_reasoning_efforts``.

The second half covers chat completions, where ``build_api_kwargs_extras`` does the clamping
onto a top-level ``reasoning_effort``. That wire is not dead: it stays reachable per-model
through ``model.api_mode: chat_completions``, and an unclamped level is a 400 there, so the
coverage is load-bearing rather than historical.
"""

from __future__ import annotations

import pytest

RESPONSES_COMMON = dict(messages=[{"role": "user", "content": "ping"}], tools=None, provider="zerosignal")


def _responses(reasoning_config, *, model, base_url):
    """Request kwargs the Responses transport would send for this model.

    The transport looks the profile up itself from ``provider="zerosignal"``, so the
    registered singleton — the one the ``seeded`` fixture warmed — is what answers.
    """
    from agent.transports.codex import ResponsesApiTransport

    return ResponsesApiTransport().build_kwargs(
        model=model, reasoning_config=reasoning_config, base_url=base_url, **RESPONSES_COMMON
    )


def _extras(profile, reasoning_config, *, model, base_url, supports_reasoning=False):
    """The chat-completions fallback's ``(extra_body, top_level)`` split."""
    return profile.build_api_kwargs_extras(
        reasoning_config=reasoning_config, supports_reasoning=supports_reasoning,
        model=model, base_url=base_url,
    )


# ── The Responses wire (the default) ────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "model, effort, expected",
    [
        # glm-5.3-flash declares low/high/max: Hermes' default ``medium`` must not 400.
        ("glm-5.3-flash", "medium", "low"),
        ("glm-5.3-flash", "minimal", "low"),
        ("glm-5.3-flash", "low", "low"),
        ("glm-5.3-flash", "high", "high"),
        ("glm-5.3-flash", "xhigh", "high"),
        ("glm-5.3-flash", "max", "max"),
        ("glm-5.3-flash", "ultra", "max"),  # Hermes-internal tier never reaches the wire
        ("glm-5.3-flash", "none", "low"),  # thinking cannot be disabled: the floor is the honest match
        # glm-5.2 declares the full ladder: everything passes verbatim.
        ("glm-5.2", "medium", "medium"),
        ("glm-5.2", "none", "none"),
        ("glm-5.2", "xhigh", "xhigh"),
        # an unrecognized vendor tier is dropped from the declared set, the rest still clamps.
        ("some/vendor-tier-model", "medium", "high"),
    ],
)
def test_responses_effort_is_clamped_onto_the_catalog_levels(seeded, model, effort, expected):
    profile, base_url, _stub = seeded
    kwargs = _responses({"enabled": True, "effort": effort}, model=model, base_url=base_url)
    assert kwargs["reasoning"]["effort"] == expected


def test_responses_always_sends_an_effort_even_when_nothing_was_requested(seeded):
    """Unlike the chat wire, the Responses transport has no "send nothing" state: it defaults
    to ``medium`` and clamps. A cold catalog would therefore put an unclamped ``medium`` on
    the wire, which is why the cache is seeded at client construction."""
    _profile, base_url, _stub = seeded
    assert _responses(None, model="glm-5.3-flash", base_url=base_url)["reasoning"]["effort"] == "low"
    assert _responses({"enabled": True}, model="glm-5.2", base_url=base_url)["reasoning"]["effort"] == "medium"


def test_responses_omits_reasoning_for_a_model_that_rejects_it(seeded):
    """``supported: false`` in the catalog is the profile's ``()``, which the transport reads
    as "no reasoning fields at all" — a 400 on any of them otherwise."""
    _profile, base_url, _stub = seeded
    for cfg in ({"enabled": True, "effort": "high"}, {"enabled": False}, None):
        kwargs = _responses(cfg, model="mistralai/Mistral-Nemo-Instruct-2407", base_url=base_url)
        assert "reasoning" not in kwargs


def test_responses_internal_disable_sends_no_reasoning_block(seeded):
    """Hermes' auxiliary calls (titles, compression) pass ``enabled: False``."""
    _profile, base_url, _stub = seeded
    assert "reasoning" not in _responses({"enabled": False}, model="glm-5.2", base_url=base_url)


def test_responses_carries_a_stable_session_correlator(seeded):
    """Pins what makes the proxy's pre-seal strip load-bearing rather than decorative.

    The transport sets ``prompt_cache_key`` on a generic host with no way for a profile to
    opt out, and it is constant for the whole session — the same value whichever operator
    serves a turn. ``inject.DropPromptCacheKey`` removes it before the envelope is sealed.
    That strip is only worth its code if this is still true, so if Hermes ever stops sending
    the field, this test failing is the signal to re-examine the strip, not to delete this.
    """
    _profile, base_url, _stub = seeded
    first = _responses(None, model="glm-5.2", base_url=base_url)
    second = _responses({"enabled": True, "effort": "high"}, model="glm-5.2", base_url=base_url)
    assert first["prompt_cache_key"] == second["prompt_cache_key"]
    assert first["prompt_cache_key"]


# ── Cache seeding ───────────────────────────────────────────────────────────────────────


def test_create_client_seeds_the_cache_before_the_first_request(zerosignal_profile, proxy_stub):
    """Client construction is the one seam that runs before any request and may do I/O, so
    the catalog is warm by the time the transport asks for a vocabulary."""
    base_url, stub = proxy_stub
    assert zerosignal_profile.supported_reasoning_efforts("glm-5.3-flash") is None
    assert stub.seen_headers == []

    assert zerosignal_profile.create_client(base_url=base_url, api_key="zerosignal-local") is None

    assert len(stub.seen_headers) == 1
    assert zerosignal_profile.supported_reasoning_efforts("glm-5.3-flash") == ("low", "high", "max")


def test_create_client_survives_a_down_proxy_and_unknown_kwargs(zerosignal_profile, dead_proxy_url):
    """The core catches and logs a raise here, but a provider that needs the standard client
    must simply return None — including when the proxy is down and when the core has grown a
    kwarg this plugin has never heard of."""
    assert zerosignal_profile.create_client(
        base_url=dead_proxy_url, api_key="k", timeout=1.0, a_future_core_kwarg=object()
    ) is None


def test_create_client_binds_later_lookups_to_the_configured_proxy(zerosignal_profile, proxy_stub):
    """ZEROSIGNAL_BASE_URL is resolved into the runtime, never written back onto the profile,
    so ``self.base_url`` stays the compiled-in default. The per-request hooks must still find
    the cache for a proxy that moved off :9376."""
    base_url, _stub = proxy_stub
    assert base_url != zerosignal_profile.base_url
    zerosignal_profile.create_client(base_url=base_url, api_key="zerosignal-local")
    assert zerosignal_profile.supported_reasoning_efforts("kimi-k3") == ("low", "high", "max")


# ── The declared-efforts hook ───────────────────────────────────────────────────────────


def test_supported_reasoning_efforts_answers_from_cache_only(zerosignal_profile, proxy_stub, monkeypatch):
    base_url, stub = proxy_stub
    monkeypatch.setattr(zerosignal_profile, "base_url", base_url)
    # Cold: unknown, and no network on this hook (it runs on the per-request hot path).
    assert zerosignal_profile.supported_reasoning_efforts("glm-5.3-flash") is None
    assert stub.seen_headers == []
    # Seeded by the picker's fetch: tri-state answers.
    zerosignal_profile.fetch_models(api_key="zerosignal-local")
    assert zerosignal_profile.supported_reasoning_efforts("glm-5.3-flash") == ("low", "high", "max")
    assert zerosignal_profile.supported_reasoning_efforts("mistralai/Mistral-Nemo-Instruct-2407") == ()
    assert zerosignal_profile.supported_reasoning_efforts("moonshotai/kimi-k2.7-code") is None
    assert zerosignal_profile.supported_reasoning_efforts("glm-4.7-flash") is None  # supported, no levels
    assert zerosignal_profile.supported_reasoning_efforts("") is None


def test_profile_sends_no_attribution_headers(zerosignal_profile):
    """No User-Agent or other client fingerprint on inference requests by default."""
    assert zerosignal_profile.default_headers == {}
    assert zerosignal_profile.build_client_kwargs_extras(base_url=zerosignal_profile.base_url) == {}


# ── The chat-completions wire (opt-in per model) ────────────────────────────────────────


@pytest.mark.parametrize(
    "reasoning_config, expected_top_level",
    [
        ({"enabled": True, "effort": "low"}, {"reasoning_effort": "low"}),
        ({"enabled": True, "effort": "medium"}, {"reasoning_effort": "medium"}),
        ({"enabled": True, "effort": "high"}, {"reasoning_effort": "high"}),
        ({"enabled": True, "effort": "xhigh"}, {"reasoning_effort": "xhigh"}),
        ({"enabled": True, "effort": "max"}, {"reasoning_effort": "max"}),
        ({"enabled": True, "effort": "ultra"}, {"reasoning_effort": "max"}),  # Hermes-internal tier clamps
        ({"enabled": True, "effort": "none"}, {"reasoning_effort": "none"}),  # explicit user request
        ({"enabled": False}, {}),  # Hermes' internal disable: unknown model → omit, never guess
        (None, {}),  # nothing requested → model keeps its own default
        ({"enabled": True}, {}),
        ({"enabled": True, "effort": ""}, {}),
        ({"enabled": True, "effort": "future-tier"}, {}),  # unknown level omitted, not forwarded
    ],
)
@pytest.mark.parametrize("supports_reasoning", [False, True])
def test_chat_passes_through_when_the_catalog_is_unknown(
    zerosignal_profile, dead_proxy_url, reasoning_config, expected_top_level, supports_reasoning
):
    extra_body, top_level = _extras(
        zerosignal_profile, reasoning_config, model="glm-5.3",
        base_url=dead_proxy_url, supports_reasoning=supports_reasoning,
    )
    assert extra_body == {}
    assert top_level == expected_top_level


@pytest.mark.parametrize(
    "model, effort, expected",
    [
        ("glm-5.3-flash", "medium", "low"),
        ("glm-5.3-flash", "none", "low"),
        ("glm-5.2", "medium", "medium"),
        ("glm-5.2", "xhigh", "xhigh"),
        # supported but no declared levels → unknown → OpenAI-compatible pass-through.
        ("glm-4.7-flash", "medium", "medium"),
        # no reasoning block at all → unknown → pass-through.
        ("moonshotai/kimi-k2.7-code", "high", "high"),
        ("some/vendor-tier-model", "medium", "high"),
    ],
)
def test_chat_effort_is_clamped_onto_the_catalog_levels(seeded, model, effort, expected):
    profile, base_url, _stub = seeded
    extra_body, top_level = _extras(profile, {"enabled": True, "effort": effort}, model=model, base_url=base_url)
    assert extra_body == {}
    assert top_level == {"reasoning_effort": expected}


def test_chat_model_that_rejects_reasoning_fields_gets_none_of_them(seeded):
    profile, base_url, _stub = seeded
    for cfg in ({"enabled": True, "effort": "high"}, {"enabled": False}, {"enabled": True, "effort": "none"}):
        assert _extras(profile, cfg, model="mistralai/Mistral-Nemo-Instruct-2407", base_url=base_url) == ({}, {})


def test_chat_internal_disable_sends_none_only_where_the_model_accepts_it(seeded):
    """Hermes' auxiliary calls (titles, compression) pass ``enabled: False``."""
    profile, base_url, _stub = seeded
    assert _extras(profile, {"enabled": False}, model="glm-5.2", base_url=base_url) == ({}, {"reasoning_effort": "none"})
    assert _extras(profile, {"enabled": False}, model="glm-5.3-flash", base_url=base_url) == ({}, {})
    assert _extras(profile, {"enabled": False}, model="kimi-k3", base_url=base_url) == ({}, {})


def test_chat_cold_cache_is_seeded_once_from_the_request_path(zerosignal_profile, proxy_stub):
    """A oneshot turn on the fallback wire with no picker and no client construction before
    it still gets the clamp: the first request seeds the cache synchronously from the proxy,
    later requests answer from memory."""
    base_url, stub = proxy_stub
    before = len(stub.seen_headers)
    first = _extras(zerosignal_profile, {"enabled": True, "effort": "medium"}, model="glm-5.3-flash", base_url=base_url)
    second = _extras(zerosignal_profile, {"enabled": True, "effort": "medium"}, model="glm-5.3-flash", base_url=base_url)
    assert first == second == ({}, {"reasoning_effort": "low"})
    assert len(stub.seen_headers) == before + 1


def test_chat_proxy_down_never_raises_and_does_not_retry_per_request(zerosignal_profile, dead_proxy_url, monkeypatch):
    import sys

    module = sys.modules[type(zerosignal_profile).__module__]
    calls = []
    real = module._fetch_catalog_items
    monkeypatch.setattr(module, "_fetch_catalog_items", lambda *a, **k: calls.append(a) or real(*a, **k))
    for _ in range(3):
        assert _extras(zerosignal_profile, {"enabled": True, "effort": "medium"}, model="glm-5.3-flash", base_url=dead_proxy_url) \
            == ({}, {"reasoning_effort": "medium"})
    assert len(calls) == 1  # one bounded attempt, then a cooldown


def test_chat_transport_carries_reasoning_effort_with_supports_reasoning_false(seeded):
    """On the fallback wire the main turn reaches the profile with ``supports_reasoning=False``
    (the core allowlist is keyed by upstream hostnames and never lists 127.0.0.1). The field
    must still land as a top-level request kwarg, and no ``reasoning`` dict may appear in
    ``extra_body``."""
    from agent.transports.chat_completions import ChatCompletionsTransport

    profile, base_url, _stub = seeded
    build = ChatCompletionsTransport().build_kwargs
    common = dict(
        messages=[{"role": "user", "content": "ping"}], tools=None, provider_profile=profile,
        provider_name="zerosignal", supports_reasoning=False, base_url=base_url,
    )
    default = build(model="glm-5.3-flash", reasoning_config={"enabled": True, "effort": "medium"}, **common)
    high = build(model="kimi-k3", reasoning_config={"enabled": True, "effort": "high"}, **common)
    off = build(model="glm-5.2", reasoning_config={"enabled": False}, **common)
    unset = build(model="glm-5.3-flash", reasoning_config=None, **common)

    assert default["reasoning_effort"] == "low"
    assert high["reasoning_effort"] == "high"
    assert off["reasoning_effort"] == "none"
    assert "reasoning_effort" not in unset
    for kwargs in (default, high, off, unset):
        assert "reasoning" not in (kwargs.get("extra_body") or {})
