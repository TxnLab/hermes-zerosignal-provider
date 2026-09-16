"""Reasoning-effort translation: a clamped top-level ``reasoning_effort`` on the wire,
emitted without gating on the host allowlist, and never Hermes' nested ``reasoning`` dict."""

from __future__ import annotations

import pytest


@pytest.mark.parametrize(
    "reasoning_config, expected_top_level",
    [
        ({"enabled": True, "effort": "low"}, {"reasoning_effort": "low"}),
        ({"enabled": True, "effort": "high"}, {"reasoning_effort": "high"}),
        ({"enabled": True, "effort": "xhigh"}, {"reasoning_effort": "xhigh"}),
        ({"enabled": True, "effort": "max"}, {"reasoning_effort": "max"}),
        ({"enabled": True, "effort": "ultra"}, {"reasoning_effort": "max"}),  # Hermes-internal tier clamps
        ({"enabled": True, "effort": "none"}, {"reasoning_effort": "none"}),
        ({"enabled": False}, {"reasoning_effort": "none"}),  # explicit disable → the off switch
        (None, {}),  # nothing requested → model keeps its own default
        ({"enabled": True}, {}),
        ({"enabled": True, "effort": ""}, {}),
        ({"enabled": True, "effort": "future-tier"}, {}),  # unknown level omitted, not forwarded
    ],
)
@pytest.mark.parametrize("supports_reasoning", [False, True])
def test_reasoning_effort_is_emitted_top_level_regardless_of_host_allowlist(
    zerosignal_profile, reasoning_config, expected_top_level, supports_reasoning
):
    extra_body, top_level = zerosignal_profile.build_api_kwargs_extras(
        reasoning_config=reasoning_config,
        supports_reasoning=supports_reasoning,
        model="glm-5.3",
        base_url="http://127.0.0.1:9376/v1",
    )
    assert extra_body == {}
    assert top_level == expected_top_level


def test_transport_carries_reasoning_effort_with_supports_reasoning_false(zerosignal_profile):
    """The main turn reaches the profile with ``supports_reasoning=False`` (the core allowlist
    is keyed by upstream hostnames and never lists 127.0.0.1). The field must still land as a
    top-level request kwarg, and no ``reasoning`` dict may appear in ``extra_body``."""
    from agent.transports.chat_completions import ChatCompletionsTransport

    build = ChatCompletionsTransport().build_kwargs
    common = dict(
        messages=[{"role": "user", "content": "ping"}], tools=None,
        provider_profile=zerosignal_profile, provider_name="zerosignal", supports_reasoning=False,
    )
    on = build(model="kimi-k3", reasoning_config={"enabled": True, "effort": "high"}, **common)
    off = build(model="kimi-k3", reasoning_config={"enabled": False}, **common)
    unset = build(model="glm-5.3-flash", reasoning_config=None, **common)

    assert on["reasoning_effort"] == "high"
    assert off["reasoning_effort"] == "none"
    assert "reasoning_effort" not in unset
    for kwargs in (on, off, unset):
        assert "reasoning" not in (kwargs.get("extra_body") or {})


def test_supported_reasoning_efforts_is_pass_through(zerosignal_profile):
    """Decision: no per-model clamp until the network publishes measured vocabularies."""
    for model in ("glm-5.3", "kimi-k3", "openai/gpt-6-astra", "", None):
        assert zerosignal_profile.supported_reasoning_efforts(model) is None


def test_profile_sends_no_attribution_headers(zerosignal_profile):
    """No User-Agent or other client fingerprint reaches the network by default."""
    assert zerosignal_profile.default_headers == {}
    assert zerosignal_profile.build_client_kwargs_extras(base_url=zerosignal_profile.base_url) == {}
