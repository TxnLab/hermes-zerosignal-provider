"""Catalog-derived capability picks: image input and the cheap auxiliary model.

Both read the same cached ``/v1/models`` payload the effort clamp uses, and both are
cache-only: a cold cache means "no answer", never a network call and never a guess.

Scope note for ``resolve_aux_model``: Hermes consults it only on the ``prefer_fast`` rung,
which is title generation AND only when ``auxiliary.title_generation.prefer_fast_model`` is
set — that config defaults off — and even then only after its own live-catalog family match
misses. On a stock config the hook never runs, and ``default_aux_model`` is what every
auxiliary call gets. The tests below therefore pin the hook's own behaviour, not a ladder
outcome, and the curated constant stays load-bearing.
"""

from __future__ import annotations


# ── Vision ──────────────────────────────────────────────────────────────────────────────


def test_provider_declares_vision_support(zerosignal_profile):
    """A regression guard on the profile field only — it asserts what the profile says, not
    what any node does with an image. The per-model half is ``default_vision_model`` below.
    """
    assert zerosignal_profile.supports_vision is True


def test_vision_default_is_the_cheapest_image_capable_model(seeded):
    profile, _base_url, _stub = seeded
    assert profile.default_vision_model() == "glm-5.3-flash"


def test_vision_default_is_none_while_the_cache_is_cold(zerosignal_profile, proxy_stub):
    _base_url, stub = proxy_stub
    assert zerosignal_profile.default_vision_model() is None
    assert stub.seen_headers == []  # cache-only: this hook never probes


def test_vision_default_is_none_when_no_model_takes_images(seeded):
    profile, base_url, stub = seeded
    stub.models = [{"id": "text-only", "pricing": {"completion": "0.000001"},
                    "architecture": {"input_modalities": ["text"], "output_modalities": ["text"]}}]
    profile.fetch_models(api_key="k", base_url=base_url)
    assert profile.default_vision_model() is None


# ── Auxiliary model ─────────────────────────────────────────────────────────────────────


def test_aux_model_is_the_cheapest_text_model_in_the_catalog(seeded):
    profile, _base_url, _stub = seeded
    assert profile.resolve_aux_model() == "glm-4.7-flash"


def test_aux_model_honours_the_vision_request(seeded):
    profile, _base_url, _stub = seeded
    assert profile.resolve_aux_model(vision=True) == "glm-5.3-flash"


def test_aux_model_prefers_an_unpriced_model_over_nothing_never_over_a_priced_one(seeded):
    """An entry with no ``pricing`` is an unknown, not a bargain: the proxy drops the block
    for image-only and zero-rate routes AND whenever its fee cache is cold. Make the rule
    bite by leaving only unpriced entries — the pick must be empty, not arbitrary."""
    profile, base_url, stub = seeded
    stub.models = [{"id": "unpriced-a", "reasoning": {"supported": True}},
                   {"id": "unpriced-b", "architecture": {"input_modalities": ["text", "image"]}}]
    profile.fetch_models(api_key="k", base_url=base_url)
    assert profile.resolve_aux_model() == ""
    assert profile.default_vision_model() is None


def test_aux_model_never_picks_a_model_that_cannot_answer_in_text(seeded):
    """``some/image-only-model`` is the cheapest entry in the stub catalog by a wide margin
    and must still lose: its only output modality is an image."""
    profile, _base_url, _stub = seeded
    assert profile.resolve_aux_model() != "some/image-only-model"
    assert profile.default_vision_model() != "some/image-only-model"


def test_aux_model_is_empty_while_the_cache_is_cold(zerosignal_profile, proxy_stub):
    """``""`` is the documented "no answer" — the caller then falls through to
    ``default_aux_model``."""
    _base_url, stub = proxy_stub
    assert zerosignal_profile.resolve_aux_model() == ""
    assert stub.seen_headers == []  # cache-only: this hook never probes


def test_curated_aux_constant_remains_the_cold_cache_fallback(zerosignal_profile):
    assert zerosignal_profile.default_aux_model in zerosignal_profile.fallback_models


# ── Which proxy the hooks read ──────────────────────────────────────────────────────────


def test_a_picker_probe_elsewhere_cannot_redirect_the_hooks(zerosignal_profile, proxy_stub, monkeypatch):
    """The base an inference client was built against is where requests actually go, so it
    outranks any base a catalog fetch merely succeeded against.

    Scenario: the real proxy is on ZEROSIGNAL_BASE_URL and a second, stale ``zs-proxy`` is
    still listening on the default :9376 serving a different catalog. The picker probes the
    default; the per-request hooks must keep describing the proxy the requests go to, or
    every later turn is clamped against a vocabulary from the wrong machine.
    """
    import sys

    base_url, _stub = proxy_stub
    module = sys.modules[type(zerosignal_profile).__module__]
    zerosignal_profile.create_client(base_url=base_url, api_key="zerosignal-local")
    assert zerosignal_profile.resolve_aux_model() == "glm-4.7-flash"

    monkeypatch.setattr(
        module, "_fetch_catalog_items",
        lambda *a, **k: [{"id": "stale-proxy-model", "pricing": {"completion": "0.000001"},
                          "reasoning": {"supported": True, "allowed_efforts": ["low"]}}],
    )
    assert zerosignal_profile.fetch_models(api_key="k", base_url=zerosignal_profile.base_url)

    assert zerosignal_profile.resolve_aux_model() == "glm-4.7-flash"
    assert zerosignal_profile.supported_reasoning_efforts("glm-5.3-flash") == ("low", "high", "max")
    assert zerosignal_profile.supported_reasoning_efforts("stale-proxy-model") is None


def test_a_stale_entry_is_still_served(zerosignal_profile, proxy_stub, monkeypatch):
    """Past the TTL the hooks keep answering from the stale map while a refresh is scheduled.
    Returning None instead would unclamp the very requests the cache exists to protect."""
    import sys

    base_url, _stub = proxy_stub
    module = sys.modules[type(zerosignal_profile).__module__]
    zerosignal_profile.create_client(base_url=base_url, api_key="zerosignal-local")
    monkeypatch.setattr(module, "_CATALOG_TTL_SECONDS", -1.0)  # everything is stale now

    assert zerosignal_profile.supported_reasoning_efforts("glm-5.3-flash") == ("low", "high", "max")


# ── Output caps stay unset on purpose ───────────────────────────────────────────────────


def test_profile_supplies_no_max_tokens_cap(zerosignal_profile):
    """Hermes omits ``max_tokens`` unless a profile supplies one, and that is what we want:
    the proxy then sizes each reservation from the chosen operator's own declared capacity
    instead of a single guessed ceiling that would drop narrower operators."""
    assert zerosignal_profile.default_max_tokens is None
    assert zerosignal_profile.get_max_tokens("glm-5.3") is None
