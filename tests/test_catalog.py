"""Model catalog behaviour: keyless fallback, live fetch against a proxy-shaped stub,
graceful failure when the proxy is down, and the base-URL override."""

from __future__ import annotations

from conftest import PROXY_CATALOG

STUB_IDS = [m["id"] for m in PROXY_CATALOG]


# ── Keyless: curated fallback, no network ───────────────────────────────────────────────


def test_picker_shows_fallback_models_without_a_key(zerosignal_profile, monkeypatch):
    """With no ZEROSIGNAL_API_KEY set the picker must list ``fallback_models`` and must not
    probe the proxy (core skips ``fetch_models`` when no key is present)."""

    def _must_not_fetch(self, **kwargs):
        raise AssertionError("fetch_models must not run without a key")

    monkeypatch.setattr(type(zerosignal_profile), "fetch_models", _must_not_fetch)
    from hermes_cli.models import provider_model_ids

    assert provider_model_ids("zerosignal", force_refresh=True) == list(zerosignal_profile.fallback_models)


def test_fallback_models_are_curated_and_include_the_aux_model(zerosignal_profile):
    assert zerosignal_profile.fallback_models
    assert len(zerosignal_profile.fallback_models) == len(set(zerosignal_profile.fallback_models))
    assert zerosignal_profile.default_aux_model in zerosignal_profile.fallback_models


# ── Live fetch against the proxy's /v1/models shape ─────────────────────────────────────


def test_fetch_models_returns_ids_from_proxy_shaped_catalog(zerosignal_profile, proxy_stub):
    base_url, stub = proxy_stub
    ids = zerosignal_profile.fetch_models(api_key="zerosignal-local", base_url=base_url)
    assert ids == STUB_IDS
    # The placeholder key is forwarded as a Bearer token; the proxy ignores it.
    assert stub.seen_headers[-1].get("authorization") == "Bearer zerosignal-local"


def test_fetch_models_tolerates_a_missing_key(zerosignal_profile, proxy_stub):
    """The proxy's catalog is unauthenticated: no key means no Authorization header, and
    the ids still come back."""
    base_url, stub = proxy_stub
    ids = zerosignal_profile.fetch_models(api_key=None, base_url=base_url)
    assert ids == STUB_IDS
    assert "authorization" not in stub.seen_headers[-1]


def test_fetch_models_dedupes_and_skips_malformed_entries(zerosignal_profile, proxy_stub):
    base_url, stub = proxy_stub
    stub.models = [{"id": "a"}, {"object": "model"}, "junk", {"id": "a"}, {"id": "b"}]
    assert zerosignal_profile.fetch_models(api_key="k", base_url=base_url) == ["a", "b"]


def test_fetch_models_returns_none_when_proxy_is_down(zerosignal_profile, dead_proxy_url):
    """Connection refused → None, never an exception (the picker then uses fallback_models)."""
    assert zerosignal_profile.fetch_models(api_key="zerosignal-local", base_url=dead_proxy_url, timeout=2.0) is None


# ── Through Hermes' credential + catalog path with a key set ────────────────────────────


def test_credentials_honour_base_url_override(monkeypatch):
    monkeypatch.setenv("ZEROSIGNAL_API_KEY", "zerosignal-local")
    monkeypatch.setenv("ZEROSIGNAL_BASE_URL", "http://127.0.0.1:9999/v1")
    from hermes_cli.auth import resolve_api_key_provider_credentials

    creds = resolve_api_key_provider_credentials("zerosignal")
    assert creds["api_key"] == "zerosignal-local"
    assert creds["base_url"] == "http://127.0.0.1:9999/v1"


def test_credentials_default_to_the_local_proxy(monkeypatch):
    monkeypatch.setenv("ZEROSIGNAL_API_KEY", "anything-non-empty")
    from hermes_cli.auth import resolve_api_key_provider_credentials

    assert resolve_api_key_provider_credentials("zerosignal")["base_url"] == "http://127.0.0.1:9376/v1"


def test_picker_uses_live_catalog_when_key_is_set(zerosignal_profile, proxy_stub, monkeypatch):
    """A key plus a reachable proxy → the picker carries live ids, not just the curated list."""
    base_url, _stub = proxy_stub
    monkeypatch.setenv("ZEROSIGNAL_API_KEY", "zerosignal-local")
    monkeypatch.setenv("ZEROSIGNAL_BASE_URL", base_url)
    from hermes_cli.models import provider_model_ids

    ids = provider_model_ids("zerosignal", force_refresh=True)
    assert "moonshotai/kimi-k2.7-code" in ids  # only the live catalog knows this id
    assert ids != list(zerosignal_profile.fallback_models)
    assert set(zerosignal_profile.fallback_models) <= set(ids)  # curated ids still lead
