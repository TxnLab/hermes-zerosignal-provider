"""Discovery and registration invariants, loaded through Hermes' real discovery path.

The plugin is a directory Hermes imports from ``$HERMES_HOME``; nothing here reads the
plugin's source or counts registry entries. Each test asserts an outcome a user sees.
"""

from __future__ import annotations

import pytest

from conftest import install_plugin

DEFAULT_BASE_URL = "http://127.0.0.1:9376/v1"


# ── Discovery paths ─────────────────────────────────────────────────────────────────────


def test_hermes_does_not_bundle_a_zerosignal_provider(fresh_hermes_home):
    """With an empty HERMES_HOME nothing named zerosignal exists — so every registration
    below comes from this plugin, not from a same-named built-in."""
    from providers import get_provider_profile

    assert get_provider_profile("zerosignal") is None


def test_discovered_from_user_model_providers_dir(fresh_hermes_home):
    """``$HERMES_HOME/plugins/model-providers/zerosignal/`` — the manual git-clone install."""
    install_plugin(fresh_hermes_home / "plugins" / "model-providers" / "zerosignal")
    from providers import get_provider_profile

    profile = get_provider_profile("zerosignal")
    assert profile is not None
    assert type(profile).__name__ == "ZeroSignalProfile"
    assert profile.base_url == DEFAULT_BASE_URL


def test_discovered_from_flat_installed_plugins_dir(fresh_hermes_home):
    """``$HERMES_HOME/plugins/zerosignal/`` — where ``hermes plugins install`` clones a
    catalog plugin; imported only because plugin.yaml declares ``kind: model-provider``."""
    install_plugin(fresh_hermes_home / "plugins" / "zerosignal")
    from providers import get_provider_profile

    profile = get_provider_profile("zerosignal")
    assert profile is not None
    assert profile.display_name == "ZeroSignal"


# ── Profile identity ────────────────────────────────────────────────────────────────────


def test_profile_metadata(zerosignal_profile):
    p = zerosignal_profile
    assert p.name == "zerosignal"
    assert p.display_name == "ZeroSignal"
    assert p.auth_type == "api_key"
    assert p.api_mode == "chat_completions"
    assert p.env_vars == ("ZEROSIGNAL_API_KEY", "ZEROSIGNAL_BASE_URL")
    assert p.base_url == DEFAULT_BASE_URL
    assert p.supports_health_check is True
    assert p.supports_model_listing is True
    assert p.signup_url.startswith("https://docs.zerosignal.ai/")
    assert "wallet" in p.description


@pytest.mark.parametrize("alias", ["zs", "zero-signal", "zerosignal"])
def test_aliases_resolve_to_the_canonical_profile(zerosignal_profile, alias):
    from providers import get_provider_profile

    assert get_provider_profile(alias) is zerosignal_profile


# ── Core auto-wiring from the registered profile ────────────────────────────────────────


def test_auth_registry_has_the_provider():
    from hermes_cli.auth import PROVIDER_REGISTRY

    cfg = PROVIDER_REGISTRY["zerosignal"]
    assert cfg.auth_type == "api_key"
    assert cfg.name == "ZeroSignal"
    assert cfg.inference_base_url == DEFAULT_BASE_URL
    assert cfg.api_key_env_vars == ("ZEROSIGNAL_API_KEY",)
    assert cfg.base_url_env_var == "ZEROSIGNAL_BASE_URL"


@pytest.mark.parametrize("alias", ["zs", "zero-signal", "zerosignal", "ZeroSignal"])
def test_resolve_provider_accepts_every_alias(alias):
    """``--provider zs`` and friends resolve without any credential present."""
    from hermes_cli.auth import resolve_provider

    assert resolve_provider(alias) == "zerosignal"


def test_provider_label_is_the_display_name():
    """``hermes_cli.models.provider_label`` (status line, picker headers) uses display_name.

    Note: ``hermes_cli.models.normalize_provider`` only knows Hermes' static alias table, so
    plugin aliases resolve through ``hermes_cli.auth.resolve_provider`` (the ``--provider``
    path) and not through that helper; the canonical slug works everywhere.
    """
    from hermes_cli.models import provider_label

    assert provider_label("zerosignal") == "ZeroSignal"


def test_picker_has_a_zerosignal_row():
    """CANONICAL_PROVIDERS feeds ``hermes model``, ``/model`` and the setup wizard."""
    from hermes_cli.models import CANONICAL_PROVIDERS

    rows = [p for p in CANONICAL_PROVIDERS if p.slug == "zerosignal"]
    assert len(rows) == 1
    assert rows[0].label == "ZeroSignal"
    assert "wallet" in rows[0].tui_desc  # the picker subtitle is the profile description


def test_setup_wizard_knows_the_env_vars():
    from hermes_cli.config import OPTIONAL_ENV_VARS

    key = OPTIONAL_ENV_VARS["ZEROSIGNAL_API_KEY"]
    assert key["category"] == "provider"
    assert key["password"] is True
    url = OPTIONAL_ENV_VARS["ZEROSIGNAL_BASE_URL"]
    assert url["password"] is False


def test_plugin_manifest_declares_the_model_provider_kind():
    """The manifest is what routes a flat ``hermes plugins install`` clone to provider
    discovery instead of the general PluginManager."""
    import yaml
    from conftest import REPO_ROOT

    manifest = yaml.safe_load((REPO_ROOT / "plugin.yaml").read_text(encoding="utf-8"))
    assert manifest["name"] == "zerosignal"
    assert manifest["kind"] == "model-provider"
    assert manifest["version"] and manifest["description"]
