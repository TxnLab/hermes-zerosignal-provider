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
    # Responses is ZeroSignal's preferred wire. Hermes resolves this through
    # determine_api_mode, which has no host mandate for 127.0.0.1, so the profile's value is
    # honoured; the chat wire stays reachable per-model via model.api_mode.
    assert p.api_mode == "codex_responses"
    assert p.supports_vision is True
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


def test_hermes_actually_resolves_the_profile_onto_the_responses_wire(zerosignal_profile):
    """The profile attribute is not the wire — ``determine_api_mode`` is.

    It never reads ``profile.api_mode``. It resolves a transport through
    ``get_provider()`` and maps that, and two earlier branches can force
    ``chat_completions`` outright: ``is_actual_route`` and a host mandate on the base URL.
    Asserting the attribute alone would pass while Hermes sent chat completions, so this
    drives the real resolver.
    """
    from hermes_cli.providers import determine_api_mode

    assert determine_api_mode("zerosignal", DEFAULT_BASE_URL, "glm-5.3") == "codex_responses"


def test_a_models_dev_entry_would_silently_take_the_wire_back(zerosignal_profile):
    """``get_provider`` checks models.dev BEFORE the plugin-profile branch.

    The profile's ``api_mode`` is honoured only by the last branch, which builds a
    ProviderDef by reverse-mapping it. A models.dev entry short-circuits ahead of that, and
    its default overlay transport is ``openai_chat`` — so publishing ZeroSignal to models.dev
    flips this plugin back to chat completions with no change here and no error. The fix at
    that point is a HERMES_OVERLAYS entry upstream, not an edit to this repo.

    This pins the precedence so the day it happens is a failure here rather than a silent
    downgrade of the wire in production.
    """
    from hermes_cli.providers import get_provider

    pdef = get_provider("zerosignal")
    assert pdef is not None
    assert pdef.source == "plugin-profile", (
        "zerosignal now resolves from %r; its transport no longer comes from this profile's "
        "api_mode — add a HERMES_OVERLAYS entry upstream" % pdef.source
    )
    assert pdef.transport == "codex_responses"


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
