"""Shared fixtures: an isolated HERMES_HOME with the plugin installed the way a user would.

Mirrors the hermetic-test invariants of hermes-agent's own ``tests/conftest.py``:

* ``HERMES_HOME`` is a throwaway directory, set BEFORE any Hermes module is imported, so
  nothing reads or writes the operator's real ``~/.hermes``.
* No ZeroSignal credential env vars leak in from the developer's shell.
* The plugin is copied into ``$HERMES_HOME/plugins/model-providers/zerosignal/`` at
  session start. ``hermes_cli/auth.py`` and ``hermes_cli/models_catalog_static.py`` extend
  their registries from provider discovery at import time, so the plugin must already be
  on disk when those modules are first imported — exactly as it is for a real install.
"""

from __future__ import annotations

import atexit
import os
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PLUGIN_FILES = ("__init__.py", "plugin.yaml")
CREDENTIAL_ENV_VARS = ("ZEROSIGNAL_API_KEY", "ZEROSIGNAL_BASE_URL")


def install_plugin(target: Path) -> Path:
    """Copy the plugin's two files into ``target`` (a plugin directory)."""
    target.mkdir(parents=True, exist_ok=True)
    for name in PLUGIN_FILES:
        shutil.copy(REPO_ROOT / name, target / name)
    return target


def clear_provider_caches() -> None:
    """Force ``providers`` to re-run discovery on the next lookup.

    Same reset hermes-agent's discovery tests use: empty the registry, drop the cached
    list, and evict imported plugin modules so their ``register_provider`` runs again.
    """
    import providers as _pkg

    _pkg._REGISTRY.clear()
    _pkg._ALIASES.clear()
    _pkg._PROVIDER_LIST_CACHE = None
    _pkg._discovered = False
    for mod in list(sys.modules):
        if mod.startswith(("plugins.model_providers", "_hermes_user_provider")):
            del sys.modules[mod]


# ── Session sandbox, before any test module imports Hermes ──────────────────────────────
_SESSION_HOME = Path(tempfile.mkdtemp(prefix="hermes-zerosignal-test-home-"))
os.environ["HERMES_HOME"] = str(_SESSION_HOME)
os.environ["HERMES_TEST_ISOLATION"] = str(_SESSION_HOME)
atexit.register(shutil.rmtree, _SESSION_HOME, True)
for _var in CREDENTIAL_ENV_VARS:
    os.environ.pop(_var, None)
install_plugin(_SESSION_HOME / "plugins" / "model-providers" / "zerosignal")


@pytest.fixture(autouse=True)
def _no_zerosignal_credentials(monkeypatch):
    for var in CREDENTIAL_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def zerosignal_profile():
    """The profile as Hermes discovered it from the session HERMES_HOME."""
    from providers import get_provider_profile

    profile = get_provider_profile("zerosignal")
    assert profile is not None, "zerosignal profile must be discovered from HERMES_HOME"
    return profile


@pytest.fixture
def fresh_hermes_home(tmp_path, monkeypatch):
    """An empty HERMES_HOME with discovery reset, restored (and reset again) afterwards."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    clear_provider_caches()
    yield tmp_path
    clear_provider_caches()
