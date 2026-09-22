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
import json
import os
import shutil
import socket
import sys
import tempfile
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from threading import Thread

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


@pytest.fixture(autouse=True)
def _fresh_catalog_cache(zerosignal_profile):
    """Every test starts with an empty per-model reasoning cache and leaves none behind."""
    module = sys.modules[type(zerosignal_profile).__module__]
    module.reset_catalog_cache()
    yield
    module.reset_catalog_cache()


# ── A loopback stand-in for zs-proxy's /v1/models ──────────────────────────────────────

# Entries shaped like the real proxy's catalog (extra fields are ignored). ``reasoning`` is
# what the effort clamp reads; ``architecture`` and ``pricing`` are what the vision and
# auxiliary-model picks read. Values mirror what operators declared on 2026-09-16; the
# per-token ``pricing`` strings are the proxy's own payer-net shape.
PROXY_CATALOG = [
    {
        "id": "glm-5.3-flash", "object": "model", "owned_by": "zerosignal",
        "context_length": 1000000,
        "architecture": {"input_modalities": ["text", "image"], "output_modalities": ["text"]},
        "pricing": {"prompt": "0.00000009075", "completion": "0.0000003025"},
        "reasoning": {"supported": True, "allowed_efforts": ["low", "high", "max"]},
        "tool_use": True,
    },
    {"id": "glm-5.2", "object": "model",
     "architecture": {"input_modalities": ["text"], "output_modalities": ["text"]},
     "pricing": {"prompt": "0.0000004", "completion": "0.0000012"},
     "reasoning": {"supported": True,
                   "allowed_efforts": ["none", "minimal", "low", "medium", "high", "xhigh", "max"]}},
    {"id": "kimi-k3", "object": "model",
     "pricing": {"prompt": "0.0000002", "completion": "0.0000006"},
     "reasoning": {"supported": True, "allowed_efforts": ["low", "high", "max"]}},
    # Cheapest text model in the catalog — the auxiliary pick when no vision is needed.
    {"id": "glm-4.7-flash", "object": "model",
     "pricing": {"prompt": "0.00000002", "completion": "0.00000008"},
     "reasoning": {"supported": True}},
    # No pricing block: the proxy omits one for free and image-only routes, and an entry
    # without a token price must never be chosen as an auxiliary or vision default.
    {"id": "mistralai/Mistral-Nemo-Instruct-2407", "object": "model", "reasoning": {"supported": False}},
    {"id": "moonshotai/kimi-k2.7-code", "object": "model",
     "architecture": {"input_modalities": ["text", "image"], "output_modalities": ["text"]},
     "pricing": {"prompt": "0.0000003", "completion": "0.0000009"}},
    {"id": "some/vendor-tier-model", "object": "model",
     "reasoning": {"supported": True, "allowed_efforts": ["turbo", "high"]}},  # unknown tier dropped
    # Cheapest entry of all, but it cannot answer in text: it must lose every pick.
    {"id": "some/image-only-model", "object": "model",
     "architecture": {"input_modalities": ["text"], "output_modalities": ["image"]},
     "pricing": {"prompt": "0.000000005", "completion": "0.00000001"}},
]


class ProxyModelsStub(BaseHTTPRequestHandler):
    """Serves ``/v1/models`` like zs-proxy and records every request's headers."""

    models: list[dict] = []
    seen_headers: list[dict] = []

    def do_GET(self):
        type(self).seen_headers.append({k.lower(): v for k, v in self.headers.items()})
        if self.path.rstrip("/").endswith("/models"):
            body = json.dumps({"object": "list", "data": self.models}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):  # noqa: A002 — BaseHTTPRequestHandler's signature
        pass


@pytest.fixture
def proxy_stub():
    """A live loopback stub; yields ``(base_url, handler_class)``."""
    ProxyModelsStub.models = [dict(m) for m in PROXY_CATALOG]
    ProxyModelsStub.seen_headers = []
    server = HTTPServer(("127.0.0.1", 0), ProxyModelsStub)
    Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/v1", ProxyModelsStub
    finally:
        server.shutdown()


def unused_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture
def dead_proxy_url() -> str:
    """A loopback base URL nothing listens on (connection refused)."""
    return f"http://127.0.0.1:{unused_port()}/v1"


@pytest.fixture
def seeded(zerosignal_profile, proxy_stub):
    """Profile whose catalog cache was seeded the way the picker seeds it: via fetch_models.

    Returns ``(profile, base_url, handler_class)``. Seeding also records the stub's base URL
    as the active proxy, which is what the per-request hooks key their cache lookups on.
    """
    base_url, stub = proxy_stub
    assert zerosignal_profile.fetch_models(api_key="zerosignal-local", base_url=base_url)
    return zerosignal_profile, base_url, stub
