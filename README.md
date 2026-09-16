# hermes-zerosignal-provider

A [Hermes Agent](https://github.com/NousResearch/hermes-agent) model-provider plugin for
[ZeroSignal](https://zerosignal.ai): a local proxy onto a pay-per-request, privacy-preserving
inference network. Your wallet is the credential, so there is no API key to manage.

With the plugin installed, ZeroSignal appears as a provider in `hermes model` and `/model`,
the picker lists the models the proxy currently serves, and Hermes' reasoning-effort setting
is forwarded to the network as `reasoning_effort`.

## Prerequisites

The ZeroSignal proxy must be running and funded on the same machine:

```sh
zs-proxy proxy start
zs-proxy fund
```

See the [ZeroSignal docs](https://docs.zerosignal.ai/using-the-proxy/guides/hermes) for
installing `zs-proxy` and for the other ways to point Hermes at the proxy.

## Install

Once the plugin is listed in the Hermes plugin catalog:

```sh
hermes plugins install zerosignal
```

Until then, clone it into your Hermes home:

```sh
git clone https://github.com/TxnLab/hermes-zerosignal-provider \
  ~/.hermes/plugins/model-providers/zerosignal
```

Then give Hermes a key value so it treats the provider as configured. The proxy ignores the
`Authorization` header, so any non-empty value works. Add to `~/.hermes/.env`:

```sh
ZEROSIGNAL_API_KEY=zerosignal-local
```

Run `hermes model`, pick **ZeroSignal**, and choose a model.

## Configuration

| Variable | Purpose |
| --- | --- |
| `ZEROSIGNAL_API_KEY` | Required by Hermes, ignored by the proxy. Any non-empty value. |
| `ZEROSIGNAL_BASE_URL` | Optional. Set it if the proxy listens somewhere other than `http://127.0.0.1:9376/v1`. |

The Hermes reasoning effort (`agent.reasoning_effort` in `config.yaml`, or `--reasoning` on
the command line) is sent to the proxy as a top-level `reasoning_effort`. Serving nodes
reject levels outside what they declare, so the plugin clamps the requested level onto the
model's `allowed_efforts` from the proxy's `/v1/models` catalog, never upward (Hermes'
default `medium` becomes `low` on a node that declares `low/high/max`). Models the catalog
does not describe get the request as-is.

## Development

The plugin is validated and tested against one pinned `hermes-agent` commit, recorded as
`hermes_ref` under `[tool.hermes-zerosignal-provider]` in `pyproject.toml`. Hermes does not
build wheels outside Nix, so install it editable from a checkout at that commit:

```sh
python3.11 -m venv .venv && . .venv/bin/activate
ref="$(python -c 'import tomllib; print(tomllib.load(open("pyproject.toml","rb"))["tool"]["hermes-zerosignal-provider"]["hermes_ref"])')"
git clone https://github.com/NousResearch/hermes-agent hermes-agent-src && git -C hermes-agent-src checkout "$ref"
pip install -e ./hermes-agent-src
pip install -e ".[dev]"

hermes plugins validate .    # the catalog admission check
pytest
```

`.github/workflows/validate.yml` runs the same steps. To move to a newer Hermes, bump
`hermes_ref`, re-run both, and mention the upstream commit range in the commit message.

## License

MIT. See [LICENSE](LICENSE).
