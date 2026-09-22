# hermes-zerosignal-provider

A [Hermes Agent](https://github.com/NousResearch/hermes-agent) model-provider plugin for
[ZeroSignal](https://zerosignal.ai): a local proxy onto a pay-per-request, privacy-preserving
inference network. Your wallet is the credential, so there is no API key to manage.

With the plugin installed, ZeroSignal appears as a provider in `hermes model` and `/model`,
the picker lists the models the proxy currently serves, and Hermes' reasoning-effort setting
is forwarded to the network.

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

### Wire protocol

Requests use the Responses API, which is ZeroSignal's preferred wire — reasoning effort goes
native as `reasoning.effort` rather than a top-level scalar. Chat completions is still fully
supported and tested; select it per-model with:

```yaml
model:
  api_mode: chat_completions
```

Two things had to land before Responses could be the default, and both have.

Hermes' Responses transport attaches a `prompt_cache_key` that is constant for a whole
session and identical whichever operator serves a turn, and a profile cannot turn it off. The
proxy now strips it before sealing, so it never reaches a node. This is defense in depth
rather than a new guarantee — an operator that serves a turn decrypts the conversation and
sees your payer address regardless — but the token was a short, opaque join key that survived
context truncation and compaction, which is exactly where matching on the prompt itself stops
working.

And a node only serves `/v1/responses` when its backend implements it natively or the
operator enabled `llm.openai.translate_responses_to_chat`. Nothing advertises that capability
and nothing routes on it, so a misconfigured operator used to answer 404 with no way for the
proxy to try elsewhere. Nodes now probe the route at startup and refuse to boot without it.

Both wires clamp the reasoning effort; the rest of this section applies to either.

### Reasoning effort

The Hermes reasoning effort (`agent.reasoning_effort` in `config.yaml`, or `--reasoning` on
the command line) is sent to the proxy, and the requested level is clamped onto the model's
`allowed_efforts` from the proxy's `/v1/models` catalog (Hermes' default `medium` becomes
`low` on a node that declares `low/high/max`). Models the catalog does not describe get the
transport's own vocabulary.

Two caveats worth knowing. `allowed_efforts` on that endpoint is a union across every
operator serving the model, not any single node's list, and nothing routes on it — so a
level inside the union can still be refused by the operator that gets the request. And the
clamp is not purely downward: when nothing weaker than your request is supported it selects
the weakest level that is, which for an under-declared model means paying for more reasoning
tokens than you asked for.

On the Responses wire there is also no "send nothing and let the model decide" — the
transport always sends an effort, defaulting to `medium`.

### Model capabilities

Image input, and the model used for fast side tasks like title generation, are read from the
live catalog rather than pinned in source — the cheapest catalog model that can answer in
text wins, and entries the proxy did not price are skipped. Most auxiliary work still uses
the curated `default_aux_model`, because Hermes only consults the catalog hook on its
`prefer_fast` path.

The plugin deliberately sets **no** `max_tokens` cap. With the field absent the proxy sizes
each reservation from the chosen operator's own declared capacity; a fixed ceiling would
replace that with one guessed number and exclude operators whose context window cannot
honour it.

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
