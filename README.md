# airouter-litellm

[AI Router](https://airouter.io) model selection for your [LiteLLM proxy](https://docs.litellm.ai/docs/simple_proxy).

Your applications send requests to one virtual model, `airouter-auto`. Before LiteLLM
routes such a request, this callback asks the AI Router which of your models fits the
prompt best for quality, cost and latency, and sends the request to that deployment.

- **Your gateway stays in charge.** Provider keys, deployments, budgets, rate limits and
  spend tracking stay in LiteLLM. The completion runs on your own provider accounts.
- **Any provider, any client format.** Works for `/v1/chat/completions` (including
  streaming), `/v1/messages` (Anthropic format) and `/v1/responses`, whatever runs
  behind your deployments.
- **Never blocks a request.** If the AI Router is unreachable, slower than the timeout,
  has no evidence for the prompt or returns an error, the request runs on your default
  model.
- **Respects your access rules.** Only models the calling key and its team may use are
  considered.
- **Usage shows up in your AI Router dashboard.** Token usage of every routed call is
  reported in the background, outside the request path.

## Requirements

- LiteLLM proxy 1.104 or later (tested with 1.104.0), Python 3.10 or later.
- An AI Router organization with a plan that includes the Decision API, and an API key
  of that organization.

## Installation

Choose one of the two ways.

### Option A: single file next to your config

LiteLLM loads callback modules from the directory of its `config.yaml`. Download
[`airouter_litellm.py`](airouter_litellm.py) and put it there. Its only dependency,
`httpx`, ships with LiteLLM.

```bash
curl -fsSLO https://raw.githubusercontent.com/heureka-labs/airouter-litellm/main/airouter_litellm.py
```

With Docker, mount both files into the container:

```bash
docker run -d -p 4000:4000 \
  -v $(pwd)/config.yaml:/app/config.yaml \
  -v $(pwd)/airouter_litellm.py:/app/airouter_litellm.py \
  -e AIROUTER_API_KEY=... \
  ghcr.io/berriai/litellm:<version> --config /app/config.yaml
```

### Option B: Python package

Install into the environment the proxy runs in, or into your own image based on the
LiteLLM image:

```bash
pip install "git+https://github.com/heureka-labs/airouter-litellm@v0.1.1"
```

```dockerfile
FROM ghcr.io/berriai/litellm:<version>
RUN pip install "git+https://github.com/heureka-labs/airouter-litellm@v0.1.1"
```

## Configuration

**1. Add the virtual model.** It is a regular model entry. Point it at the model you
rely on today, the one you trust to solve your requests, typically a frontier model.
It answers whenever the AI Router does not decide, so a request without a decision is
handled exactly as before. The AI Router moves a request to another model only when it
has evidence that this model fits:

```yaml
model_list:
  - model_name: airouter-auto
    litellm_params:
      model: azure/my-gpt-5-2-deployment
      api_base: https://my-resource.openai.azure.com
      api_key: os.environ/AZURE_API_KEY
    model_info:
      base_model: azure/gpt-5.2

  # ... your other deployments, unchanged
```

**2. Register the callback:**

```yaml
litellm_settings:
  callbacks: airouter_litellm.selection
```

If you already have callbacks, add it to the list:
`callbacks: ["airouter_litellm.selection", "langfuse"]`.

**3. Set the environment and restart the proxy:**

| Variable | Required | Default | Meaning |
| --- | --- | --- | --- |
| `AIROUTER_API_KEY` | yes | | API key of your AI Router organization |
| `AIROUTER_VIRTUAL_MODEL` | no | `airouter-auto` | Name of the virtual model |
| `AIROUTER_SELECT_TIMEOUT_MS` | no | `2000` | Time budget for the whole selection step; when exceeded, the default model answers |
| `AIROUTER_MODEL_MAP` | no | | JSON object from AI Router model name to your model name, e.g. `{"claude-4-5-sonnet": "vertex-sonnet"}` |
| `AIROUTER_EXCLUDE_MODELS` | no | | Comma-separated model names that must never be selected |
| `AIROUTER_LOG_LEVEL` | no | | `INFO` logs every selection, `WARNING` only problems |
| `AIROUTER_BASE_URL` | no | `https://api.airouter.io` | API endpoint |

**4. Use it.** Point applications, or a team's default model, at `airouter-auto`.
Requests to any other model are not touched.

## Model mapping

The AI Router knows models by catalog name (`gpt-5-mini`, `gemini-3.5-flash`, ...),
your proxy by your `model_name`. The callback matches them automatically. For each of
your model groups it compares, in this order, the `model_name`, `model_info.base_model`
and `litellm_params.model` with the AI Router catalog, after removing provider prefixes
(`azure/`, `vertex_ai/`, `bedrock/us.anthropic.`) and version suffixes (`@20250929`,
`-v1:0`).

- Deployments with custom names, typical on Azure, need `model_info.base_model`. LiteLLM
  uses the same field for cost tracking.
- The proxy's model list is re-read every minute, so models added later are matched
  without a restart.
- Several deployments of the same model under one `model_name` (for example two regions)
  are fine: the callback selects the model group, LiteLLM balances the deployments.
- A model group without a match is never selected and is logged once:

  ```
  airouter: model group 'vertex-sonnet' has no AI Router catalog match and is never selected; ...
  ```

  Map it with `AIROUTER_MODEL_MAP`, exclude it with `AIROUTER_EXCLUDE_MODELS`, or ask
  [support@airouter.io](mailto:support@airouter.io) to add the model to the catalog,
  for example a self-hosted model.

## What is sent to the AI Router

- Per routed request: the text content of the messages (system, user, assistant and
  tool text; images and files are left out), the names of the candidate models, whether
  tools are present, and `max_tokens` if set.
- Per executed call: decision id, model name, input and output token counts from the
  provider's usage, and latency.
- Nothing about requests to other models, and never provider keys.

## Verify

Start the proxy with `AIROUTER_LOG_LEVEL=INFO` and send a request:

```bash
curl http://localhost:4000/v1/chat/completions \
  -H "Authorization: Bearer $LITELLM_KEY" -H "Content-Type: application/json" \
  -d '{"model": "airouter-auto", "messages": [{"role": "user", "content": "Hello"}]}'
```

The proxy log shows the decision:

```
airouter_litellm INFO airouter: airouter-auto -> my-mini (gpt-5-mini) in 420 ms
```

`selection skipped` lines name the reason when the default model answered instead.
Routed calls appear in the AI Router dashboard.

## Behaviour details

- **Reasoning effort.** If the AI Router selects an effort level, it is passed on as
  `reasoning_effort` (chat) or `reasoning.effort` (Responses API), unless the client set
  one itself. Anthropic-format requests keep the provider default.
- **Fallbacks.** If LiteLLM falls back to another model after the selection, usage is
  reported for the model that actually answered.
- **Delivery of usage reports.** Reports are sent in batches every few seconds and
  retried on network errors, HTTP 429 and 5xx. They are kept in memory only, so reports
  still queued when the proxy stops are lost.

## Development

```bash
pip install -e '.[test]'
pytest                                   # unit tests with LiteLLM's real access checks
tests/e2e/run_e2e.sh "$VIRTUAL_ENV/bin"  # real proxy, mock deployments, fake AI Router
```

## License

[Apache License 2.0](LICENSE)
