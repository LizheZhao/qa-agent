# Enterprise LLM gateway

`ChatEnterpriseGateway` is a LangChain `BaseChatModel` configured explicitly with `base_url`, `vendor`
(`claude` or `openai`), `model_name`, `client_code`, `token`, and `user`. The package does not read
environment variables. The application maps `LLM_SERVICE_URL`, `EXTERNAL_VENDOR`, `EXTERNAL_MODEL`,
`CLIENT_CODE`, `GATEWAY_TOKEN` or `TOKEN`, and `GATEWAY_USER` or `EXTERNAL_USER`.
`GATEWAY_TIMEOUT_SECONDS` is optional, must be positive, and defaults to 180 seconds to match the
established enterprise adapter behavior.

The application automatically loads a gitignored repository-root `.env` using Pydantic settings;
real environment variables override file values. Copy `.env.example` locally and never commit or
print its credentials. Compose also passes that optional `.env` into the worker. Worker startup also
requires the MongoDB settings listed in `.env.example` because conversation history has no process-memory
runtime fallback.

Requests post to `{base_url}/client/{client_code}/{vendor}/generate` with token/user authentication
headers. Claude uses content blocks, prepends system text to a user turn, merges same-role turns, and
avoids assistant prefill with `Continue.`. OpenAI uses Responses-style message blocks. `bind_tools`
uses LangChain's maintained conversion helper and normalizes Anthropic-like schemas before mapping
native definitions and every provider-specific tool choice. Native structured tool calling is the
only supported behavior. The adapter does not inject a tool-protocol system prompt, parse JSON calls
from model prose, or retry using a fallback mode. A gateway that does not support native tools fails
visibly.

Native calls return standard `AIMessage.tool_calls`. History retains prior assistant tool decisions
and `ToolMessage` call IDs. Text, native calls, model, and normalized usage are extracted from supported
Claude, OpenAI, and data-envelope forms. Errors distinguish configuration, HTTP, envelope/provider,
timeout, transport, and invalid response shape.

The gateway contract places provider data at top-level `data.response`, alongside the `result` status
object. The adapter temporarily accepts legacy `result.data`, but top-level data is authoritative.
Native tool calls use the first populated representation in this order: `data.tool_calls`,
`data.function_call`, `data.output`, provider `response.output`, then Anthropic `response.content`.
Later convenience/verbose copies are ignored, and duplicate provider call IDs within the selected
representation are emitted only once.

Live smoke tests use the same application settings and gateway factory, including automatic `.env`
loading. They are gated separately so normal test runs never contact the enterprise network. Run:

```bash
RUN_LIVE_GATEWAY_TESTS=1 uv run pytest -m live -v tests/integration/test_live_gateway.py
```

The tests make one plain request and one forced native `calculate_sum` request, then validate the returned
tool name and arguments. They do not print credentials or response content. Normal tests use
a test-only `ScriptedChatModel` and injected `MemorySessionStore` under `tests` and require no service,
database, or secret. Those test adapters are not part of the enterprise gateway distribution or worker
runtime behavior.

For a sanitized connection, envelope, latency, and native-tool diagnostic, run:

```bash
uv run python scripts/diagnose_enterprise_llm.py
uv run python scripts/diagnose_enterprise_llm.py --test-tool
```

The diagnostic hides prompt/response content by default and never prints credentials. `--show-content`
is intended only for controlled local troubleshooting.

## Live agent verification

After configuring `LLM_SERVICE_URL`, `EXTERNAL_VENDOR`, `EXTERNAL_MODEL`, `CLIENT_CODE`, a token via
`GATEWAY_TOKEN` (or `TOKEN`), and a user via `GATEWAY_USER` (or `EXTERNAL_USER`), run:

```bash
RUN_LIVE_GATEWAY_TESTS=1 uv run pytest -m live -v tests/integration/test_live_gateway.py
uv run uvicorn agentic_orchestration.main:app --host 127.0.0.1 --port 8000
```

In another shell:

```bash
curl -sS -H 'Content-Type: application/json' \
  -d '{"message":"Reply with a short greeting."}' http://localhost:8000/api/chat

curl -sS -H 'Content-Type: application/json' \
  -d '{"message":"For a marketing experiment, treatment produced 17 conversions and control produced 25. Ask the marketing-science specialist to calculate and explain the combined observed conversions."}' \
  http://localhost:8000/api/chat

curl -sS -H 'Content-Type: application/json' \
  -d '{"session_id":"RETURNED_ID","message":"What did you just calculate?"}' \
  http://localhost:8000/api/chat

curl -sS http://localhost:8000/api/sessions/RETURNED_ID/history
```

The exact prose is model-dependent. Verify the structured `calculate_sum` call, its completed status,
the numeric result, second-turn context, and `durable: true`. Restarting the worker and reading the same
session should preserve history. Do not record unsanitized responses in shared logs.
