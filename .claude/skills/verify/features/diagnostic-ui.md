# Diagnostic UI

Internal React/Vite app at `/diagnostics`: Lab (writable chat), Agents, Conversations, Runs, plus a shared node Inspector.

## Sub-features

- Lab: composer (`aria-label="Diagnostic message"`), "New chat", clarification form
- Agents: compiled graph catalog and canvas (React Flow)
- Conversations: sessions 10 per page, transcript, selected-turn trace canvas
- Runs: every router invocation, including failures
- Tabs stay mounted and keep their deep link when switched

## How to get to it (user POV)

Browse to `http://127.0.0.1:8765/diagnostics/lab` (also `/agents`, `/conversations`, `/runs?trace=<id>`) after `npm run build`.

## Driving it with the built-in browser

1. `cd frontend; npm ci; npm run build` (blocked by an npm SSL error on 2026-10-02; retry on another network).
2. Launch and doctor, then `navigate` to the URL, `read_page` and assert the nav `aria-label="Diagnostic workspace"` lists four tabs.
3. Conversations: pick a session, click an assistant response, confirm the canvas and inspector update to that turn's `trace_id`.
4. Cross-check what the UI shows against the matching `/api/diagnostics/traces/{id}` response.

## Gotchas

- Without `frontend/dist`, `/diagnostics` is a 503 placeholder page.
- Never hand-edit `frontend/dist` or `src/api/schema.d.ts`; `npm run check:api` regenerates and diffs the types.
- Vitest (`npm test`) covers components with fixtures; it is not a live-UI proof.
