# Frontend engineering contract

This subtree is the internal diagnostic UI, not a customer-facing product. It is a strict TypeScript,
React, and Vite application served beneath `/diagnostics/`.

## Application and feature ownership

- **New standard:** `src/app/App.tsx` owns application composition: providers, top-level navigation or
  routing, error boundaries, and the outer layout. It must not call feature API methods, implement
  trace/session/chat workflows, run domain projections, or render feature-detail markup such as the
  composer, inspector, browser lists, or trace views.
- **New standard:** code owned by one domain belongs under `src/features/<domain>/`, including its React
  components, request/state orchestration, feature-only types, tests, and CSS. Keep a small implementation
  together; do not create folders, hooks, context providers, or barrel files merely to split code.
- Do not replace an application monolith with a single feature monolith such as `useApp`,
  `AppController`, or one catch-all feature stylesheet. A feature boundary should own a named workflow
  or view and expose an explicit component or hook interface to its caller.
- **New standard:** put a component in `src/components/` only when at least two feature directories use
  it and its props do not encode one feature's domain model. Otherwise keep it with its owning feature.
- Preserve behavior during ownership-only refactors. Changes to requests, URLs, selection behavior, or
  rendered states require explicit task scope and tests; moving code is not itself authorization to
  alter them.

## API, state, and projection boundaries

- FastAPI OpenAPI is the API source of truth. `src/api/schema.d.ts` is generated and must not be edited.
  `src/api/types.ts` may provide aliases to generated schemas but must not duplicate their field shapes.
- Keep shared fetch mechanics and `ApiError` normalization in `src/api/client.ts`. An endpoint wrapper
  used by only one feature belongs with that feature; cross-feature endpoint wrappers may remain in
  `src/api/`. All wrappers use generated response types and the shared error behavior.
- Keep renderer-neutral trace transformation in `features/traces/projection.ts` and canvas layout
  transformation in `canvas-model.ts`. Those modules must not import React, browser globals, or perform
  network requests. Keep their Vitest tests colocated.
- Server-resource loading belongs to the feature that displays it. Transient UI selections may stay in
  the nearest common owner; do not introduce application-wide context solely to avoid passing props.
- Preserve the `/diagnostics/` Vite base and documented Lab, Agents, Conversations, and Runs URL
  semantics. Encode opaque IDs when placing them in path segments or query strings.

## Styling

- **New standard:** `src/styles/global.css` is limited to resets, typography, CSS custom properties used
  across the application, and element-level application defaults. Feature class selectors do not belong
  there.
- **New standard:** import feature/component CSS from its owning feature or component. Split styles by
  ownership; do not move all current selectors into one replacement stylesheet.
- Add an application-level CSS custom property only when more than one feature uses the value. Keep
  one-off layout and visual values local to their owner.

## Verification

Use Node.js 24.15 or newer on the Node 24 LTS line (or Node 26+). `npm ci` is the reproducible setup
command when dependencies are not already installed. For frontend source or configuration changes, run:

```bash
cd frontend
npm test
npm run build
```

`npm run build` includes TypeScript checking; `npm run typecheck` is an optional faster iteration check.
If FastAPI routes/response models or the generated frontend API contract changed, also run:

```bash
cd frontend
npm run check:api
```

There is currently no frontend lint or formatting script. Do not invent or report one. Run root gates
only when the change also touches their stated scope.
