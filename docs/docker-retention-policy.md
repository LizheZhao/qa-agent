# Docker build and retention policy

The application is distributed as one `agentic-orchestration` image containing the root application and
the agent, platform, integration, and tool distributions selected by the deployment manifest. Internal
Python distributions do not receive separate images unless they become independently deployed services.

Containers are runtime instances of immutable images. They are never used as build inputs or as a
catalogue of releases.

## Build policy

Routine local and CI verification is image-free. It runs the Python quality gates, deployment
validation, dependency-projection check, and `docker compose config --quiet`. It must not invoke
`docker build`, `docker compose build`, or `docker compose up`.

An image is built only for an explicitly named release. With ds6 as the only available builder, each
authorization covers one candidate build and its verification; it is not standing permission for later
builds. The build must record the source commit and use the committed, generated dependency projections:

- `requirements-runtime.txt` is the hash-pinned third-party runtime closure; and
- `requirements-build.txt` is the hash-pinned wheel-build backend closure.

Both files are mechanically derived from `uv.lock` by
`scripts/generate_dependency_projections.py`. `uv.lock` remains the only lockfile. Internal package
version changes must leave the runtime projection unchanged when the external closure is unchanged.

The Dockerfile keeps the large third-party environment in a stable layer and installs the small
workspace wheels in a later layer. It uses a multi-stage build so uv, build tooling, source trees, tests,
and package-manager caches do not enter the runtime image. The runtime process runs as a non-root user.

## Release identity

One build produces one image digest. Multiple tags may point to that digest without creating multiple
images:

| Tag | Mutability | Purpose |
| --- | --- | --- |
| `<application-version>` | immutable | human-readable release identity, for example `0.8.0` |
| `git-<short-sha>` | immutable | source identity |
| `candidate` | temporary | rollout validation only |

Development tags such as `ckpt*`, `item*`, `review-fixes`, `scope-gate`, and repeated `local` builds are
not retained releases. Deployment records retain the resolved digest so rollback does not depend on a
moving tag.

The ds6 deployment currently uses `deployments/local.yaml` and intentionally identifies itself as
`local-development`. The application release version describes artifact provenance; the deployment name
describes its environment. A release does not imply a production deployment.

Compose never builds or pulls an image. The required release must already exist in the local Docker
daemon, and operators select its explicit tag or digest before starting the worker. The local registry at
`localhost:5000` is not part of this deployment path unless that requirement changes explicitly.

## Retention limits

| Boundary | Limit |
| --- | ---: |
| Project images | active release plus one rollback image |
| Candidate image | one, only while replacement is in progress |
| Project containers | one active container plus one temporary rollout candidate |
| Registry `agentic-orchestration` artifacts | zero unless registry deployment is explicitly required |
| Host-wide build cache observation | alert above 5 GB or 14 days |

Image limits count distinct image IDs, not tags. At the first deployment there may be no rollback; after
a later successful replacement, retain exactly one known-good predecessor.

Build cache is reported host-wide because Docker does not provide reliable project ownership for the
reported total. The threshold is an operational warning, not permission to remove shared cache.

## Scope and classification

Only resources whose repository is exactly `agentic-orchestration` are governed by this policy.
Registry-qualified project references, near-match names, bare image identifiers, and unrelated resources
are separated so uncertain ownership cannot be interpreted as permission to act.

The following are expressly outside this policy:

- all `ask-genome-*` images, containers, and registry repositories;
- all `ap*` images, containers, and registry repositories;
- near-match names such as `agentic-orchestration-experiment`;
- the `myregistry` container and its shared volume;
- registry-wide garbage collection; and
- shared or unattributable BuildKit cache.

An unresolved resource is left unchanged.

## Reporting

The footprint reporter inventories Docker without changing it:

```bash
uv run python scripts/report_docker_footprint.py --registry http://localhost:5000
uv run python scripts/report_docker_footprint.py --json --fail-on-breach
uv run python scripts/report_docker_footprint.py --resolve-image-ids
```

The reporter counts distinct image IDs, resolves in-scope containers when requested, treats an
unreachable or uninspected registry as `UNKNOWN`, and separates project policy findings from host-wide
observations. `--fail-on-breach` fails on both `BREACH` and `UNKNOWN`.

It is read-only by construction: Docker commands come from a fixed inspection allowlist, parameterized
container IDs must be hexadecimal, subprocesses use argument vectors with `shell=False`, and registry
access uses HTTP `GET` only. Unit tests enforce these restrictions.

## Replacement and cleanup

Before a release build, record the current footprint and confirm sufficient free space for the existing
images, candidate, and build overhead. Validate the candidate's health, build metadata, deployment, API,
runtime package inventory, and layer sizes before promotion. Promotion uses the already tested digest and
does not rebuild it.

After a successful replacement, prepare an exact deletion manifest containing each in-scope container
name/ID and image tag/ID, its classification, reference count, deletion reason, and recovery path. Remove
only reviewed, obsolete, exact-match `agentic-orchestration` resources. Preserve the active image and one
known-good rollback.

Never automate `docker system prune`. Remove a cache record only when its provenance proves exclusive
project ownership. An obsolete manifest in the exact registry repository may be reviewed separately,
but registry-wide garbage collection and deletion of the shared registry volume remain prohibited under
this policy.
