# Deployment and versioning

The tracked YAML deployment manifest records architectural composition: deployment identity,
application, core platform, enterprise gateway, and named agents with exact package versions,
entrypoints, metadata entrypoints, and routing eligibility.

`schemas/deployment.schema.json` validates its structure. Startup and
`scripts/validate_deployment.py` compare every declaration with `importlib.metadata.version`, validate
entrypoint syntax, import only declared agents, and fail without installing anything. Deployment
manifests, rather than package scanning, select the installed agents and entry agent.

The local manifest is `deployments/local.yaml`. Its `local-development` name describes the deployment
environment; it does not make a versioned application image mutable or unreleasable. The manifest's
deployment revision changes whenever its composition changes and is distinct from individual package
versions.

## Dependency identity

`uv.lock` is the single exact dependency closure for the workspace. Do not copy third-party versions
into the deployment manifest or introduce package-specific lockfiles.

The generated `requirements-runtime.txt` and `requirements-build.txt` files are installation projections,
not independent lockfiles. Regenerate them after dependency changes and verify them with:

```bash
uv run python scripts/generate_dependency_projections.py --check
```

The runtime projection excludes workspace distributions, so an internal-only version bump does not
invalidate the large third-party Docker layer. The build projection supplies the hash-pinned Hatchling
closure used to build workspace wheels without build isolation.

## Package and deployment versions

When an architectural package changes, keep its `pyproject.toml` version, exported agent manifest when
applicable, deployment declaration, and `uv.lock` synchronized. If the installed composition changes,
advance the deployment revision as well.

The internal `orchestration-tools` distribution is pinned in `uv.lock` but remains release-coupled to the
application and omitted from the architectural deployment manifest. Its installable package prevents
tool imports from depending on the repository working directory.

## Image identity

The Docker build embeds the deployment manifest and `/app/build-metadata.json`. The metadata generator
reads the deployment and architectural package versions from the validated manifest, then adds the
source commit and lockfile SHA-256. `/build` exposes that bounded metadata when present; a source-tree run
without generated metadata returns an equivalent safe runtime view.

Release images use one immutable application-version tag and one immutable `git-<short-sha>` tag on the
same digest. Deploy and roll back by resolved digest when possible. A temporary `candidate` tag may exist
only during rollout. Tags never replace the application, deployment, and package identities recorded
inside the image.

The current application version is `0.8.0`, with local deployment revision `0.9.0`. The local deployment
selects `router-agent==0.4.0` as its entry agent and declares `marketing-science-agent==0.3.0` and
`ask-genome-agent==0.1.0` as eligible children. Startup fails if any installed distribution, exported
manifest, agent ID, version, entrypoint, routing eligibility, capability declaration, or required tool
does not validate.

## Runtime configuration

MongoDB location and credentials are runtime configuration and are never embedded in deployment metadata
or the image. Worker startup requires a transaction-capable MongoDB deployment, verifies the configured
database, and initializes the durable session collections and unique turn index before serving requests.

The default Compose configuration runs an image already present in the local Docker daemon and has no
build stanza. It does not pull missing images. Building an image is a separately authorized release
operation governed by the [Docker build and retention policy](docker-retention-policy.md).
