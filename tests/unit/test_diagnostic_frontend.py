from pathlib import Path

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from starlette.routing import Mount
from starlette.staticfiles import StaticFiles

from agentic_orchestration.diagnostics.frontend import configure_diagnostic_frontend


@pytest.mark.unit
async def test_missing_frontend_build_returns_clear_diagnostic(tmp_path: Path) -> None:
    app = FastAPI()
    configure_diagnostic_frontend(app, dist_path=tmp_path / "missing")

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/diagnostics/sessions/example")

    assert response.status_code == 503
    assert "npm install" in response.text


@pytest.mark.unit
async def test_built_frontend_serves_index_and_assets(tmp_path: Path) -> None:
    dist = tmp_path / "dist"
    assets = dist / "assets"
    assets.mkdir(parents=True)
    (dist / "index.html").write_text("<title>built diagnostics</title>", encoding="utf-8")
    (assets / "app.js").write_text("export {};", encoding="utf-8")
    app = FastAPI()
    configure_diagnostic_frontend(app, dist_path=dist)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        deep_link = await client.get("/diagnostics/sessions/example")

    assert deep_link.status_code == 200
    assert "built diagnostics" in deep_link.text
    asset_mount = next(
        route
        for route in app.routes
        if isinstance(route, Mount) and route.path == "/diagnostics/assets"
    )
    assert isinstance(asset_mount.app, StaticFiles)
    asset_path, asset_stat = asset_mount.app.lookup_path("app.js")
    assert asset_stat is not None
    assert Path(asset_path).read_text(encoding="utf-8") == "export {};"
