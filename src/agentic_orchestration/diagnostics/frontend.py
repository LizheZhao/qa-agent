"""Static Vite build serving restricted to the diagnostic browser route."""

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from starlette.responses import Response

DEFAULT_FRONTEND_DIST = Path(__file__).parents[3] / "frontend" / "dist"


def configure_diagnostic_frontend(
    application: FastAPI, *, dist_path: Path = DEFAULT_FRONTEND_DIST
) -> None:
    index_path = dist_path / "index.html"
    assets_path = dist_path / "assets"
    if assets_path.is_dir():
        application.mount(
            "/diagnostics/assets",
            StaticFiles(directory=assets_path),
            name="diagnostic-assets",
        )

    @application.get("/diagnostics", include_in_schema=False)
    @application.get("/diagnostics/{client_path:path}", include_in_schema=False)
    async def diagnostic_frontend(client_path: str = "") -> Response:
        del client_path
        if index_path.is_file():
            return HTMLResponse(index_path.read_text(encoding="utf-8"))
        return HTMLResponse(
            """
            <!doctype html>
            <title>Diagnostic UI is not built</title>
            <main style="font: 15px system-ui; max-width: 640px; margin: 64px auto;">
              <h1>Diagnostic UI is not built</h1>
              <p>
                Run <code>cd frontend &amp;&amp; npm install &amp;&amp; npm run build</code>,
                then reload.
              </p>
            </main>
            """,
            status_code=503,
        )
