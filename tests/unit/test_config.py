from pathlib import Path

import pytest

from agentic_orchestration.config import Settings


@pytest.mark.unit
def test_dotenv_is_loaded_and_environment_takes_precedence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / ".env").write_text(
        "LLM_SERVICE_URL=https://from-file.invalid\nEXTERNAL_MODEL=file-model\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("EXTERNAL_MODEL", "environment-model")

    settings = Settings()

    assert settings.llm_service_url == "https://from-file.invalid"
    assert settings.external_model == "environment-model"
