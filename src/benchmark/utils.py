import os
import logging
from typing import Dict
from pathlib import Path
from dataclasses import dataclass, asdict
import traceback


from src.utils import load_yaml_file

logger = logging.getLogger(__name__)


def set_environment_variables(path='config/benchmark.yaml') -> None:
    logger.info('Setting up benchmark env variables')
    config = load_yaml_file(path)

    for key, value in config.items():
        os.environ[key] = value

    return None


@dataclass
class QueryRuntimeError:
    id: str
    query: str
    error_message: str
    error_traceback: str

    def to_csv_row(self) -> Dict[str, str]:
        return asdict(self)


def locate_error_file_in_traceback(e: Exception) -> str:
    tb = traceback.extract_tb(e.__traceback__)
    found_frame = None
    for frame in reversed(tb):
        if "ask-genome/src" in frame.filename:
            found_frame = frame
            break

    if found_frame:
        error_file = found_frame.filename.split("ask-genome", 1)[1]
    else:
        error_file = tb[-1].filename

    return error_file


def get_benchmark_dir_path(
        client_code: str = None,
        model_group_id: int = None,
        benchmark_version: str = None,
) -> Path:
    if client_code is None:
        client_code = os.getenv("CLIENT_CODE")
    if model_group_id is None:
        model_group_id = os.getenv("MODEL_GROUP_ID")
    if benchmark_version is None:
        benchmark_version = os.getenv("BENCHMARK_VERSION")

    benchmark_dir = os.getenv("BENCHMARK_DIR")
    path = Path(benchmark_dir) / client_code / str(model_group_id) / benchmark_version

    return path


def get_question_id_path(question_id):
    path = os.path.join(
        get_benchmark_dir_path(),
        os.getenv("RUN_NAME"),
        question_id
    )

    return path


def get_table_path(question_id, table_id):
    path = os.path.join(
        get_question_id_path(question_id),
        f"{table_id}.csv",
    )

    return path


def get_text_path(question_id, text_id):
    path = os.path.join(
        get_question_id_path(question_id),
        f"{text_id}.txt",
    )

    return path


def create_question_id_dir(question_id: str) -> None:
    path = get_question_id_path(question_id)
    os.makedirs(path, exist_ok=True)

    return None
