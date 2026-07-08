import os
from dataclasses import dataclass
from typing import Union, List

import pandas as pd


from src.benchmark.utils import QueryRuntimeError
from src.benchmark.utils import get_table_path, get_text_path, create_question_id_dir, get_benchmark_dir_path


@dataclass
class Readout:
    id: str
    query: str
    data_table_exist: int
    pivot_biz_table_exist: int
    planner_data_exist: int
    context_str: str
    pretext: str
    principle_pretext: str


def export_dataframe(question_id: str, table_id: str, df: pd.DataFrame) -> None:
    create_question_id_dir(question_id)
    path = get_table_path(question_id, table_id)

    if len(df):
        df.to_csv(path, index=False)

    return None


def import_dataframe(question_id: str, table_id: str) -> pd.DataFrame:
    path = get_table_path(question_id, table_id)
    df = pd.read_csv(path)

    return df


def export_text(question_id: str, text_id: str, text: str) -> None:
    create_question_id_dir(question_id)
    path = get_text_path(question_id, text_id)
    with open(path, "w") as file:
        file.write(text)


def import_text(question_id: str, text_id: str) -> str:
    path = get_text_path(question_id, text_id)
    with open(path, "r") as file:
        text = file.read()

    return text


def export_readout_csv(dataset_pred: List[Union[Readout, QueryRuntimeError]]) -> None:
    readout_text_rows = []
    for readout in dataset_pred:
        row = {
            "id": readout.id,
            "query": readout.query,
        }

        if isinstance(readout, QueryRuntimeError):
            row["error"] = readout.error_message
            row["error_traceback"] = readout.error_traceback
        else:
            row["data_table_exist"] = readout.data_table_exist
            row["pivot_biz_table_exist"] = readout.pivot_biz_table_exist
            row["planner_data_exist"] = readout.planner_data_exist
            row["context_str"] = readout.context_str
            row["pretext"] = readout.pretext
            row["principle_pretext"] = readout.principle_pretext

        readout_text_rows.append(row)

    df_readout_text = pd.DataFrame(readout_text_rows)
    readout_text_path = os.path.join(get_benchmark_dir_path(), f"{os.getenv('RUN_NAME')}_readout.csv")
    df_readout_text.to_csv(readout_text_path, index=False)

    return None
