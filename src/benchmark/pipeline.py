import os
from typing import List, Union
import traceback
import tqdm
import pickle
import pandas as pd
import logging

from src.data.data_interface import ProcessIndicator
from src.model.filter_generator import generate_ner_filter
from src.model.readout import process_data
from src.benchmark.data_filter import DataFilter, Rejected, QueryRuntimeError, data_filter_factory
from src.benchmark.data_filter import (
    get_benchmark_dir_path, load_benchmark_csv, get_row_eval_metric,
    export_predicted_data_filter_csv, export_data_filter_comp_csv
)
from src.benchmark.readout import Readout, export_dataframe, export_readout_csv
from src.benchmark.utils import locate_error_file_in_traceback

logger = logging.getLogger(__name__)


def run_benchmark_prediction(client_code: str, model_group_id: int, benchmark_version: str, gt_file: str):
    data_filter_gt = load_benchmark_csv(gt_file)
    data_filter_pred: List[Union[DataFilter, Rejected, QueryRuntimeError]] = []
    readout_pred: List[Union[Readout, QueryRuntimeError]] = []
    process_indicator = ProcessIndicator.from_local(client_code, model_group_id)
    data_filter_class = data_filter_factory()

    for row in tqdm.tqdm(data_filter_gt):
        try:
            readout_data = generate_ner_filter(client_code, model_group_id, row.query, process_indicator)
            intentions = readout_data.ner_filters["intention"]

            if (len(intentions) == 1) and (intentions[0] == "none"):
                response_data_filter = Rejected(row.id, row.query)
                # response_readout = Readout(...)   TBD: design issue / should readout logic fork out here?
            else:
                response_data_filter = data_filter_class.from_api(row.id, row.query, readout_data.ner_filters)
        except Exception as e:
            response_data_filter = QueryRuntimeError(
                id=row.id,
                query=row.query,
                error_message=f"{str(e)}\n\n{traceback.format_exc()}",
                error_traceback=locate_error_file_in_traceback(e),
            )

        if isinstance(response_data_filter, QueryRuntimeError):
            # If data_filter contains error, pass the error class to readout
            response_readout = response_data_filter
        elif isinstance(response_data_filter, Rejected):
            # If data_filter is Rejected, create empty readout
            response_readout = Readout(
                id=row.id,
                query=row.query,
                data_table_exist=0,
                pivot_biz_table_exist=0,
                planner_data_exist=0,
                context_str="Rejected",
                pretext="",
                principle_pretext=""
            )
        else:
            try:
                context_str, data_table, _, benchmark_str, planner_data, pretext, pivot_biz_table, principle_pretext = (
                    process_data(client_code, model_group_id, readout_data, process_indicator))
                export_dataframe(row.id, "data_table", data_table)
                export_dataframe(row.id, "pivot_biz_table", pivot_biz_table)
                export_dataframe(row.id, "planner_data", planner_data)

                # if context_str:
                #     llm_response = response_generate(row.query, context_str)
                #     processed_text = f"{pretext} \n\n{llm_response} \n\n{benchmark_str}"
                # else:
                #     if pretext:
                #         processed_text = pretext
                #     else:
                #         processed_text = prompt_template.rejection_response

                response_readout = Readout(
                    id=row.id,
                    query=row.query,
                    data_table_exist=1 if len(data_table) else 0,
                    pivot_biz_table_exist=1 if len(pivot_biz_table) else 0,
                    planner_data_exist=1 if len(planner_data) else 0,
                    context_str=context_str,
                    pretext=pretext,
                    principle_pretext=principle_pretext
                )
            except Exception as e:
                response_readout = QueryRuntimeError(
                    id=row.id,
                    query=row.query,
                    error_message=f"{str(e)}\n\n{traceback.format_exc()}",
                    error_traceback=locate_error_file_in_traceback(e),
                )

        data_filter_pred.append(response_data_filter)
        readout_pred.append(response_readout)

    with open(get_benchmark_dir_path() / f"{os.getenv('RUN_NAME')}_data_filter.pkl", "wb") as f:
        pickle.dump(data_filter_pred, f)

    export_predicted_data_filter_csv(data_filter_pred)
    export_readout_csv(readout_pred)

    return data_filter_gt, data_filter_pred


def run_data_filter_report(data_filter_gt, data_filter_pred) -> None:
    benchmark_report = []
    for gt, pred in zip(data_filter_gt, data_filter_pred):
        benchmark_report.append(get_row_eval_metric(gt, pred))

    # TBD: for value, compute confusion matrix
    df_benchmark_report = pd.DataFrame(benchmark_report)
    col_mean = df_benchmark_report.mean()
    logger.info(f"Benchmark Results\n==========\n{col_mean}\n==========")

    # export comp report
    export_data_filter_comp_csv(data_filter_gt, data_filter_pred)

    return None
