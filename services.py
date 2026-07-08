from typing import Dict, Tuple, Optional, Iterator, List
import os
import io
import base64
import asyncio
import traceback
import json

import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from opentelemetry import trace
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor

from src.benchmark.data_filter import data_filter_factory
from src.data.data_interface import ProcessIndicator, ReadoutData
from src.model.common import PromptTemplates
from src.model.filter_generator import generate_ner_filter
from src.model.readout import process_data, stream_response, response_generate
from src.utils import get_available_client_and_model_group
from src.integrations.observability import init_tracer


class ClientModelGroup(BaseModel):
    data: Dict[str, Tuple[str, int]]


class ApiVersion(BaseModel):
    ci_branch: str
    ci_commit: str


class DataFilterResponse(BaseModel):
    ner_filters: Dict
    ner_results: Dict
    df_activity_group: str
    df_measure_group: str
    df_measure: str


class CompleteResponse(BaseModel):
    ner_filters: Dict
    ner_results: Dict
    readout: str
    llm_response: str
    pivot_biz_table: Dict[str, List]
    pivot_table: Dict[str, List]
    benchmark_data: Dict[str, List]
    planner_data: Dict[str, List]


def df_to_b64(df: pd.DataFrame) -> str:
    buffer = io.BytesIO()
    df.to_feather(buffer)

    return base64.b64encode(buffer.getvalue()).decode()


def b64_to_df(encoded_str: str):
    buffer = io.BytesIO(base64.b64decode(encoded_str))
    return pd.read_feather(buffer)


app = FastAPI(root_path="/ask_genome_core")

init_tracer(os.getenv("APP_ID"), os.getenv("APP_ENV", "DEV"))
tracer = trace.get_tracer(__name__)
FastAPIInstrumentor.instrument_app(app)
RequestsInstrumentor().instrument()


@app.exception_handler(Exception)
async def unhandled_exception(request: Request, exc: Exception):
    tb = ''.join(traceback.format_exception(exc.__class__, exc, exc.__traceback__))
    response = JSONResponse(
        status_code=500,
        content={
            "detail": str(exc),
            "type": exc.__class__.__name__,
            "traceback": tb,
        },
    )

    return response


@app.get("/get_api_version", response_model=ApiVersion)
async def api_version_endpoint():
    response = ApiVersion(
        ci_branch=os.getenv("CI_BRANCH"),
        ci_commit=os.getenv("CI_COMMIT"),
    )

    return response


@app.get("/get_client_and_model_group", response_model=ClientModelGroup)
async def client_and_model_group_endpoint():
    client_model_group_options = get_available_client_and_model_group()

    response = ClientModelGroup(data=client_model_group_options)

    return response


@app.post("/get_data_filter", response_model=DataFilterResponse)
async def data_filter_endpoint(client_code: str, model_group_id: str, query: str, validation: bool = True):
    os.environ["CLIENT_CODE"] = client_code
    os.environ["MODEL_GROUP_ID"] = model_group_id

    process_indicator = ProcessIndicator.from_local(client_code, int(model_group_id))
    readout_data = generate_ner_filter(client_code, int(model_group_id), query, process_indicator)

    intentions = readout_data.ner_filters["intention"]
    if validation and ("none" not in intentions):
        data_filter_class = data_filter_factory()
        data_filter_class.from_api("NA", query, readout_data.ner_filters)

    response = DataFilterResponse(
        ner_filters=readout_data.ner_filters,
        ner_results=readout_data.ner_results,
        df_activity_group=df_to_b64(readout_data.df_activity_group),
        df_measure_group=df_to_b64(readout_data.df_measure_group),
        df_measure=df_to_b64(readout_data.df_measure),
    )

    return response


async def event_stream(
    it: Optional[Iterator[str]],
    pretext: str,
    benchmark_str: str,
) -> Iterator[str]:
    if it is None:
        final_resp = pretext or PromptTemplates().rejection_response
        yield f"event: main\ndata: {json.dumps({'delta': final_resp}, ensure_ascii=False)}\n\n"
        return

    found_think = False

    for chunk in it:
        if not found_think:
            if "</think>" in chunk:
                before, after = chunk.split("</think>", 1)

                if before:
                    yield f"event: think\ndata: {json.dumps({'delta': before}, ensure_ascii=False)}\n\n"

                found_think = True

                first_main = f"{pretext}\n\n{after}" if pretext else after
                yield f"event: main\ndata: {json.dumps({'delta': first_main}, ensure_ascii=False)}\n\n"
            else:
                yield f"event: think\ndata: {json.dumps({'delta': chunk}, ensure_ascii=False)}\n\n"
        else:
            yield f"event: main\ndata: {json.dumps({'delta': chunk}, ensure_ascii=False)}\n\n"

        await asyncio.sleep(0)

    if benchmark_str:
        yield f"event: main\ndata: {json.dumps({'delta': benchmark_str}, ensure_ascii=False)}\n\n"


@app.post("/stream_response", response_class=StreamingResponse)
async def stream_response_endpoint(client_code: str, model_group_id: str, query: str, readout_data: DataFilterResponse):
    os.environ["CLIENT_CODE"] = client_code
    os.environ["MODEL_GROUP_ID"] = model_group_id

    process_indicator = ProcessIndicator.from_local(client_code, int(model_group_id))
    readout_data = ReadoutData(
        ner_filters=readout_data.ner_filters,
        ner_results=readout_data.ner_results,
        df_activity_group=b64_to_df(readout_data.df_activity_group),
        df_measure_group=b64_to_df(readout_data.df_measure_group),
        df_measure=b64_to_df(readout_data.df_measure),
    )

    context_str, data, benchmark_data, benchmark_str, planner_data, pretext, pivot_biz_table, \
        principle_pretext, overall_view, readout_adj = process_data(client_code, int(model_group_id), readout_data,
                                                                    process_indicator)

    if context_str:
        it = stream_response(query, context_str)
    else:
        it = None

    return StreamingResponse(
        event_stream(it, pretext, benchmark_str),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/get_complete_response", response_model=CompleteResponse)
async def complete_response_endpoint(client_code: str, model_group_id: str, query: str):
    os.environ["CLIENT_CODE"] = client_code
    os.environ["MODEL_GROUP_ID"] = model_group_id

    process_indicator = ProcessIndicator.from_local(client_code, int(model_group_id))
    readout_data = generate_ner_filter(client_code, int(model_group_id), query, process_indicator)

    if readout_data.ner_filters.get("data") is None:
        context_str = ""
        llm_response = ""
        pivot_biz_table = pd.DataFrame()
        data = pd.DataFrame()
        benchmark_data = pd.DataFrame()
        planner_data = pd.DataFrame()

    else:
        context_str, data, benchmark_data, benchmark_str, planner_data, pretext, pivot_biz_table, \
            principle_pretext, overall_view, readout_adj = process_data(client_code, int(model_group_id), readout_data,
                                                                        process_indicator)

        llm_response = response_generate(query, context_str)

    response = CompleteResponse(
        ner_filters=readout_data.ner_filters,
        ner_results=readout_data.ner_results,
        readout=context_str,
        llm_response=llm_response,
        pivot_biz_table=pivot_biz_table.to_dict(orient="list"),
        pivot_table=data.to_dict(orient="list"),
        benchmark_data=benchmark_data.to_dict(orient="list"),
        planner_data=planner_data.to_dict(orient="list"),
    )

    return response
