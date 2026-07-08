import os
from dataclasses import dataclass
from typing import List, Dict
import requests

from opentelemetry import trace

from src.integrations.observability import init_tracer

init_tracer(os.getenv("APP_ID"), os.getenv("APP_ENV", "DEV"))
tracer = trace.get_tracer(__name__)


@dataclass
class Prediction:
    label: int
    label_str: str
    probabilities: Dict[str, float]


@dataclass
class MultiTaskPredictions:
    predictions: List[Dict[str, Prediction]]
    architectures: List[str]
    classifier_model_version: str
    classifier_model_hash: str


def batch_request(texts: List[str]) -> MultiTaskPredictions:
    request_data = {
        "texts": texts
    }

    endpoint = f"{os.getenv('CLASSIFIER_SERVICE_URL')}/predict"
    with tracer.start_as_current_span("classifier"):
        response = requests.post(endpoint, json=request_data)

    if response.status_code == 200:
        response_data = response.json()
    else:
        raise RuntimeError(f"Request failed with status code: {response.status_code} \n {response.content}")

    normalized_data: List[Dict[str, Prediction]] = []

    for i, row in enumerate(response_data['predictions']):
        normalized_data.append({})
        for k, _ in row.items():
            normalized_data[i][k] = Prediction(
                label=row[k]["label"],
                label_str=row[k]["label_str"],
                probabilities=row[k]["probabilities"],
            )

    normalized_response = MultiTaskPredictions(
        predictions=normalized_data,
        architectures=response_data["architectures"],
        classifier_model_version=response_data["classifier_model_version"],
        classifier_model_hash=response_data["classifier_model_hash"],
    )

    return normalized_response


def single_request(text: str) -> Dict[str, str]:
    normalized_response = batch_request([text])
    normalized_prediction = {k: v.label_str for k, v in normalized_response.predictions[0].items()}
    probabilities = {k: v.probabilities for k, v in normalized_response.predictions[0].items()}
    normalized_prediction['probabilities'] = probabilities
    return normalized_prediction
