from typing import Iterator, Dict
import requests
import os

from openai import OpenAI
from opentelemetry import trace

from src.integrations.observability import init_tracer

init_tracer(os.getenv("APP_ID"), os.getenv("APP_ENV", "DEV"))
tracer = trace.get_tracer(__name__)


def generate_text(
        prompt,
        temperature=0.2,
        max_new_tokens=3000
) -> str:
    if os.getenv("LLM_SERVICE_URL") is None:
        raise RuntimeError("LLM_SERVICE_URL is not defined.")

    client = OpenAI(
        api_key="none",
        base_url=f'{os.getenv("LLM_SERVICE_URL")}/v1'
    )

    with tracer.start_as_current_span("llm.generate"):
        response = client.chat.completions.create(
            model=os.getenv("LLM_MODEL_NAME"),
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            temperature=temperature,
            max_completion_tokens=max_new_tokens
        )

    response_str = response.choices[0].message.content

    return response_str


def stream_text(
        prompt: str,
        temperature=0.2,
        max_new_tokens=3000
) -> Iterator[str]:
    if os.getenv("LLM_SERVICE_URL") is None:
        raise RuntimeError("LLM_SERVICE_URL is not defined.")

    client = OpenAI(
        api_key="none",
        base_url=f'{os.getenv("LLM_SERVICE_URL")}/v1'
    )

    stream = client.chat.completions.create(
        model=os.getenv("LLM_MODEL_NAME"),
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        stream=True,
        temperature=temperature,
        max_completion_tokens=max_new_tokens
    )

    for chunk in stream:
        content = chunk.choices[0].delta.content
        if content:
            yield content


def generate_structured_response(
        prompt: str,
        dimension: str,
        response_structure: Dict[str, Dict],
        temperature=0.2,
        top_k=10,
        top_p=0.9,
        repetition_penalty=1.03,
        max_new_tokens=512
) -> Dict:
    url = f"{os.getenv('LLM_SERVICE_URL')}/generate"
    headers = {
        "Content-Type": "application/json"
    }

    payload = {
        "inputs": prompt,
        "parameters": {
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "repetition_penalty": repetition_penalty,
            "max_new_tokens": max_new_tokens,
            "grammar": {
                "type": "json",
                "value": {
                    "properties": response_structure,
                    "required": list(response_structure.keys())
                }
            }
        }
    }

    with tracer.start_as_current_span("llm.structured_output", attributes={"data_filter.dimension": dimension}):
        response = requests.post(url, headers=headers, json=payload)
        if response.status_code == 200:
            response_data = response.json()
        else:
            raise RuntimeError(f"Request failed with status code: {response.status_code} \n {response.content}")

    return response_data
