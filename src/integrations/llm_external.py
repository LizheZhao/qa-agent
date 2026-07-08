import os
import json
import logging
import requests
import re
from typing import Iterator, Dict

from opentelemetry import trace
from src.integrations.observability import init_tracer

from langchain_core.output_parsers.json import JsonOutputParser
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_anthropic import ChatAnthropic

from src.integrations.sdk_utils import NEROutput, ResponseOutput

init_tracer(os.getenv("APP_ID"), os.getenv("APP_ENV", "DEV"))
tracer = trace.get_tracer(__name__)
logger = logging.getLogger(__name__)

ENTERPRISE_AI_CLIENT_CODE = "RETDEMO"
_ENTERPRISE_AI_MODEL_ENV = {
    "claude": "CLAUDE_EXTERNAL_MODEL",
    "openai": "OPENAI_EXTERNAL_MODEL",
    "gemini": "GEMINI_EXTERNAL_MODEL",
}


def extract_assistant_text(data: object) -> str:
    if isinstance(data, dict):
        inner = data.get("data")
        if isinstance(inner, dict) and isinstance(inner.get("response"), str):
            return inner["response"]
        if isinstance(data.get("assistant"), str):
            return data["assistant"]
        if isinstance(data.get("message"), str):
            return data["message"]
        if "reply" in data and isinstance(data["reply"], str):
            return data["reply"]
        try:
            output = data.get("output")
            if isinstance(output, list) and output:
                content = output[0].get("content")
                if isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and part.get("type") in {"output_text", "text"}:
                            if isinstance(part.get("text"), str):
                                return part["text"]
        except Exception:
            pass
    return json.dumps(data, ensure_ascii=False, indent=2)


def call_api(system_instruction: str, user_text: str, llm_vendor: str = None,
             parse_json: bool = False, model: str = None, reasoning_effort: str = None,
             tools: list = None, verbose: bool = False):
    if llm_vendor is None:
        llm_vendor = os.getenv("EXTERNAL_VENDOR")

    if llm_vendor in _ENTERPRISE_AI_MODEL_ENV:
        return call_enterprise_ai_api(llm_vendor, system_instruction, user_text, model=model,
                                      parse_json=parse_json, reasoning_effort=reasoning_effort,
                                      tools=tools, verbose=verbose)

    if llm_vendor == "deepseek":
        url = "https://api.deepseek.com/chat/completions"
        body = {
            "model": model or os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
            "messages": [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_text}
            ]
        }
        if tools:
            # deepseek is OpenAI-compatible; wrap the anthropic-style tool dicts
            body["tools"] = [{"type": "function",
                              "function": {"name": t["name"], "description": t["description"],
                                           "parameters": t["input_schema"]}} for t in tools]
    else:
        raise Exception("Unknown llm_vendor, supported: 'claude', 'openai', 'gemini', 'deepseek'")

    try:
        api_key = os.getenv("DEEPSEEK_API_KEY")
        resp = requests.post(
            url,
            json=body,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
        )
    except Exception as e:
        raise Exception(e)

    try:
        data = resp.json()
        if tools:
            # return the raw response dict; the caller extracts the tool call from it
            if verbose:
                logger.info(f"tool call response: {json.dumps(data, default=str)[:500]}")
            return data
        response = data["choices"][0]["message"]["content"]
        if parse_json:
            parser = JsonOutputParser()
            response = parser.parse(response)
    except Exception as e:
        raise Exception(e)

    return response


def _enterprise_ai_headers() -> Dict[str, str]:
    return {
        "Content-Type": "application/json",
        "X-HTTP-AUTH-USER": os.getenv("ENTERPRISE_AI_USER"),
        "X-HTTP-AUTH-TOKEN": os.getenv("ENTERPRISE_AI_TOKEN"),
    }


def call_enterprise_ai_api(provider: str, system_instruction: str, user_text: str, model: str = None,
                           parse_json: bool = False, stream: bool = False, reasoning_effort: str = None,
                           tools: list = None, verbose: bool = False):
    if provider not in _ENTERPRISE_AI_MODEL_ENV:
        raise Exception(f"Unknown provider '{provider}', supported: {list(_ENTERPRISE_AI_MODEL_ENV)}")
    if model is None:
        model = os.getenv(_ENTERPRISE_AI_MODEL_ENV[provider])
    base_url = os.getenv("ENTERPRISE_AI_BASE_URL")
    if base_url is None:
        raise Exception(f"Unknown base url for external llm")

    url = f"{base_url}/client/{ENTERPRISE_AI_CLIENT_CODE}/{provider}/generate"
    body = {
        "provider": provider,
        "model": model,
        "system_instructions": system_instruction,
        "contents": user_text,
        "stream": stream,
    }
    if reasoning_effort:
        body["reasoning"] = {"effort": reasoning_effort}
    if tools:
        body["tools"] = tools
        if verbose:
            logger.info(f"tool call request to {provider}: {[t['name'] for t in tools]}")

    try:
        resp = requests.post(url, json=body, headers=_enterprise_ai_headers())
    except Exception as e:
        raise Exception(e)

    try:
        data = resp.json()
        if tools:
            # return the raw response dict; the caller extracts the tool call from it
            if verbose:
                logger.info(f"tool call response: {json.dumps(data, default=str)[:500]}")
            return data
        response = extract_assistant_text(data)
        if parse_json:
            parser = JsonOutputParser()
            response = parser.parse(response)
    except Exception as e:
        raise Exception(e)

    return response


def call_claude_api(system_instruction: str, user_text: str, model: str = None, parse_json: bool = False, reasoning_effort: str = None):
    return call_enterprise_ai_api("claude", system_instruction, user_text, model=model, parse_json=parse_json, reasoning_effort=reasoning_effort)


def call_openai_enterprise_api(system_instruction: str, user_text: str, model: str = None, parse_json: bool = False, reasoning_effort: str = None):
    return call_enterprise_ai_api("openai", system_instruction, user_text, model=model, parse_json=parse_json, reasoning_effort=reasoning_effort)


def call_gemini_enterprise_api(system_instruction: str, user_text: str, model: str = None, parse_json: bool = False, reasoning_effort: str = None):
    return call_enterprise_ai_api("gemini", system_instruction, user_text, model=model, parse_json=parse_json, reasoning_effort=reasoning_effort)


def _anthropic_enabled() -> bool:
    return bool(os.getenv("ANTHROPIC_API_KEY"))


def call_anthropic_chat(system_instruction: str, user_text: str, parse_json: bool = False,
                        model: str = None, reasoning_effort: str = None, schema=None):
    """Response/NER generation via the langchain Anthropic chat package (when ANTHROPIC_API_KEY is set).

    When a pydantic schema is given (e.g. NEROutput / ResponseOutput from sdk_utils) it uses
    with_structured_output and returns a JSON string of the validated object, so the existing
    downstream parsers (JsonOutputParser for NER, _parse_denial_response for responses) keep
    working unchanged. Separate from call_api so the endpoint path used by insights_only /
    st-main is untouched."""
    llm = ChatAnthropic(model=model or os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6"),
                        api_key=os.getenv("ANTHROPIC_API_KEY"), max_tokens=8192)
    messages = [SystemMessage(content=system_instruction), HumanMessage(content=user_text)]
    if schema is not None:
        parsed = llm.with_structured_output(schema).invoke(messages)
        data = parsed.model_dump() if hasattr(parsed, "model_dump") else dict(parsed)
        return data if parse_json else json.dumps(data)
    result = llm.invoke(messages)
    response = result.content if isinstance(result.content, str) else str(result.content)
    if parse_json:
        response = JsonOutputParser().parse(response)
    return response


def generate_filter_response(input_question, input_prompt, field_type, additional_instruction=None):
    system_instruction = """
    You are a natural language processing assistant specialized in marketing data analysis. 
    You will receive a user question and are expected to reason through based on the task instruction. Then, extract the information and generate filters in valid json format.
    """

    if additional_instruction:
        system_instruction += additional_instruction

    user_text = input_prompt + "## input question: \n question: {question}".format(question=input_question)

    if _anthropic_enabled():
        response = call_anthropic_chat(system_instruction, user_text, parse_json=False,
                                       reasoning_effort="low", schema=NEROutput)
    else:
        response = call_api(system_instruction, user_text, parse_json=False, reasoning_effort="low")
    if "error" in response:
        return {f"error_{field_type}": response["error"]}
    return response


def generate_analysis_response(input_prompt, llm_vendor, field_type):
    _DENIAL_INSTRUCTION = """
    Respond ONLY with a JSON object, no preamble, no markdown fences, in exactly this shape:
    {"is_denial": <true|false>, "denial_reason": <string or null>, "answer": <string>}

    Set is_denial to true if the readout cannot answer the user's question. When is_denial is
    true, give a brief denial_reason and set answer to "". When is_denial is false, set
    denial_reason to null and put the full answer in answer. Emit is_denial first.
    """.strip()

    system_instruction = f"""
    You are a natural language processing assistant specialized in marketing data analysis.
    You will receive a user question and are expected to give response to the question based on the task instruction and context.

    {_DENIAL_INSTRUCTION}
    """

    user_text = input_prompt

    if _anthropic_enabled():
        response = call_anthropic_chat(system_instruction, user_text, parse_json=False, schema=ResponseOutput)
    else:
        response = call_api(system_instruction, user_text, parse_json=False)
    if isinstance(response, dict) and "error" in response:
        return {f"error_{field_type}": response["error"]}
    return response