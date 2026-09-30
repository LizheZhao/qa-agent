import os
import json
import requests
from typing import Dict

from opentelemetry import trace
from src.integrations.observability import init_tracer

from langchain_core.output_parsers.json import JsonOutputParser

init_tracer(os.getenv("APP_ID"), os.getenv("APP_ENV", "DEV"))
tracer = trace.get_tracer(__name__)

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
             parse_json: bool = False, model: str = None, reasoning_effort: str = None):
    if llm_vendor is None:
        llm_vendor = os.getenv("EXTERNAL_VENDOR")

    if llm_vendor in _ENTERPRISE_AI_MODEL_ENV:
        return call_enterprise_ai_api(llm_vendor, system_instruction, user_text, model=model,
                                      parse_json=parse_json, reasoning_effort=reasoning_effort)

    if llm_vendor == "deepseek":
        url = "https://api.deepseek.com/chat/completions"
        body = {
            "model": model or os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
            "messages": [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_text}
            ]
        }
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


def _post_enterprise_ai(provider: str, system_instruction: str, user_text: str, model: str = None,
                        stream: bool = False, reasoning_effort: str = None,
                        extra_body: dict = None, timeout: int = 120):
    """Build + post the gateway request, return the decoded json payload.

    extra_body is merged into the body verbatim. The gateway forwards unknown keys straight to
    the provider api, so this is the passthrough slot for provider-native fields
    (text.format, tools, max_output_tokens, verbose)."""
    if provider not in _ENTERPRISE_AI_MODEL_ENV:
        raise Exception(f"Unknown provider '{provider}', supported: {list(_ENTERPRISE_AI_MODEL_ENV)}")
    if model is None:
        model = os.getenv(_ENTERPRISE_AI_MODEL_ENV[provider])
    base_url = os.getenv("ENTERPRISE_AI_BASE_URL")
    if base_url is None:
        raise Exception("Unknown base url for external llm")

    url = f"{base_url}/client/{ENTERPRISE_AI_CLIENT_CODE}/{provider}/generate"
    body = {
        "provider": provider,
        "model": model,
        "system_instructions": system_instruction,
        "contents": user_text,
        "stream": stream,
    }
    # "none" is a truthy string; treat it as "omit" so the provider applies its own default
    if reasoning_effort and reasoning_effort != "none":
        body["reasoning"] = {"effort": reasoning_effort}
    if extra_body:
        body.update(extra_body)

    try:
        resp = requests.post(url, json=body, headers=_enterprise_ai_headers(), timeout=timeout)
    except Exception as e:
        raise Exception(e)

    return resp


def call_enterprise_ai_api(provider: str, system_instruction: str, user_text: str, model: str = None,
                           parse_json: bool = False, stream: bool = False, reasoning_effort: str = None,
                           extra_body: dict = None, timeout: int = 120):
    resp = _post_enterprise_ai(provider, system_instruction, user_text, model=model, stream=stream,
                               reasoning_effort=reasoning_effort, extra_body=extra_body, timeout=timeout)
    try:
        data = resp.json()
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


def _strictify_schema(node):
    """pydantic model_json_schema() -> openai strict json_schema. Mutates and returns node.

    Strict mode wants "additionalProperties": false and every property listed in "required"
    on each object; pydantic omits fields that have a default. A $ref must carry no siblings."""
    if not isinstance(node, dict):
        return node
    if "$ref" in node:
        return {"$ref": node["$ref"]}
    node.pop("title", None)
    node.pop("default", None)
    for definition in (node.get("$defs") or {}).values():
        _strictify_schema(definition)
    for key in ("anyOf", "oneOf", "allOf"):
        if key in node:
            node[key] = [_strictify_schema(sub) for sub in node[key]]
    if "items" in node:
        node["items"] = _strictify_schema(node["items"])
    if node.get("type") == "object":
        properties = node.setdefault("properties", {})
        for name in list(properties):
            properties[name] = _strictify_schema(properties[name])
        node["additionalProperties"] = False
        node["required"] = list(properties)
    return node


def _json_schema_format(schema_model, strict: bool = True) -> dict:
    """The responses-api "text" field that forces a schema-conforming json answer."""
    return {"format": {"type": "json_schema",
                       "name": schema_model.__name__,
                       "strict": strict,
                       "schema": _strictify_schema(schema_model.model_json_schema())}}


def _extract_responses_output(data: object) -> str:
    """Assistant text out of a verbose (raw provider) responses-api payload.

    extract_assistant_text cannot be reused here: on a reasoning model output[0] is a reasoning
    item with empty content, so an output[0]-only lookup misses the message."""
    response = data.get("data", {}).get("response") if isinstance(data, dict) else None
    if not isinstance(response, dict):
        raise Exception(f"unexpected gateway payload: {json.dumps(data, ensure_ascii=False)[:500]}")
    status = response.get("status")
    if status and status != "completed":
        reason = (response.get("incomplete_details") or {}).get("reason")
        raise Exception(f"response status '{status}'" + (f", reason '{reason}'" if reason else ""))
    parts = []
    for item in response.get("output") or []:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        for part in item.get("content") or []:
            if isinstance(part, dict) and part.get("type") in {"output_text", "text"}:
                parts.append(part.get("text") or "")
    if not parts:
        raise Exception(f"no message item in output: {json.dumps(response.get('output'))[:500]}")
    return "".join(parts)


def call_enterprise_ai_structured(system_instruction: str, user_text: str, schema_model,
                                  model: str = None, reasoning_effort: str = "low",
                                  max_output_tokens: int = 4096, strict: bool = True,
                                  timeout: int = 120):
    """Schema-forced structured output through the gateway; returns a validated pydantic instance.

    Pinned to openai: text.format is responses-api specific and has no equivalent on the
    gemini / claude providers. No provider sdk is involved - the schema is derived from the
    model with plain pydantic and the answer is validated back with model_validate_json."""
    extra_body = {
        "verbose": True,
        "max_output_tokens": max_output_tokens,
        "text": _json_schema_format(schema_model, strict=strict),
    }
    resp = _post_enterprise_ai("openai", system_instruction, user_text, model=model,
                               reasoning_effort=reasoning_effort, extra_body=extra_body,
                               timeout=timeout)
    resp.raise_for_status()
    text = _extract_responses_output(resp.json())
    try:
        return schema_model.model_validate_json(text)
    except Exception as e:
        raise ValueError(f"parse failed: {e}; payload was {text[:500]}")


def generate_filter_response(input_question, input_prompt, field_type, additional_instruction=None):
    system_instruction = """
    You are a natural language processing assistant specialized in marketing data analysis. 
    You will receive a user question and are expected to reason through based on the task instruction. Then, extract the information and generate filters in valid json format.
    """

    if additional_instruction:
        system_instruction += additional_instruction

    user_text = input_prompt + "## input question: \n question: {question}".format(question=input_question)

    response = call_api(system_instruction, user_text, parse_json=False, reasoning_effort="low")
    if "error" in response:
        return {f"error_{field_type}": response["error"]}
    return response


def generate_analysis_response(input_prompt, llm_vendor, field_type):
    system_instruction = f"""
    You are a natural language processing assistant specialized in marketing data analysis.
    You will receive a user question and are expected to give response to the question based on the task instruction and context.
    """

    user_text = input_prompt

    response = call_api(system_instruction, user_text, parse_json=False)
    if isinstance(response, dict) and "error" in response:
        return {f"error_{field_type}": response["error"]}
    return response


def generate_denial_response(input_prompt, llm_vendor, field_type="denial"):
    """Classify an already-generated response as a denial. Returns the raw JSON text;
    parsing is done by readout._parse_denial_response so both paths share one parser."""
    system_instruction = """
    You are a strict classifier auditing a marketing analysis assistant's own answer.
    Decide only whether that answer denied the user's request, and return the requested JSON
    object. Do not rewrite or restate the answer. Do not add commentary or code fences.
    """

    response = call_api(system_instruction, input_prompt, llm_vendor=llm_vendor,
                        parse_json=False)
    if isinstance(response, dict) and "error" in response:
        return {f"error_{field_type}": response["error"]}
    return response