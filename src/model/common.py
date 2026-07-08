import os
from dataclasses import dataclass, field
from jinja2 import Template
import logging
from typing import List, Dict, Any
import requests

from src.utils import load_yaml_file


logger = logging.getLogger(__name__)


@dataclass
class PromptTemplates:
    readout_template: Template = field(init=False)
    rejection_response: str = field(init=False)
    client_specific_general: str = field(init=False)
    client_specific_instructions: list = field(init=False)
    no_metric_rejection_message: Template = field(init=False)
    no_tactic_rejection_message: Template = field(init=False)
    out_of_scope_rejection_message: Template = field(init=False)
    no_tactic_absent_message: str = field(init=False)
    no_tactic_combination_message: str = field(init=False)
    change_intention_message: str = field(init=False)
    change_time_message: str = field(init=False)

    def __post_init__(self):
        prompt_config = load_yaml_file("./config/prompt.yaml")

        self.readout_template = Template(prompt_config["READOUT_PROMPT"])
        self.rejection_response = prompt_config["REJECTION_RESPONSE"]
        self.no_tactic_rejection_message = prompt_config["REJECTION_RESPONSE_NO_TACTIC"]
        self.no_metric_rejection_message = prompt_config["REJECTION_RESPONSE_NO_METRIC"]
        self.out_of_scope_rejection_message = prompt_config["REJECTION_RESPONSE_OUT_OF_SCOPE"]
        self.no_tactic_absent_message = prompt_config["REJECTION_RESPONSE_NO_TACTIC_ABSENT"]
        self.no_tactic_combination_message = prompt_config["REJECTION_RESPONSE_NO_TACTIC_COMBINATION"]
        self.change_intention_message = prompt_config["REJECTION_RESPONSE_CHANGE_INTENTION"]
        self.change_time_message = prompt_config["REJECTION_RESPONSE_CHANGE_TIME"]

        client_uid = f'{os.getenv("CLIENT_CODE")}-{os.getenv("MODEL_GROUP_ID")}'

        client_specific_prompt = prompt_config["CLIENT_SPECIFIC_PROMPT"]

        client_data = client_specific_prompt.get(client_uid, {})
        self.client_specific_general = client_data.get("GENERAL", "")
        self.client_specific_instructions = client_data.get("INSTRUCTIONS", [])

    # def build_readout_prompt(self, query: str, readout: str) -> str:
    #     truncated_readout = truncate_by_tokens(
    #         text=readout,
    #         buffer=500    # Rendered prompt excluding readout generally encode into <200 tokens
    #     )
    #
    #     rendered_prompt = self.readout_template.render(
    #         client_specific_general=self.client_specific_general,
    #         client_specific_instructions=self.client_specific_instructions,
    #         query=query,
    #         readout=truncated_readout
    #     )
    #
    #     return rendered_prompt
    def build_readout_prompt(self, query: str, readout: str) -> str:
        truncated_readout = truncate_by_tokens(
                    text=readout,
                    buffer=500    # Rendered prompt excluding readout generally encode into <200 tokens
                )
        rendered_prompt = self.readout_template.render(
            client_specific_general=self.client_specific_general,
            client_specific_instructions=self.client_specific_instructions,
            query=query,
            readout=readout,)

        return rendered_prompt


def tokenize_text(text: str) -> List[Dict[str, Any]]:
    headers = {
        "Content-Type": "application/json",
    }
    payload = {
        "inputs": text
    }

    response = requests.post(
        f"{os.getenv('LLM_SERVICE_URL')}/tokenize",
        headers=headers,
        json=payload,
        timeout=5 # temporarily raise to 5s
    )
    response.raise_for_status()

    return response.json()


def truncate_by_tokens(text: str, buffer: int = 50) -> str:
    max_tokens = int(os.getenv("LLM_MAX_INPUT_TOKENS")) - buffer
    tokens = tokenize_text(text)

    total_tokens = len(tokens)

    if total_tokens <= max_tokens:
        return text

    if max_tokens < 1:
        raise RuntimeError(f"Text cannot be truncated. LLM max input too small: {os.getenv('LLM_MAX_INPUT_TOKENS')}")

    truncated_tokens = tokens[:max_tokens]
    final_stop_offset = truncated_tokens[-1]["stop"]
    truncated_text = text[:final_stop_offset]

    logger.warning(
        f"Prompt was truncated from {total_tokens} tokens to {max_tokens} tokens."
    )

    return truncated_text
