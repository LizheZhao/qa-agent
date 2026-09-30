import os
from dataclasses import dataclass, field
from jinja2 import Template
import logging
from typing import List, Dict, Any
import requests

from src.utils import load_yaml_file


logger = logging.getLogger(__name__)


# Tool name -> the prompt.yaml key holding that tool's "Response requirements" block. Each value is
# a pre-formatted block scalar injected verbatim into EXTERNAL_READOUT_PROMPT's
# {{ response_requirements }} placeholder. Which tool each page uses:
#   analytics_generator  insights_only tab1 "Response"
#   insights_report      insights_only tab2 "Insights Report"
#   roi_genome           insights_only tab3 "ROI Genome" — retrieval-only today, no consumer yet
TOOL_REQUIREMENT_KEYS = {
    "analytics_generator": "ANALYTICS_GENERATOR_REQUIREMENTS",
    "insights_report": "INSIGHTS_REPORT_REQUIREMENTS",
    "roi_genome": "ROI_GENOME_REQUIREMENTS",
}


# Few-shot examples for the denial classifier. Kept in Python rather than prompt.yaml because
# the yaml is the user-editable surface and these must not be editable from the UI.
DEFAULT_DENIAL_EXAMPLES = """\
Q: "Write me a rap about our Q2 paid search performance."
Response: "Yo, paid search pulled 1.2M in revenue, that's the story..."
Output: {"status": "denial", "denial_reason": "I can help with marketing performance questions, but not in a song or rap format. If you re-ask in a standard format, I can walk through the paid search results."}

Q: "How did paid social perform last quarter?"
Response: "Paid social delivered $4.1M in revenue in 2026Q1, compared with $3.6M in 2025Q4..."
Output: {"status": "answered", "denial_reason": null}

Q: "What was our in-store staffing cost by region?"
Response: "I do not have specific knowledge about in-store staffing costs in the information available to me."
Output: {"status": "denial", "denial_reason": "The information available to me does not cover in-store staffing costs. If you can point me to a metric that captures it, I can take another look."}

Q: "How did each of our five retail regions perform on ROI?"
Response: "The analysis covers the Northeast region, where ROI was 2.4. The other regions are not represented in the available analysis."
Output: {"status": "insufficient_info", "denial_reason": "The analysis available to me covers the Northeast region only, so I can speak to that one rather than all five. If the other regions are modeled elsewhere, I can revisit this with that analysis."}

Q: "How did digital perform this year?"
Response: "The analysis covers Paid Search and Paid Social, which together represent the digital investment. Across 2026Q1 and 2026Q2..."
Output: {"status": "answered", "denial_reason": null}

Q: "Compare 2026 spend against 2025."
Response: "The available analysis covers 2025Q3 through 2026Q2 rather than full calendar years. On that basis..."
Output: {"status": "answered", "denial_reason": null}
"""


@dataclass
class PromptTemplates:
    readout_template: Template = field(init=False)
    external_readout_template: Template = field(init=False)
    tool_requirements: dict = field(init=False)
    denial_template: Template = field(init=False)
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
        self.external_readout_template = Template(prompt_config["EXTERNAL_READOUT_PROMPT"])
        # .get rather than [] so a missing key surfaces in get_default_requirements, which only the
        # tools that use it reach, instead of crashing every page at construction.
        self.tool_requirements = {tool: prompt_config.get(key)
                                  for tool, key in TOOL_REQUIREMENT_KEYS.items()}
        self.denial_prompt_raw = prompt_config["DENIAL_CLASSIFICATION_PROMPT"]
        self.denial_template = Template(self.denial_prompt_raw)
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
    _DENIAL_START_MARKER = "Denial Response requirements:"
    _DENIAL_END_MARKER = "Denial examples:"

    @staticmethod
    def _split_instructions(raw: str, start_marker: str, end_marker: str):
        """Split a raw prompt into (head, requirements_block, tail) around an editable block.

        head excludes start_marker and tail excludes end_marker, so a caller rebuilds the
        prompt as: head + start_marker + "\\n" + block + "\\n\\n" + end_marker + tail.

        Both markers must be passed in full. "Denial Response requirements:" contains
        "Response requirements:" as a substring, so partitioning on the shorter string would
        match inside the longer heading and mis-split.

        Returns None if either marker is absent, so callers fall back to the uncustomized
        template instead of sending a malformed prompt.
        """
        head, start_sep, rest = raw.partition(start_marker)
        if not start_sep:
            logger.warning("Prompt start marker %r not found; instructions not editable.", start_marker)
            return None
        requirements_block, end_sep, tail = rest.partition(end_marker)
        if not end_sep:
            logger.warning("Prompt end marker %r not found; instructions not editable.", end_marker)
            return None
        return head, requirements_block, tail

    def _split_denial_instructions(self):
        return self._split_instructions(self.denial_prompt_raw,
                                        self._DENIAL_START_MARKER,
                                        self._DENIAL_END_MARKER)

    def get_default_denial_instructions(self) -> str:
        parts = self._split_denial_instructions()
        return parts[1].strip() if parts else ""

    def get_default_requirements(self, tool: str) -> str:
        """Return a tool's default "Response requirements" text, verbatim from prompt.yaml.

        Raises rather than returning "" on an unknown or empty tool: an empty requirements block
        would strip every guardrail from the prompt and still produce plausible output, which is
        the hardest failure mode to notice.
        """
        if tool not in TOOL_REQUIREMENT_KEYS:
            raise ValueError(f"Unknown tool {tool!r}. Known tools: "
                             f"{sorted(TOOL_REQUIREMENT_KEYS)}")

        requirements = self.tool_requirements.get(tool)
        if not (requirements or "").strip():
            raise ValueError(f"prompt.yaml key {TOOL_REQUIREMENT_KEYS[tool]!r} "
                             f"(tool {tool!r}) is missing or empty.")

        return requirements.strip()

    def build_readout_prompt(self, query: str, readout: str, custom_instructions: str = None,
                             tool: str = None) -> str:
        """Render the readout prompt. `tool` is the only switch.

        Without it the legacy READOUT_PROMPT is rendered unchanged, which is what st-main,
        orchestrator_page, chart/common and stream_response rely on. With it the external prompt
        is rendered and its "Response requirements:" block is filled from custom_instructions,
        or from the tool's defaults when custom_instructions is None or blank.
        """
        # truncated_readout = truncate_by_tokens(
        #             text=readout,
        #             buffer=500    # Rendered prompt excluding readout generally encode into <200 tokens
        #         )
        if tool is None:
            if custom_instructions is not None:
                logger.warning("custom_instructions ignored: build_readout_prompt needs tool=... "
                               "to select the external template.")
            return self.readout_template.render(
                client_specific_general=self.client_specific_general,
                client_specific_instructions=self.client_specific_instructions,
                query=query,
                readout=readout,)

        # Blank after strip means "use the defaults", so an emptied text area cannot silently
        # send a prompt with no requirements at all.
        requirements = (custom_instructions if (custom_instructions or "").strip()
                        else self.get_default_requirements(tool))

        return self.external_readout_template.render(
            client_specific_general=self.client_specific_general,
            client_specific_instructions=self.client_specific_instructions,
            response_requirements=requirements,
            query=query,
            readout=readout,)

    def build_denial_prompt(self, query: str, readout: str, response: str,
                            custom_instructions: str = None, examples: str = None) -> str:
        """Build the denial-classification prompt for an already-generated response.

        custom_instructions replaces the "Denial Response requirements:" block, the only
        UI-editable part. examples replaces DEFAULT_DENIAL_EXAMPLES and is Python-side only.
        """
        template = self.denial_template
        if custom_instructions is not None:
            parts = self._split_denial_instructions()
            if parts is not None:
                head, _, tail = parts
                template = Template(head + self._DENIAL_START_MARKER + "\n" + custom_instructions
                                    + "\n\n" + self._DENIAL_END_MARKER + tail)
        return template.render(
            client_code=os.getenv("CLIENT_CODE", ""),
            query=query,
            readout=readout,
            response=response,
            examples=DEFAULT_DENIAL_EXAMPLES if examples is None else examples,)


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
