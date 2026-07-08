import os
import json
import re
import random
import pandas as pd
import numpy as np
import copy
import time
import logging
import traceback
from collections import Counter
from typing import Dict
from datetime import datetime
from dateutil.relativedelta import relativedelta
from collections import defaultdict

import spacy
from dns.rdtypes.util import priority_processing_order
from spacy.lang.en import English
import nltk
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from langchain_core.output_parsers.json import JsonOutputParser
from src.data.data_interface import NerData, NerDataConfig, NerFilters, SpacyConfig, ReadoutData, ProcessIndicator
import src.model.data_filtering as legacy_filtering
import src.model.data_filtering_other_category as general_filtering
from src.integrations.llm import generate_text, generate_structured_response
from src.integrations.embedding import query_embeddings, get_similarity_score, get_cached_embeddings
from src.integrations.classifier import single_request
from src.utils import load_yaml_file
from src.model.common import PromptTemplates

logger = logging.getLogger(__name__)


### get ner filter from llm / bert ###

def get_predefined_list(fields: list, df: pd.DataFrame) -> str:
    selections = {k: df[k].dropna().unique() for k in fields if k in df.columns}
    predefined_list = '\n'.join(
        [f'{k}: {[val for val in v if np.nan_to_num(val) and val != "overall"]}' for k, v in selections.items()])
    return predefined_list


def get_answer_format(dimension: str, fields: list, df: pd.DataFrame, custom_cols: list = None) -> str:
    if not custom_cols:
        custom_cols = []
    if dimension == 'classification':
        answer_format = json.dumps({"intention": "one of the relevant values from predefined list or 'irrelevant'"})
    elif dimension == 'tag':
        answer_format = json.dumps({k: "'relevant' or 'irrelevant'" for k in fields})
    elif dimension == 'trend':
        answer_format = json.dumps(
            {"trend": "'yes' or 'no'", "rank": "'top' or 'bottom' or 'na'", "how_many": "number or 'na'"})
    elif set(fields).intersection(set(custom_cols)):
        answer_format = json.dumps(
            {k: f"'irrelevant' or 'all' or 'specific'" for k in fields})
    else:
        selections = {k: df[k].dropna().unique() for k in fields if k in df.columns}
        answer_format = json.dumps(
            {k: f"'irrelevant' or 'all' or list of relevant entries from {list(v)}" for k, v in selections.items()})
    return answer_format


def get_static_examples(df: pd.DataFrame, fields: list) -> pd.DataFrame:
    tmp = df[df['valid_query_columns'].fillna('').apply(
        lambda x: any(f for f in fields if f in [xval.strip() for xval in x.split(",")]))]
    return tmp


def calculate_embedding_similarity_scores(query: str, nerData: NerData) -> NerData:
    if len(nerData.df_examples) > 0:
        example_embeddings = get_cached_embeddings(nerData.df_examples['query'].tolist())
        target_embeddings = query_embeddings(query)
        similarities = get_similarity_score(target_embeddings, example_embeddings)
        nerData.df_examples['similarity_score'] = similarities
    else:
        logger.info(f"Empty in-context examples dataframe. No similarity scores calculated.")
    return nerData


def get_similar_examples(df: pd.DataFrame, n: int, ascending: bool = False) -> pd.DataFrame:
    """
    given query and df, select most n and least m similar questions as examples
    """
    tmp = df.sort_values(by=['similarity_score'], ascending=ascending).head(n)
    return tmp


def construct_learning_examples(fields: list, examples_df: pd.DataFrame, custom_cols: list = None,
                                top: int = 10, bot: int = 5, n_total=30) -> str:
    if not custom_cols:
        custom_cols = []

    enabled_examples_df = examples_df[examples_df['enabled'] == 1].reset_index(drop=True)

    # static and dynamic learning examples
    static_df, dynamic_df = enabled_examples_df[enabled_examples_df['static'] == 1], enabled_examples_df[
        enabled_examples_df['static'] == 0]

    # get valid static df for ner fields
    selected_static_df = get_static_examples(static_df, fields)

    # get dynamic examples based on similarity scores
    selected_dynamic_df1 = get_similar_examples(dynamic_df, n=top, ascending=False)
    selected_dynamic_df2 = get_similar_examples(dynamic_df, n=bot, ascending=True)

    # combine
    selected_examples = pd.concat([selected_static_df, selected_dynamic_df1, selected_dynamic_df2]).reset_index(
        drop=True).drop_duplicates(['query']).head(n_total).to_dict(orient='records')  # limit to 30

    # convert to string
    constructed_example = ''
    for example in selected_examples:
        ans = {}
        for field in fields:
            a = example.get(field, 'irrelevant')
            if a not in ['irrelevant', 'relevant', 'all', 'na', 'specific'] and not any(
                    [x for x in fields if x in ['intention', 'trend', 'how_many', 'rank']]):
                if isinstance(a, str):
                    a = [x.strip() for x in a.split(",")]
                else:
                    a = "irrelevant"
            if (field in custom_cols) and isinstance(a, list):
                a = "specific"
            ans[field.replace("_external", "")] = a
        q = example['query']
        constructed_example += 'question: ' + q.lower() + '\n'
        constructed_example += 'answer: ' + json.dumps(ans) + '\n\n'
    return constructed_example


def get_few_shot_learning_examples_for_time():
    example_questions = ["What are the results from the last 4 quarters?",
                         "Show me the data for last year.",
                         "Give me the performance data from January to March 2025.",
                         "Compare the first half of the last two years.",
                         "How did we perform in April 2025?",
                         "Compare the performance of search in the last three Q1s.",
                         "How did PMAX campaigns perform compared to last year?",
                         "How did PMAX campaigns perform since 2019?",
                         "how has the pd handraiser campaign performed over the last 3 years?",
                         "what is our most efficient marketing tactic over this year?",
                         "what have category impacts been to the pd business over the last year?",
                         "How have search trends changed year over year?",
                         "what was the month-over-month trend for roi in 2024?",
                         "what was the household penetration efficiencies in 2024 h2 and q2?",
                         "Show campaign impact.",
                         "what's the contribution before 2019?",
                         "what was the year - over - year trend for total shopper in 2024",
                         "How much did the paid search ROI change from Q3 FY25 to Q4 FY25?",
                         ]
    example_answers = [{"period_type": "quarter", "num_periods": 4},
                       {"period_type": "year", "num_periods": 1},
                       {"start_date": "202501", "end_date": "202503"},
                       {"period_type": "half year", "num_periods": 2, "specific_sub_period": "H1"},
                       {"start_date": "202504", "end_date": "202504"},
                       {"period_type": "quarter", "num_periods": 3, "specific_sub_period": "Q1"},
                       {"period_type": "year", "num_periods": 2, "include_current": True},
                       {"start_date": "201901"},
                       {"period_type": "year", "num_periods": 3},
                       {"period_type": "year", "num_periods": 1, "include_current": True},
                       {"period_type": "year", "num_periods": 1},
                       {},
                       {"start_date": "202401", "end_date": "202412"},
                       {"period_type": "quarter", "start_date": "202404", "end_date": "202412"},
                       {},
                       {"start_date": "201701", "end_date": "201812"},
                       {"start_date": "202401", "end_date": "202412"},
                       {"start_date": "202507", "end_date": "202512", "period_type": "quarter", "num_periods": 2},
                       ]

    example_blocks = []
    for idx, (question, params) in enumerate(zip(example_questions, example_answers)):
        if not params:
            output = {"start": "irrelevant", "end": "irrelevant"}
        else:
            output = generate_time_periods(**params)
        block = f"""Input {idx}:  
    {question}  
    Output {idx}:  
    {json.dumps(output)}"""
        example_blocks.append(block)

    # Join all blocks for prompt
    final_prompt = "\n\n".join(example_blocks)
    return final_prompt


def generate_time_periods(
        period_type: str = None,
        num_periods: int = None,
        specific_sub_period: str = None,
        start_date: str = None,
        end_date: str = None,
        today: datetime = None,
        include_current: bool = False,
):
    """
    Generate start and end periods in YYYYMM format based on period_type, num_periods, sub_period, or specific dates.

    Rules:
    - If period_type + num_periods are given:
        - Only generate periods that have ended (except for type 'year')
    - If period_type + num_periods + specific_sub_period:
        - Return that many start/end ranges for that sub-period in past years.
    - If start and end given → return directly.
    - If only start → end = today
    - If only end → start = 201701
    """
    if today is None:
        today = datetime.today()

    def to_yyyymm(dt):
        return dt.strftime("%Y%m")

    def get_sub_period_range(year, tag):
        mapping = {
            "Q1": (1, 3),
            "Q2": (4, 6),
            "Q3": (7, 9),
            "Q4": (10, 12),
            "H1": (1, 6),
            "H2": (7, 12),
        }
        if tag not in mapping:
            raise ValueError(f"Invalid sub-period tag: {tag}")
        m_start, m_end = mapping[tag]
        return f"{year}{m_start:02d}", f"{year}{m_end:02d}"

    offset = 0
    if include_current:
        offset = 1
    elif specific_sub_period:
        s, e = get_sub_period_range(today.year, specific_sub_period)
        if str(to_yyyymm(today)) > e:
            offset = 1

    # === Direct Date Range ===
    if start_date and end_date:
        return {"start": [start_date], "end": [end_date]}
    elif start_date:
        return {"start": [start_date], "end": [to_yyyymm(today)]}
    elif end_date:
        return {"start": ["201701"], "end": [end_date]}

    # === Specific Sub-Periods ===
    if period_type and num_periods and specific_sub_period:
        results = {"start": [], "end": []}
        for i in range(num_periods):
            year = today.year - (num_periods - offset - i)
            s, e = get_sub_period_range(year, specific_sub_period)
            results["start"].append(s)
            results["end"].append(e)
        return results

    # === Regular Past Periods ===
    if not (period_type and num_periods):
        return {"start": "irrelevant", "end": "irrelevant"}

    results = {"start": [], "end": []}
    current = today.replace(day=1)

    # Align to latest *completed* period
    if period_type == "month":
        current -= relativedelta(months=1)

    elif period_type == "quarter":
        if today.month in [1, 2, 3]:
            current = datetime(today.year, 1, 1)
        elif today.month in [4, 5, 6]:
            current = datetime(today.year, 4, 1)
        elif today.month in [7, 8, 9]:
            current = datetime(today.year, 7, 1)
        else:
            current = datetime(today.year, 10, 1)

    elif period_type == "half year":
        if today.month <= 6:
            current = datetime(today.year, 1, 1)
        else:
            current = datetime(today.year, 7, 1)

    elif period_type == "year":
        current = datetime(today.year, 1, 1)
    else:
        raise ValueError(f"Invalid period type: {period_type}")

    # Generate backwards
    for i in range(num_periods):
        if period_type == "month":
            start = current - relativedelta(months=(num_periods - offset - i))
            end = start
        elif period_type == "quarter":
            start = current - relativedelta(months=(num_periods - offset - i) * 3)
            end = start + relativedelta(months=2)
        elif period_type == "half year":
            start = current - relativedelta(months=(num_periods - offset - i) * 6)
            end = start + relativedelta(months=5)
        elif period_type == "year":
            start = datetime(current.year - (num_periods - offset - i), 1, 1)
            end = datetime(start.year, 12, 1)
        results["start"].append(to_yyyymm(start))
        results["end"].append(to_yyyymm(end))

    return results


def construct_prompt(questionnaire: dict, dimension: str, query: str, examples_df: pd.DataFrame, data: pd.DataFrame,
                     level_type: dict, enterprise_ai_ner: bool = False, top: int = 10, bot: int = 5, n_total=30) -> str:
    prompt_config = load_yaml_file("./config/prompt.yaml")
    custom_cols = level_type.get("general_tagging", []) + level_type.get("general_level", [])
    base_prompt = questionnaire.get('prompt', '')
    fields = questionnaire.get('field', [])
    if set(fields).intersection(set(custom_cols)) and not enterprise_ai_ner:
        base_prompt = prompt_config['CUSTOM_FIELD_BASE_PROMPT']
    if dimension == "time":
        base_prompt = prompt_config["DYNAMIC_TIME_PROMPT"]
    context = questionnaire.get('context', '')
    try:
        if dimension == "time":
            examples = get_few_shot_learning_examples_for_time()
        elif enterprise_ai_ner:
            external_fields = [x+"_external" if x+"_external" in examples_df.columns else x for x in fields]
            examples = construct_learning_examples(external_fields, examples_df, custom_cols, top, bot, n_total)
        else:
            examples = construct_learning_examples(fields, examples_df, custom_cols, top, bot, n_total)
    except Exception as e:
        logger.info("Failed to construct learning examples, error: {}".format(e))
        examples = ""
    predefined_list = get_predefined_list(fields, data)
    answer_format = get_answer_format(dimension, fields, data, custom_cols)
    now = datetime.now()
    current_month_name = now.strftime("%B")
    current_month = now.month
    current_year = now.year
    date = f'{current_month_name}, {current_year} ({current_year}{current_month:02d})'
    prompt = base_prompt.format(cols=', '.join(fields), predefined_list=predefined_list,
                                question=query, example=examples, answer_format=answer_format,
                                context=context, date=date)
    return prompt


def construct_prompt_deepseek(questionnaire: dict, dimension: str, query: str, examples_df: pd.DataFrame,
                              data: pd.DataFrame, top: int = 10, bot: int = 5) -> str:
    base_template = """
    You are a natural language processing engine assistant specialized in marketing data analysis.
    The Assistant is asked to perform a task given a query and instruction. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.

    Query: {question}

    Task Instruction: {instruction}

    Answer Process: {answer_process}

    Assistant: {think_tag}
    """

    answer_process = """
    1. Analyze the query and the task instruction, including special instructions
    2. Give explicit reasoning and enclosed within <think> </think> tags
    3. Format the output as JSON with brand and brand_focused_marketing as keys
    """

    instruction = questionnaire.get('instruction', '')
    fields = questionnaire.get('field', [])
    predefined_list = get_predefined_list(fields, data)
    examples = construct_learning_examples(fields, examples_df, top=top, bot=bot)

    answer_process = answer_process.format(cols=','.join(fields))
    instruction = instruction.format(cols=','.join(fields), predefined_list=predefined_list)
    prompt = base_template.format(question=query, instruction=instruction, answer_process=answer_process,
                                  example=examples, think_tag="<think></think>")
    return prompt


def _check_activity_group(ner_dict: dict, fields=None) -> bool:
    # if all tags and business dimensions are irrelevant and intention is not sales
    # check activity_group
    if fields is None:
        fields = ['programmatic', 'hispanic', 'retailer', 'halo', 'business_driver', 'media_channel', 'platform']
    if all([ner_dict.get(x, 'irrelevant') == 'irrelevant' for x in fields]):
        if ner_dict.get('intention', '') != 'sales':
            return True
    return False


def _check_intention_rejection(pred_intention: list, rejection_labels: list, available_labels: list) -> bool:
    # if 'none' is in pred intention list, will reject
    if any(x for x in pred_intention if x in rejection_labels):
        return True
    # if all pred intentions are not valid, will reject
    if not any(x in available_labels for x in pred_intention):
        return True
    return False


def get_metric_by_intention(intention_filter: dict,
                            nerDataConfig: NerDataConfig) -> dict:
    # get metrics based on intention
    metric_info = nerDataConfig.metric_info[intention_filter["intention"][0]]

    # TBD: revert temp override main_metric key
    metric_info["main_metric"] = metric_info.pop("mainMetric")
    metric_info.pop("rankMetric")
    metric_info.pop("sortMetric")
    # TBD: revert temp override main_metric key
    return metric_info


def get_data_level_by_intention(nerDataConfig: NerDataConfig,
                                nerData: NerData, metric_info: dict) -> tuple[list, list, pd.DataFrame]:
    # get data levels and ignore fields
    data_levels = nerDataConfig.data_levels
    data_levels, ignore_fields = (data_levels["biLevels"], []) if metric_info["data"] == 'bi' \
        else (data_levels["brLevels"], data_levels["brLevelsToIgnore"])
    # assign data
    filtered_data = nerData.df_bi.copy() if metric_info["data"] == 'bi' else nerData.df_br.copy()
    return data_levels, ignore_fields, filtered_data


def construct_response_structure(dimension: str,
                                 fields: list,
                                 nerDataConfig: NerDataConfig,
                                 filtered_data: pd.DataFrame,
                                 level_type: dict) -> dict:
    if dimension == "classification":
        valid_options = list(nerDataConfig.metric_info.keys())
        response_structure = {
            "intention": {
                "type": "string",
                "enum": valid_options
            }
        }
    elif dimension == "trend":
        response_structure = {
            # "trend": {
            #     "type": "string",
            #     "enum": ["yes", "no"]
            # },
            # "rank": {
            #     "type": "string",
            #     "enum": ["top", "bottom", "na"]
            # },
            "how_many": {
                "type": "string"
            }
        }
    elif any(f for f in fields if f in level_type.get('general_level', []) + level_type.get('general_tagging', [])):
        response_structure = {
            k: {
                "type": "string",
                "enum": ["all", "irrelevant", "specific"]
            } for k in fields
        }
    elif dimension not in ["product", "time"]:
        string_valid_options = ["all", "irrelevant"]
        response_structure = {
            k: {
                "type": ["string", "array"],  # Supports both string and array
                "anyOf": [
                    {
                        "type": "string",
                        "enum": string_valid_options
                    },
                    {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": [x for x in list(filtered_data[k].dropna().unique()) if x != "overall"]
                            if k in filtered_data.columns else ["irrelevant"]
                        }
                    }
                ]
            } for k in fields
        }
    elif dimension in ["time"] and 'start' in filtered_data.columns and 'end' in filtered_data.columns:
        response_structure = {"start": {"type": ["array", "string"],
                                        "anyOf": [{"type": "array"},
                                                  {"type": "string",
                                                   "enum": ["irrelevant"]
                                                   }
                                                  ]
                                        },
                              "end": {"type": ["array", "string"],
                                      "anyOf": [{"type": "array"},
                                                {"type": "string",
                                                 "enum": ["irrelevant"]
                                                 }
                                                ]
                                      }
                              }
    else:
        response_structure = {
            k: {
                "type": ["string", "array"],
                "anyOf": [
                    {
                        "type": "string",
                        "enum": ["irrelevant"]
                    },
                    {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": [x for x in list(filtered_data[k].dropna().unique()) if x != "overall"]
                            if k in filtered_data.columns else ["irrelevant"]
                        }
                    }
                ]
            } for k in fields
        }
    return response_structure


def process_multitask_prediction(multitask_prediction: dict) -> dict:
    # change intention type from str to list
    if "intention" in multitask_prediction:
        multitask_prediction["intention"] = [multitask_prediction["intention"]]

    # business_driver for multitask
    pred_bd = multitask_prediction.get('business_driver_class')
    if pred_bd:
        multitask_prediction["business_driver"] = pred_bd
    return dict(multitask_prediction)


def validate_intention(bert_prediction: dict, spacy_filter: dict,
                       core_dimension_filters: dict, nerDataConfig: NerDataConfig) -> tuple[dict, list]:
    """
    validate intention and data source based on bert prediction and captured core dimensions
    """
    pred_intention = bert_prediction.get("intention", 'none')  # intention prediction is str
    pred_data_source = nerDataConfig.metric_info.get(pred_intention, {}).get("data")
    captured_core_dims = spacy_filter.get("core_dimension", []) + spacy_filter.get("core_dimension_composite", [])
    core_dimension_sources = []
    clarification_list = []
    for core_dim in captured_core_dims:
        sources = [x.get('source', 'bibr') for x in core_dimension_filters.get(core_dim, [])]
        core_dimension_sources.extend(sources)
    if ((len(core_dimension_sources) > 0) and pred_data_source
            and (pred_data_source not in ''.join(core_dimension_sources))):
        most_common = Counter(core_dimension_sources).most_common(1)[0][0]
        if most_common == 'bi':
            default_intention = nerDataConfig.ner_postprocess_indicators.get('default_bi_intention', 'performance')
        elif most_common == 'br':
            default_intention = nerDataConfig.ner_postprocess_indicators.get('default_br_intention', 'source of change')
        elif most_common == 'bibr':
            default_intention = pred_intention
        else:
            default_intention = "none"

        if default_intention != pred_intention:
            clarification_list.append("intention")
        bert_prediction['intention'] = default_intention
    return dict(bert_prediction), clarification_list


class multiTaskSequentialNER:
    def __init__(self, query: str, nerData: NerData, nerDataConfig: NerDataConfig, process_indicator: ProcessIndicator,
                 multitask_pred: Dict[str, str] = None, temperature: float = 0.05):
        self.query = query
        self.nerData = nerData
        self.nerDataConfig = nerDataConfig
        self.process_indicator = process_indicator
        self.generator = None

        self.data = pd.DataFrame()
        self.filtered_data = pd.DataFrame()
        self.temperature = temperature
        self.max_attempts_per_filter = 3
        self.valid_count = {}
        self.parser = JsonOutputParser()
        self.rejection_labels = ['none']
        self.data_levels = ['activity_group', 'measure_group', 'measure']
        self.ignore_fields = []

        if multitask_pred is None:
            self.ner_filter = {}
            self.ner_dict = {}
        else:
            self.ner_dict = multitask_pred
            self.ner_filter = self._assign_ner_filter(self.ner_dict)

    def _assign_ner_filter(self, pred_dict: Dict[str, any]) -> dict:
        assigned_dict = dict()

        for pred_key, pred_val in pred_dict.items():
            answer_dict = {pred_key: pred_val}
            if pred_key == "intention":
                intention_list = list(self.nerDataConfig.metric_info.keys()) + self.rejection_labels
                intention_filter = legacy_filtering._get_intention_answer_dict(dimension="classification",
                                                                               answer_dict=answer_dict,
                                                                               intention_list=intention_list)
                assigned_dict.update(intention_filter)

                if not _check_intention_rejection(intention_filter['intention'], self.rejection_labels,
                                                  list(self.nerDataConfig.metric_info.keys())):
                    metric_info = get_metric_by_intention(intention_filter, self.nerDataConfig)
                    self.data_levels, self.ignore_fields, self.data = (
                        get_data_level_by_intention(self.nerDataConfig, self.nerData, metric_info))
                    assigned_dict.update(metric_info)
                    self.filtered_data = self.data.copy()

            elif pred_key == "period_type":
                # raw pass-through; period_type is resolved together with start/end at the time dimension
                # via _get_time_period_answer_dict
                assigned_dict.update({"period_type": pred_val})

            else:
                if self.process_indicator.legacy_preprocessing_enabled:
                    mapped_dict = legacy_filtering._get_answer_dict(self.filtered_data, 'bert_pred',
                                                                    list(answer_dict.keys()), answer_dict,
                                                                    data='bi')
                else:
                    mapped_dict = general_filtering._get_answer_dict(self.filtered_data, 'ber_pred',
                                                                     list(answer_dict.keys()), answer_dict,
                                                                     data='bi')
                assigned_dict.update(mapped_dict)
        return assigned_dict

    def _run_single_field(self, dimension) -> tuple[dict, dict]:
        dimension_questionnaire = self.nerData.questionnaire.get(dimension, {})
        fields = dimension_questionnaire.get("field", [])

        # generate prompt
        user_prompt = construct_prompt(dimension_questionnaire, dimension, self.query, self.nerData.df_examples,
                                       self.filtered_data, self.nerData.level_type, bot=0)
        basic_prompt = """
    # question
    {question}
    [/INST]
    # answer
    """
        prompt = "<s>[INST]" + user_prompt + "[/INST]</s>\n[INST]" + basic_prompt.format(question=self.query)

        # generate response structure
        response_structure = construct_response_structure(dimension, fields, self.nerDataConfig,
                                                          self.filtered_data, self.nerData.level_type)

        # prompt = construct_prompt_deepseek(dimension_questionnaire, dimension, self.query,
        #                                    self.nerData.df_examples, self.filtered_data)
        # response_structure = {}

        # call llm/model endpoint
        if self.generator:
            answer = self.generator(prompt,
                                    max_new_tokens=2000,
                                    temperature=self.temperature,
                                    top_p=0.9,
                                    top_k=10)
            answer = answer[0]['generated_text']
        else:
            # answer = generate_text(prompt, temperature=self.temperature)
            structured_answer = generate_structured_response(
                prompt=prompt,
                dimension=dimension,
                response_structure=response_structure,
                temperature=self.temperature
            )
            answer = structured_answer.get("generated_text", "")

        if len(answer) < 1:
            return {}, {}

        answer = self.parser.parse(answer)
        logger.debug(f'{dimension} prompt: {prompt}')
        logger.debug(f'llm answer: {answer}')

        # map model res to filter
        if dimension == 'classification':
            # check llm intention response
            dim_filter = legacy_filtering._get_intention_answer_dict(dimension, answer,
                                                                     list(
                                                                         self.nerDataConfig.metric_info.keys()) + self.rejection_labels)
            dim_filter['llm_intention'] = dim_filter['intention']
            answer['llm_intention'] = dim_filter['intention'][0]
            # BERT intention classifier
            # bert_pred = single_request(self.query)
            # dim_filter['intention'] = [bert_pred.intention]
            # dim_filter['intention_proba'] = bert_pred.intention_proba
            # answer['intention'] = bert_pred.intention
            # logger.debug(f"bert answer: {bert_pred.intention}")
        elif dimension == 'time':
            time_results = {"period_type": self.ner_dict.get("period_type", "irrelevant"),
                            "trend": self.ner_dict.get("trend", "no"),
                            "start": answer.get("start", "irrelevant"),
                            "end": answer.get("end", "irrelevant")}
            dim_filter = legacy_filtering._get_time_period_answer_dict(self.filtered_data, time_results,
                                                                       self.nerDataConfig.ner_postprocess_indicators,
                                                                       self.ner_dict.get("intention", []))
            answer.update({'period_type': dim_filter.get('period_type')})
        # elif dimension == 'time' and any('fiscal' in x for x in self.filtered_data['period_type'].unique()):
        #     answer.update({'period_type': self.ner_filter.get('period_type')})
        #     dim_filter = legacy_filtering._get_time_answer_dict_fiscal(self.filtered_data, fields, answer)
        # elif dimension == 'time':
        #     answer.update({'period_type': self.ner_filter.get('period_type')})
        #     dim_filter = legacy_filtering._get_time_answer_dict(self.filtered_data, fields, answer)
        else:
            if self.process_indicator.legacy_preprocessing_enabled:
                dim_filter = legacy_filtering._get_answer_dict(self.data, dimension, fields, answer,
                                                               data=self.ner_filter.get("data", "bi"))
            else:
                dim_filter = general_filtering._get_answer_dict(self.data, dimension, fields, answer,
                                                                data=self.ner_filter.get("data", "bi"))
        logger.debug("valid filter: ", dim_filter, " for dimension answer: ", answer)

        if dimension == 'classification' and (
                not _check_intention_rejection(dim_filter['intention'], self.rejection_labels,
                                               list(self.nerDataConfig.metric_info.keys()))):
            # get metrics based on intention
            metric_info = get_metric_by_intention(dim_filter, self.nerDataConfig)

            dim_filter.update(metric_info)
            answer.update(metric_info)
            # get data levels and ignore fields
            self.data_levels, self.ignore_fields, self.data = (
                get_data_level_by_intention(self.nerDataConfig, self.nerData, metric_info))
            self.filtered_data = self.data.copy()

        self.filtered_data = legacy_filtering._apply_answer_dict(self.filtered_data, dim_filter, res_dict=answer,
                                                                 ignore_fields=self.ignore_fields)
        return dim_filter, answer

    def run(self, generator=None) -> tuple[dict, dict]:
        self.generator = generator
        for dimension, dimension_questionnaire in self.nerData.questionnaire.items():
            start = time.time()
            answer_filter = {}
            answer_dict = {}
            # ignore activity_group dimension
            # if dimension == 'activity_group' and not _check_activity_group(self.ner_dict):
            #     continue
            # run till get valid output for each dimension
            for i in range(1, self.max_attempts_per_filter + 1):
                try:
                    answer_filter, answer_dict = self._run_single_field(dimension)
                    end = time.time()
                    self.valid_count.update({dimension: i, dimension + '_time': end - start})
                    logger.debug(f'Valid answer in {i} trials for dimension {dimension}')
                    break
                except Exception as e:
                    if i == self.max_attempts_per_filter:
                        end = time.time()
                        self.valid_count.update({dimension: i, dimension + '_time': end - start})
                        answer_filter, answer_dict = {}, {}
                        logger.info(f"Maximum number of retries exceeded for dimension {dimension}, "
                                    f"{str(e)}")

                        raise RuntimeError(f"Maximum number of retries exceeded for dimension {dimension}, "f"{str(e)}")
            self.ner_filter.update(answer_filter)
            self.ner_dict.update(answer_dict)

            # break if pred intention is 'none' or not valid
            if dimension == "classification" and _check_intention_rejection(self.ner_filter['intention'],
                                                                            self.rejection_labels,
                                                                            list(
                                                                                self.nerDataConfig.metric_info.keys())):
                break

        return self.ner_filter, self.ner_dict


### SPACY ###

class SpacyNER:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
        pass
        # self.query = query.lower()
        # self.nlp = nlp # assume entity ruler is added

    def lemmatize_words(self, phrase, nlp):
        # words = [token.text for token in nlp.tokenizer(phrase)]
        # new_phrase = [self.lemmatizer.lemmatize(word) for word in words]
        new_phrase = [token.lemma_ for token in nlp(phrase)]

        return ' '.join(new_phrase)

    def stem_words(self, phrase, nlp):
        # stemmer = PorterStemmer()
        # words = nltk.word_tokenize(phrase)
        # words = [token.text for token in nlp.tokenizer(phrase)]
        # new_phrase = [self.stemmer.stem(word) for word in words]
        words = [token.text for token in nlp.tokenizer(phrase)]
        new_phrase = [self.stemmer.stem(word) for word in words]

        return ' '.join(new_phrase)

    def find_entities_and_index(self, query, nlp):
        query_words = nltk.word_tokenize(query)
        len_query_words = len(query_words)

        entities = {}
        doc = nlp(query)
        for ent in doc.ents:
            # find the starting and ending index in the tokenized query list
            remaining = query[:ent.start_char] + query[ent.end_char + 1:]
            remaining = nltk.word_tokenize(remaining)

            ent_words = nltk.word_tokenize(ent.text)
            len_ent_words = len(ent_words)
            subset_index = -1
            for i in range(len_query_words - len_ent_words + 1):
                tmp = query_words[:i] + query_words[i + len_ent_words:]
                if tmp == remaining:
                    subset_index = i
                    break

            if subset_index >= 0:
                key = (subset_index, subset_index + len_ent_words - 1)
                entities[key] = {"text": ent.text,
                                 "label": ent.label_}

        return entities

    def find_longest_segments(self, positions):
        valid_ents = set([v['text'] for k, v in positions.items()])
        # Sort the positions based on the start positions
        sorted_positions = sorted(positions, key=lambda x: (x[0], x[1]))

        longest_segments = set()
        current_start, current_end = sorted_positions[0]

        for start, end in sorted_positions[1:]:
            if start > current_end:
                # Found a new non-overlapping segment
                longest_segments.add((current_start, current_end))
                current_start, current_end = start, end
            else:
                # Check if merging creates a valid predefined entity
                merged_segment = positions.get((current_start, end), {'text': ''})['text']
                if merged_segment in valid_ents:
                    # If it's a valid entity, update the end of the current segment
                    current_end = max(current_end, end)
                else:
                    # If not, save the current segment and start a new one
                    longest_segments.add((current_start, current_end))
                    if start >= current_end:
                        current_start, current_end = start, end

        # Add the last segment
        longest_segments.add((current_start, current_end))

        return list(longest_segments)

    def find_entities_and_index_from_doc(self, query, doc):
        entities = {}
        for ent in doc.ents:
            key = (ent.start, ent.end - 1)
            entities[key] = {"text": ent.text, "label": ent.label_}
        return entities

    def _run(self, query, nlp):
        query = query.lower()
        lem_q = self.lemmatize_words(query, nlp)
        stem_q = self.stem_words(query, nlp)

        docs = list(nlp.pipe([query, lem_q, stem_q]))

        entities = self.find_entities_and_index_from_doc(query, docs[0])
        entities.update(self.find_entities_and_index_from_doc(lem_q, docs[1]))
        entities.update(self.find_entities_and_index_from_doc(stem_q, docs[2]))

        identified_metrics, identified_core_dim, identified_core_dim_composite = set(), set(), set()
        identified_custom_level, identified_custom_tagging, identified_diag_tagging = set(), set(), set()
        identified_halo_tagging = set()
        identified_bd = set()
        if entities:
            merged_index = self.find_longest_segments(entities)
            for start, end in merged_index:
                v = entities[(start, end)]
                label = v['label']
                text = v['text']
                # cleaned_text = clean_text(nlp, text)
                if label == 'METRIC':
                    identified_metrics.add(text)
                elif label == 'CORE_DIMENSION_COMPOSITE':
                    identified_core_dim_composite.add(text)
                elif label == 'CORE_DIMENSION':
                    identified_core_dim.add(text)
                elif label == 'CUSTOM_LEVEL':
                    identified_custom_level.add(text)
                elif label == 'CUSTOM_TAGGING':
                    identified_custom_tagging.add(text)
                elif label == 'DIAG_TAGGING':
                    identified_diag_tagging.add(text)
                elif label == "HALO_TAGGING":
                    identified_halo_tagging.add(text)
                elif label == "BUSINESS_DRIVER":
                    identified_bd.add(text)
                else:
                    pass
        spacy_dict = {'metric': sorted(identified_metrics),
                      'core_dimension': sorted(identified_core_dim),
                      'core_dimension_composite': sorted(identified_core_dim_composite),
                      'custom_level': sorted(identified_custom_level),
                      'custom_tagging': sorted(identified_custom_tagging),
                      'diag_tagging': sorted(identified_diag_tagging),
                      'halo_tagging': sorted(identified_halo_tagging),
                      'business_driver': sorted(identified_bd)}
        return spacy_dict

    def replace_terms(self, query: str, core_dimensions: list[dict[str, any]], nlp) -> str:
        """
        Replace identified terms in the query with a random term from the same label category.
        """
        # Identify entity positions in the query
        entities = self.find_entities_and_index(query, nlp)
        # print(entities)
        entities.update(self.find_entities_and_index(self.lemmatize_words(query), nlp))
        # print(entities)
        entities.update(self.find_entities_and_index(self.stem_words(query), nlp))

        # Build a dictionary of replacement terms
        replacement_dict = {}

        # Create a mapping of label → possible replacements
        category_terms = {entry["label"].lower(): set() for entry in core_dimensions}
        for entry in core_dimensions:
            category_terms[entry["label"].lower()].add(' '.join([x["LOWER"] for x in entry["pattern"]]))

        # Generate replacements for identified terms
        for (start, end), entity in entities.items():
            term, label = entity["text"], entity["label"].lower()
            if label in category_terms:
                possible_replacements = list(category_terms[label] - {term})
                if possible_replacements:
                    replacement_dict[(start, end)] = random.choice(possible_replacements)

        # If no replacements found, return original query
        if not replacement_dict:
            return query

        sorted_positions = sorted(replacement_dict.keys(), reverse=True)

        # Perform replacement
        modified_query = [x for x in query.split(' ')]
        for start, end in sorted_positions:
            replacement = replacement_dict[(start, end)]
            modified_query = modified_query[:start] + [replacement] + modified_query[
                                                                      end:]
        return ' '.join(modified_query)


def _expand_term(term: str, variants: dict) -> str:
    """
    expand longest matched segment in text
    """
    tokens = term.split()
    n = len(tokens)
    i = 0
    result = []

    variant_keys = [(k.split(), k) for k in variants.keys()]

    while i < n:
        best_match = None
        best_len = 0

        # Try all variant keys at position i
        for key_tokens, original_key in variant_keys:
            L = len(key_tokens)
            if i + L <= n and tokens[i:i + L] == key_tokens:
                if L > best_len:
                    best_len = L
                    best_match = original_key

        if best_match:
            # Map using original key → mapped value
            result.append(variants[best_match])
            i += best_len
        else:
            # No match → keep original token
            result.append(tokens[i])
            i += 1

    return " ".join(result)


def normalize_query(query, cleaned_variant_mapping: dict):
    nlp = spacy.load("en_core_web_sm")

    cleaned_query = clean_text(nlp, query)
    normalized_query = _expand_term(cleaned_query, cleaned_variant_mapping)
    return normalized_query


def clean_text(nlp, text, lemma=True) -> str:
    """
    remove possessive in text, lemmatize tokens
    :param text:
    :return: cleaned text
    """
    doc = [token for token in nlp(text) if token.tag_ != "POS"]
    # cleaned_text = " ".join([token.lemma_ for token in doc])
    if lemma:
        cleaned_text = " ".join([stable_lemma(token) for token in doc if not token.is_punct])
    else:
        cleaned_text = " ".join([token.text for token in doc if not token.is_punct])
    return cleaned_text


def stable_lemma(token):
    if token.pos_ in {"NOUN", "PROPN"}:
        return token.lemma_
    else:
        return token.text.lower()


def _clean_variant_mapping(variants_df):
    """
    clean term and variants in variants df to lemmatized terms
    :param nlp:
    :param variants_df:
    :return:
    """
    nlp = spacy.load("en_core_web_sm")
    res = variants_df.copy()
    cleaned_terms = {}
    for term in variants_df["term"].dropna().unique():
        cleaned_terms[term] = clean_text(nlp, term)

    cleaned_variants = {}
    for term in variants_df["variants"].dropna().unique():
        cleaned_variants[term] = clean_text(nlp, term)
    res["cleaned_term"] = res["term"].map(cleaned_terms)
    res["cleaned_variants"] = res["variants"].map(cleaned_variants)
    variant_mapping = res.set_index("cleaned_variants")["cleaned_term"].to_dict()
    return variant_mapping


def _get_mask(df: pd.DataFrame, dict_list: list, ignored_columns: list) -> np.array:
    if not dict_list:
        return np.array([False] * len(df))
    or_masks = [np.array([False] * len(df))]

    if "custom_aggregated" in df.columns:
        normal_mask = df["custom_aggregated"] != "yes"
    else:
        normal_mask = np.array([False] * len(df))

    for dic in dict_list:
        valid_cols = [x for x in dic if x in df.columns and x not in ignored_columns]
        if valid_cols and dic.get("detail") == "ignore when cust_agg":
            mask = (normal_mask & pd.concat([df[col].isin(dic[col]) for col in valid_cols], axis=1).all(axis=1).astype(
                int).values)
            or_masks.append(mask)
        elif valid_cols:
            mask = pd.concat([df[col].isin(dic[col]) for col in valid_cols], axis=1).all(axis=1).astype(int).values
            or_masks.append(mask)
        else:
            pass
    return np.logical_or.reduce(or_masks)


def _get_target_columns(term_filters: list) -> frozenset:
    _FILTER_META_KEYS = {"AKA", "org_term", "source", "halo_to_ignore", "detail"}
    cols = set()
    for f in term_filters:
        cols.update(k for k in f if k not in _FILTER_META_KEYS)
    return frozenset(cols)


def _label_data_with_term(df, mask, term):
    if "core_dimension_term" in df.columns:
        df.loc[mask, "core_dimension_term"] += term + ','
    else:
        df["core_dimension_term"] = ''
        df.loc[mask, "core_dimension_term"] = term + ','
    return df


def categorize_filters(spacy_filter: dict, core_dimension_filters: dict, level_type: dict,
                       product_halo_indicator: bool, ner_filter: dict) -> dict:
    core_dim_res = defaultdict(list)
    core_dim_composite_res = defaultdict(list)
    custom_level_res = defaultdict(list)
    custom_tagging_res = defaultdict(list)
    diag_tagging_res = defaultdict(list)
    bd_res = defaultdict(list)
    terms_from_insights = []
    data_source = ner_filter.get('data', 'br')

    specific_levels = level_type.get('general_level', [])
    specific_taggings = level_type.get('general_tagging', [])

    for captured_type, captured_terms in spacy_filter.items():
        if captured_type == "halo_tagging":
            continue
        for captured_term in captured_terms:
            core_dim_filter = core_dimension_filters.get(captured_term, []).copy()
            # if halo_tagging in ner_filter is empty, ignore filter with halo_to_ignore indicator
            if not product_halo_indicator and data_source == 'bi':
                core_dim_filter = [f for f in core_dim_filter if f.get('halo_to_ignore', '0') == '0']

            for filter in core_dim_filter:
                if data_source not in filter.get("source", ""):
                    continue
                # if data_source == "bi" and "sales_channel" in filter:
                #     continue
                if captured_type == "core_dimension_composite":
                    core_dim_composite_res[captured_term].append(filter)
                elif captured_type == "business_driver" and set(filter.keys()).intersection(["business_driver"]):
                    bd_res[captured_term].append(filter)
                else:
                    if set(filter.keys()).intersection(["split", "split_type"]):
                        diag_tagging_res[captured_term].append(filter)
                    elif set(filter.keys()).intersection(specific_taggings):
                        custom_tagging_res[captured_term].append(filter)
                    elif set(filter.keys()).intersection(specific_levels):
                        custom_level_res[captured_term].append(filter)
                    elif core_dim_filter and captured_type != "metric":
                        core_dim_res[captured_term].append(filter)
            if not len(core_dim_filter):
                terms_from_insights.append(captured_term)

    return {"core_dimension": core_dim_res, "core_dimension_composite": core_dim_composite_res,
            "custom_level": custom_level_res, "custom_tagging": custom_tagging_res,
            "diag_tagging": diag_tagging_res, "terms_from_insights": terms_from_insights,
            "business_driver_captured": bd_res}


def apply_categorized_filters(df: pd.DataFrame, categorized_filter: dict, ignore_columns: list,
                              combine_all: bool = False) -> tuple[np.ndarray, pd.DataFrame]:
    filtered_data = df.copy()
    skip_types = {"terms_from_insights", *ignore_columns}
    union_types = {"core_dimension", "core_dimension_composite", "diag_tagging", "business_driver_captured"}
    grouped_types = {"custom_level", "custom_tagging"}

    def apply_aka_labeling(data, captured_term, term_filters):
        """Label rows matching AKA filters with their original term; returns updated data."""
        aka_filters = [f for f in term_filters if f.get("AKA", "0") == "1"]
        if aka_filters:
            org_term = next((f.get("org_term", captured_term) for f in aka_filters), captured_term)
            aka_mask = _get_mask(data, aka_filters, ignore_columns)
            data = _label_data_with_term(data, aka_mask, org_term)
        return data

    and_masks = []
    for captured_type, captured_term_dict in categorized_filter.items():
        if captured_type in skip_types:
            continue

        elif captured_type in union_types:
            # OR across all terms of this type
            or_masks = []
            for captured_term, term_filters in captured_term_dict.items():
                filtered_data = apply_aka_labeling(filtered_data, captured_term, term_filters)
                or_masks.append(_get_mask(filtered_data, term_filters, ignore_columns))
            if or_masks:
                and_masks.append(np.logical_or.reduce(or_masks))

        elif captured_type in grouped_types:
            # OR terms targeting the same column(s); AND across different column groups
            col_group_masks = defaultdict(list)
            for captured_term, term_filters in captured_term_dict.items():
                filtered_data = apply_aka_labeling(filtered_data, captured_term, term_filters)
                term_mask = _get_mask(filtered_data, term_filters, ignore_columns)
                col_group_masks[_get_target_columns(term_filters)].append(term_mask)
            if col_group_masks:
                group_masks = [np.logical_or.reduce(m) for m in col_group_masks.values()]
                and_masks.append(np.logical_and.reduce(group_masks))

        else:
            logger.info("Captured term type: %s not supported.", captured_type)

    if not and_masks:
        spacy_mask = np.ones(len(filtered_data), dtype=bool)
    elif combine_all:
        spacy_mask = np.logical_or.reduce(and_masks)
    else:
        spacy_mask = np.logical_and.reduce(and_masks)

    return spacy_mask, filtered_data


def combine_filters(data: pd.DataFrame, ner_filter: dict, ner_res: dict,
                    spacy_filter: dict, core_dimension_filters: dict,
                    ignore_fields: list, sub_cols: list, level_type: dict) -> tuple[pd.DataFrame, dict, list]:
    def get_covered_cols(d):
        covered_cols = set()

        for v in d.values():
            if not isinstance(v, dict):
                continue

            for vv in v.values():
                for vvv in vv:
                    for col in vvv:
                        if col in data.columns:
                            covered_cols.add(col)
        return covered_cols

    if spacy_filter.get('core_dimension_composite', []):
        sub_cols += level_type.get('halo_level', []) + level_type.get('halo_tagging', [])
    combined_filter = {}
    core_dimensions = spacy_filter.get('core_dimension', []) + spacy_filter.get('core_dimension_composite', [])
    if (core_dimensions and any([x for x in core_dimensions if x in core_dimension_filters]) and
            len(core_dimension_filters) > 0):
        ignore_fields += sub_cols
    combined_filter.update({k: v for k, v in ner_filter.items()})
    combined_filter['metric'].extend([x for x in spacy_filter.get('metric', []) if x not in ner_filter['metric']])

    halo_tagging = (level_type.get('halo_tagging') or [None])[0]
    product_halo_indicator = isinstance(ner_filter.get(halo_tagging, None), list)
    is_halo = ner_res.get("halo") == "yes"
    # specific_levels = {x for x in level_type.get('general_level', []) if ner_res.get(x) == 'specific'}
    # specific_taggings = {x for x in level_type.get('general_tagging', []) if ner_res.get(x) == 'specific'}

    categorized_filter = categorize_filters(spacy_filter=spacy_filter, core_dimension_filters=core_dimension_filters,
                                            level_type=level_type,
                                            product_halo_indicator=product_halo_indicator,
                                            ner_filter=ner_filter)

    # test run on spacy mask combine all to union
    spacy_mask_combined, filtered_data_combined = apply_categorized_filters(df=data,
                                                                            categorized_filter=categorized_filter,
                                                                            ignore_columns=[],
                                                                            combine_all=True)

    filtered_data_combined = filtered_data_combined[spacy_mask_combined]

    # separate data by custom_aggregation
    if "custom_aggregated" in filtered_data_combined.columns:
        norm_mask = filtered_data_combined["custom_aggregated"] != "diagnostic"
        norm_data = filtered_data_combined[norm_mask]
        diag_mask = filtered_data_combined["custom_aggregated"] == "diagnostic"
        diag_data = filtered_data_combined[diag_mask]
        # diag_mask = np.logical_and.reduce([diag_mask, spacy_mask_combined])

        norm_data_mask, norm_data_filtered = apply_categorized_filters(df=norm_data,
                                                                       categorized_filter=categorized_filter,
                                                                       ignore_columns=["diag_tagging"],
                                                                       combine_all=False)
        norm_data_filtered = norm_data_filtered[norm_data_mask].copy()
        diag_data_filtered = pd.DataFrame()

        if len(diag_data) > 0 and len(categorized_filter.get("diag_tagging", {})):
            diag_data_mask, diag_data_filtered = apply_categorized_filters(df=diag_data,
                                                                           categorized_filter=categorized_filter,
                                                                           ignore_columns=[],
                                                                           combine_all=False)
            diag_data_filtered = diag_data_filtered[diag_data_mask].copy()
    else:
        norm_data_mask, norm_data_filtered = apply_categorized_filters(df=filtered_data_combined,
                                                                       categorized_filter=categorized_filter,
                                                                       ignore_columns=[],
                                                                       combine_all=False)
        norm_data_filtered = norm_data_filtered[norm_data_mask].copy()
        diag_data_filtered = pd.DataFrame()

    if len(diag_data_filtered):
        combined_filter.update(categorized_filter)
        covered_cols = get_covered_cols(categorized_filter)
    else:
        res_filter = {k: v for k, v in categorized_filter.items() if k != "diag_tagging"}
        covered_cols = get_covered_cols(res_filter)
        combined_filter.update(res_filter)
    filtered_data = pd.concat([norm_data_filtered, diag_data_filtered], ignore_index=True).drop_duplicates()
    ignore_fields.extend(covered_cols)
    return filtered_data, dict(combined_filter), ignore_fields

    # # test run on spacy mask
    # spacy_mask_diag, filtered_data_diag = apply_categorized_filters(df=data,
    #                                                                 categorized_filter=categorized_filter,
    #                                                                 ignore_columns=[])
    # # check for rejection due to no data available
    # # if not any([x for x in core_dimensions if x in core_dimension_filters]) and \
    # #         len(core_dimension_filters) > 0 and len(core_dimensions) > 0:
    #     # filtered_data = pd.DataFrame(columns=data.columns)
    #     # if "planner" not in ner_filter.get("intention", []):
    #     #     pass
    #         # raise Exception('No Data Available due to all tactic terms have empty filters.')
    # if np.sum(spacy_mask_diag) == 0:
    #     spacy_mask, filtered_data = apply_categorized_filters(df=data,
    #                                                           categorized_filter=categorized_filter,
    #                                                           ignore_columns=["diag_tagging"])
    #     combined_filter.update({k: v for k, v in categorized_filter.items() if k != "diag_tagging"})
    #     return filtered_data, dict(combined_filter), ignore_fields, spacy_mask
    # else:
    #     combined_filter.update(categorized_filter)
    #     return filtered_data_diag, dict(combined_filter), ignore_fields, spacy_mask_diag


def get_spacy_filter(query: str, spacyConfig: SpacyConfig) -> dict:
    # initiate nlp
    nlp = spacy.load("en_core_web_sm")
    # nlp = English()
    entity_ruler = nlp.add_pipe("entity_ruler", name="entity_ruler", config={"overwrite_ents": True})
    entity_ruler.add_patterns(spacyConfig.core_dimensions)
    spacy_ner_filter = SpacyNER()._run(query, nlp)
    return spacy_ner_filter


def swap_spacy_term_in_query(query: str, spacyConfig: SpacyConfig) -> str:
    # initiate nlp
    nlp = spacy.load("en_core_web_sm")
    # nlp = English()
    entity_ruler = nlp.add_pipe("entity_ruler", name="entity_ruler")
    entity_ruler.add_patterns(spacyConfig.core_dimensions)
    spacy_NER = SpacyNER()
    replaced_query = spacy_NER.replace_terms(query, spacyConfig.core_dimensions, nlp)
    return replaced_query


def extend_acronyms(input_string: str, mapping_dict: dict) -> tuple[str, dict]:
    if not mapping_dict:
        return input_string, {}
    cleaned_string = re.sub(r'[^a-zA-Z0-9\s]', '', input_string)
    pattern = r'(?i)\b(' + '|'.join(re.escape(key) for key in mapping_dict.keys()) + r')(?=\d|\b)'
    result = re.sub(pattern, lambda x: mapping_dict[x.group().lower()], cleaned_string, flags=re.IGNORECASE)

    detected_pair = {k: mapping_dict[k] for k in re.findall(pattern, input_string, flags=re.IGNORECASE)}
    return result, detected_pair


def _check_for_response_modifier(query: str, ner_filter: dict) -> tuple[dict, list]:
    res = dict(ner_filter)
    # check for specific response modifiers
    modifiers = ["diminishing return", "roi curve", "platform", "response curve", "marginal"]
    mods = []
    clarification_list = []
    for modifier in modifiers:
        if modifier in query.lower():
            if modifier in ["diminishing return", "response curve"]:
                res["intention"] = "planner"
                clarification_list.append("intention")
            if modifier in ["marginal"] and res["intention"] != "planner":
                continue
            mods.append(modifier)

    res["response_modifier"] = mods
    return res, clarification_list


def _ner_reconciliation(clientCode: str, modelgroupId: int, ner_filter: dict,
                        ner_res: dict, nerData: NerData, nerDataConfig: NerDataConfig,
                        ignore_fields: list, spacy_mask: np.array) -> tuple[dict, dict, list, list]:
    prioritized_indicator = []

    def _check_valid_level(filter_dict, col, ignore_fields, data):
        valid_level = [x for x in filter_dict.get(col, []) if x != "overall"]
        if valid_level:
            filter_dict[col] = valid_level
        else:
            if col in data and "overall" in data[col].unique():
                filter_dict[col] = ["overall"]
            else:
                ignore_fields.append(col)

    # TODO: general reconciliation logic
    halo_level = (nerData.level_type.get('halo_level') or [None])[0]
    halo_tagging = (nerData.level_type.get('halo_tagging') or [None])[0]
    general_level = (nerData.level_type.get('general_level') or [None])  # list
    data = nerData.df_bi.copy() if ner_filter["data"] == "bi" else nerData.df_br.copy()
    if halo_level and halo_tagging and clientCode not in ["LINKEDIN"]:
        ner_filter, ner_dict = legacy_filtering._halo_level_to_tagging(halo_level, halo_tagging, data,
                                                                       ner_filter=ner_filter, ner_dict=ner_res)
    # tmp_data = tmp_data[spacy_mask].reset_index(drop=True).copy()
    tmp_data = legacy_filtering._apply_answer_dict(df=data, answer_dict=ner_filter,
                                                   res_dict=ner_res, ignore_fields=ignore_fields)
    if clientCode == "FTR" and all(tmp_data["business_driver"].dropna().unique() == "other") and ner_filter[
        "intention"] == ["contribution"]:
        ner_filter["intention"] = ["source of change"]
        ner_filter["main_metric"] = nerDataConfig.metric_info["source of change"].get("mainMetric", [])
        ner_filter["metric"] = nerDataConfig.metric_info["source of change"].get("metric", [])
        ner_res["intention"] = ["source of change"]
        ner_res["main_metric"] = ner_filter["main_metric"]
        ner_res["metric"] = ner_filter["metric"]
    if len(ner_filter['core_dimension']) + len(ner_filter['core_dimension_composite']) and ner_filter['intention'] == [
        'sales']:
        ner_filter["intention"] = ["contribution"]
        ner_filter["main_metric"] = nerDataConfig.metric_info["contribution"].get("mainMetric", [])
        ner_filter["metric"] = nerDataConfig.metric_info["contribution"].get("metric", [])
        ner_res["intention"] = ["contribution"]
        ner_res["main_metric"] = ner_filter["main_metric"]
        ner_res["metric"] = ner_filter["metric"]
    # filter priority rank
    if clientCode == "LINKEDIN" and "portfolio" in tmp_data.columns:
        if len(nerData.level_type.get("halo_level", [])) > 0:
            is_specific_bu = (isinstance(ner_filter.get(halo_level), list) and
                              ner_res.get(halo_level) not in ["irrelevant", "all"])  # specific halo
            is_specific_country = ner_res.get("country") == "specific"
            if (not is_specific_bu and is_specific_country) or ("portfolio" in ner_filter.get(halo_level, [])):
                ner_filter["portfolio"] = ["yes"]
                ignore_fields.append(halo_level)
            else:
                ner_filter["portfolio"] = ["no"]
                _check_valid_level(filter_dict=ner_filter, col=halo_level, ignore_fields=ignore_fields, data=data)
                _check_valid_level(filter_dict=ner_filter, col="kpi", ignore_fields=ignore_fields, data=data)
                _check_valid_level(filter_dict=ner_filter, col="detailed_kpi", ignore_fields=ignore_fields, data=data)
            prioritized_indicator.append("portfolio")
    # if clientCode == "LINKEDIN" and "planner" in ner_filter.get("intention", []):
    #     if halo_level and halo_tagging:
    #         ner_filter[halo_level] = ner_filter[halo_tagging]
    #         del ner_filter[halo_tagging]
    if clientCode == "LINKEDIN":
        na_terms = ner_filter.get("terms_from_insights", [])
        if "executive summary" in na_terms:
            # na_terms.remove("executive summary")
            na_terms += ["recommendation"]
    return dict(ner_filter), dict(ner_res), ignore_fields, prioritized_indicator


def _validate_postprocess_indicators(indicators: dict, bidata: pd.DataFrame, brdata: pd.DataFrame) -> dict:
    period_priority = indicators.get("period_priority", [])
    all_period_types = set(bidata['period_type'].unique()) | set(brdata['period_type'].unique())
    default_report_length = indicators.get("non_trend_report_length", [2, 2, 2, 12])
    default_report_length_trend = indicators.get("trend_report_length", [2, 2, 4, 12])

    if any('fiscal' in t for t in all_period_types):
        calendar_type = "fiscal"
    else:
        calendar_type = "custom"

    if calendar_type == "fiscal":
        periods = ["fiscal year", "fiscal half", "fiscal quarter", "fiscal month"]
    else:
        periods = ["year", "half", "quarter", "month"]

    report_length = {k: int(v) for k, v in zip(periods, default_report_length)}
    trend_report_length = {k: int(v) for k, v in zip(periods, default_report_length_trend)}

    # filtered = [x for x in period_priority if x in all_period_types]
    # # missing = [x for x in all_period_types if x not in period_priority]
    # if len(filtered) > 0:
    #     period_priority = filtered
    # else:
    #     period_priority = [x for x in periods if x in all_period_types]

    granularity = indicators.get("performance_period_granularity")
    if granularity not in all_period_types:
        granularity = [x for x in periods[::-1] if x in all_period_types][0]

    indicators["period_priority"] = periods
    indicators["calendar_type"] = calendar_type
    indicators["performance_period_granularity"] = granularity
    indicators["report_yoy"] = bool(indicators.get("report_yoy", 0))
    indicators["non_trend_report_length"] = report_length
    indicators["trend_report_length"] = trend_report_length
    return dict(indicators)


def find_latest_time(data: pd.DataFrame, ner_filter: dict, multitask_pred: dict,
                     ner_postprocess_indicator: dict) -> tuple[int, str]:
    p_processed = ner_filter.get("period_type")
    df = data[data["custom_aggregated"] == "no"] if "custom_aggregated" in data.columns else data

    # load indicators
    calendar_type = ner_postprocess_indicator.get('calendar_type', 'fiscal')
    period_priority = ner_postprocess_indicator.get('period_priority')

    valid_periods = [x for x in period_priority if x in df["period_type"].unique()]
    p_processed_index = min(valid_periods.index(p_processed[0]) + 1, len(valid_periods) - 1)
    max_p = valid_periods[p_processed_index]
    max_end_time = df[df["period_type"] == max_p]["end"].max()

    return max_end_time, max_p


def generate_ner_filter(clientCode: str, modelgroupId: int, query: str,
                        process_indicator: ProcessIndicator) -> ReadoutData:
    # ner data config
    nerData = NerData.from_local(clientCode, modelgroupId)
    nerDataConfig = NerDataConfig.from_local(clientCode, modelgroupId)
    spacyConfig = SpacyConfig.from_local(clientCode, modelgroupId)

    # validate ner postprocess indicators
    ner_postprocess_indicators = _validate_postprocess_indicators(indicators=nerDataConfig.ner_postprocess_indicators,
                                                                  bidata=nerData.df_bi, brdata=nerData.df_br)
    nerDataConfig.ner_postprocess_indicators = ner_postprocess_indicators

    # extend acronyms and special terms
    query = query.lower()
    extended_query, detected_pair = extend_acronyms(query, nerDataConfig.acronym_dict)

    # TODO: check if embeddings are in dataframe already
    nerData = calculate_embedding_similarity_scores(query, nerData)

    # multi-task bert prediction
    multitask_pred = single_request(extended_query)

    rejection_message = ""

    # spacy
    spacy_filter = dict()
    if process_indicator.spacy_enabled:
        cleaned_variant_mapping = _clean_variant_mapping(spacyConfig.variant_mapping)
        normalized_query = normalize_query(query, cleaned_variant_mapping)
        spacy_filter = get_spacy_filter(normalized_query.replace("'", ""), spacyConfig)
        logger.debug(f'spacy filter: {spacy_filter}')

    # validate intention with spacy filter
    multitask_pred = validate_intention(bert_prediction=multitask_pred, spacy_filter=spacy_filter,
                                        core_dimension_filters=spacyConfig.core_filters,
                                        nerDataConfig=nerDataConfig)

    # validate intention with response modifier
    multitask_pred = _check_for_response_modifier(query, multitask_pred)

    # prepare multitask prediction for llm
    multitask_pred = process_multitask_prediction(multitask_pred)

    if not _check_intention_rejection(multitask_pred["intention"], ['none'], list(nerDataConfig.metric_info.keys())):
        # fields that are not covered in multitask_res
        all_fields = [item for k, v in nerData.questionnaire.items() for item in v['field']]
        uncovered_fields = [item for item in all_fields if item not in multitask_pred]
        questionnaire_dimensions = list(nerData.questionnaire.keys())
        for dim in questionnaire_dimensions:
            dim_questionnaire = nerData.questionnaire[dim]
            dim_uncovered_fields = [x for x in dim_questionnaire['field'] if x in uncovered_fields]
            if len(dim_uncovered_fields):
                nerData.questionnaire[dim]['field'] = dim_uncovered_fields
            else:
                nerData.questionnaire.pop(dim)

        # generate ner from llm / bert
        ner_llm = multiTaskSequentialNER(extended_query, nerData, nerDataConfig, process_indicator, multitask_pred)
        ner_filter, ner_res = ner_llm.run()
        ner_filter.update({'acronym': detected_pair})

        # postprocessing of ner filters
        # keyword matching

        data = nerData.df_bi.copy() if ner_filter.get('data', 'bi') == 'bi' else nerData.df_br.copy()

        # find latest time in data
        latest_time, latest_period_type = find_latest_time(data, ner_filter, multitask_pred,
                                                           nerDataConfig.ner_postprocess_indicators)

        ner_filter, ner_res = legacy_filtering._postprocess_filter_given_indicator(ner_filter, ner_res, data,
                                                                                   nerDataConfig.ner_postprocess_indicators,
                                                                                   nerData.level_type)
        if process_indicator.legacy_preprocessing_enabled:
            df, ner_filter, ignore_fields = legacy_filtering.map_keyword_in_query(data, query, ner_filter,
                                                                                  ner_llm.ignore_fields)
        else:
            df, ner_filter, ignore_fields = general_filtering.map_keyword_in_query(modelgroupId, data, query,
                                                                                   ner_filter,
                                                                                   ner_llm.ignore_fields)

        # spacy
        spacy_mask = np.array([True] * len(df))
        df = df[spacy_mask].reset_index(drop=True).copy()
        if process_indicator.spacy_enabled and len(spacyConfig.core_dimensions):
            spacy_override_columns = nerDataConfig.data_levels.get("spacyOverrideColumns", [])
            df, ner_filter, ignore_fields = combine_filters(df, ner_filter, ner_res, spacy_filter,
                                                            spacyConfig.core_filters, ignore_fields,
                                                            spacy_override_columns, nerData.level_type)
            logger.debug(f'spacy filter: {spacy_filter}')

        # specific fields in ner res
        ner_filter, ner_res, ignore_fields = legacy_filtering._get_specific_values(df, ner_filter, ner_res,
                                                                                   nerData.level_type, ignore_fields)

        # reconciliation
        ner_filter, ner_res, ignore_fields, prioritized_indicator = _ner_reconciliation(clientCode, modelgroupId,
                                                                                        ner_filter, ner_res,
                                                                                        nerData, nerDataConfig,
                                                                                        ignore_fields, spacy_mask)

        # apply filters
        ag_df, mg_df, m_df = [pd.DataFrame(columns=df.columns)] * 3
        # df = df[spacy_mask].reset_index(drop=True).copy()
        if not len(df):
            rejection_message = PromptTemplates().no_tactic_rejection_message.format(tactic=', '.join(
                spacy_filter.get("core_dimension", []) + spacy_filter.get("core_dimension_composite", [])))
        else:
            df = legacy_filtering._apply_answer_dict(df, ner_filter, res_dict=ner_res, ignore_fields=ignore_fields,
                                                     prioritized_indicator=prioritized_indicator)
            # client specific rules, to be generalized
            df = legacy_filtering._apply_filtering_logic(clientCode, ner_filter, data=df)
            if any(x for x in ner_filter.get("main_metric", []) if x in df["metric"].unique()):
                is_halo_tagging = any(ner_filter.get(x) for x in nerData.level_type.get("halo_tagging", []))
                data_levels = legacy_filtering.get_report_levels(df=df,
                                                                 level=ner_llm.data_levels,
                                                                 ner_filter=ner_filter,
                                                                 is_halo_tagging=is_halo_tagging)
                ag_df, mg_df, m_df = legacy_filtering._get_level_answer(df, level=data_levels)
            else:
                rejection_message = PromptTemplates().no_metric_rejection_message
        logger.debug(f'Generating filters on client {clientCode}, modelgroup {modelgroupId}, ner filter {ner_filter}')

        nerFilters = NerFilters(query=query, ner_filter=ner_filter, ner_res=ner_res)
        nerFilters.validate()

        logger.debug(f'ner filter: {ner_filter}')

    else:
        ner_filter = multitask_pred
        ner_res = multitask_pred
        latest_time, latest_period_type = 0, "irrelevant"
        logger.debug(f'ner filter: {multitask_pred}')
        df = pd.DataFrame(columns=nerData.df_bi.columns)
        ag_df, mg_df, m_df = [pd.DataFrame(columns=nerData.df_bi.columns)] * 3
        rejection_message = PromptTemplates().out_of_scope_rejection_message

    if ("brand_focused_marketing" in df.columns) and ("custom_aggregated" in df.columns) and (
    not nerData.level_type.get("halo_tagging")) and ("yes" in df["custom_aggregated"]):
        nerData.level_type["halo_tagging"] = ["brand_focused_marketing"]
    ner_filter.update({'level': nerData.level_type})
    ner_filter.update({"rejection_message": rejection_message})
    ner_filter.update({'time': df["time"].unique().tolist()})
    ner_filter.update({'period_type': df["period_type"].unique().tolist()})
    ner_filter.update({"latest_time": int(latest_time), "latest_period_type": latest_period_type})
    ner_filter.update({"ag_df_shape": ag_df.shape, "mg_df_shape": mg_df.shape, "m_df_shape": m_df.shape})
    readoutData = ReadoutData(ner_filters=ner_filter, ner_results=ner_res, df_activity_group=ag_df,
                              df_measure_group=mg_df, df_measure=m_df)
    return readoutData