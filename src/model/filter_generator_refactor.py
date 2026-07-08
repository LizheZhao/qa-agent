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
from typing import Any, Dict, Literal, Union
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
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_anthropic import ChatAnthropic
from openai import OpenAI
from pydantic import BaseModel, create_model, field_validator
from src.data.data_interface import NerData, NerDataConfig, NerFilters, SpacyConfig, ReadoutData, ProcessIndicator
import src.model.data_filtering as legacy_filtering
import src.model.data_filtering_other_category as general_filtering
import src.integrations.llm as ds_llm
from src.integrations.llm_external import generate_filter_response
from src.integrations.embedding import query_embeddings, get_similarity_score, get_cached_embeddings
from src.integrations.classifier import single_request
from src.utils import load_yaml_file
from src.model.common import PromptTemplates
from src.model.filter_generator import (extend_acronyms, construct_prompt, calculate_embedding_similarity_scores,
                                        _clean_variant_mapping, normalize_query, get_spacy_filter,
                                        validate_intention, _check_for_response_modifier, _check_intention_rejection,
                                        get_metric_by_intention, get_data_level_by_intention,
                                        _validate_postprocess_indicators, find_latest_time, combine_filters,
                                        _ner_reconciliation, construct_response_structure)

logger = logging.getLogger(__name__)


class SimpleFilterExtractor:
    def __init__(self, query, nerData: NerData, nerDataConfig: NerDataConfig, enterprise_ai_ner,
                 temperature: float = 0.05):
        self.query = query
        self.nerData = nerData
        self.nerDataConfig = nerDataConfig
        self.temperature = temperature
        self.max_attempts_per_filter = 3
        self.valid_count = {}
        self.parser = JsonOutputParser()
        self.ner_filters = {}
        self.ner_results = {}
        self.filtered_data = nerData.df_bi.copy()
        self.prompt_config = load_yaml_file("./config/prompt.yaml")
        self.clarification_list = []
        self.enterprise_ai_ner = enterprise_ai_ner
        self.clarification_reasons = {}

    def _run_single_field(self, dimension) -> tuple[dict, dict]:
        dimension_questionnaire = self.nerData.questionnaire.get(dimension, {})
        fields = dimension_questionnaire.get("field", [])
        need_clarification = False
        if dimension == "time":
            fields = ["start", "end"]

        # available selections


        # generate prompt
        system_prompt = self.prompt_config["FILTER_EXTRACTION_SYSTEM_PROMPT"]
        user_prompt = construct_prompt(dimension_questionnaire,
                                       dimension,
                                       self.query,
                                       self.nerData.df_examples,
                                       self.filtered_data,
                                       self.nerData.level_type,
                                       self.enterprise_ai_ner,
                                       top=10,
                                       bot=0,
                                       n_total=15)
        basic_prompt = """
        # question
        {question} 
        [/INST]
        # answer
        """
        prompt = "<s>[INST]" + user_prompt + "[/INST]</s>\n[INST]" + basic_prompt.format(question=self.query)

        if not self.enterprise_ai_ner:
            logger.info("Using external llm for filter extraction.")
            # call llm/model endpoint
            response_structure = construct_response_structure(dimension, fields, self.nerDataConfig,
                                                              self.filtered_data, self.nerData.level_type)
            structured_answer = ds_llm.generate_structured_response(
                prompt=prompt,
                dimension=dimension,
                response_structure=response_structure,
                temperature=self.temperature
            )
            answer = structured_answer.get("generated_text", "")
            answer = self.parser.parse(answer)
            need_clarification = False
            clarification_reason = ""

        else:
            logger.info("Using internal llm for filter extraction.")
            # call llm/model endpoint - external vendor
            response = generate_filter_response(input_question=self.query,
                                                input_prompt=prompt,
                                                field_type=dimension,
                                                additional_instruction=system_prompt)

            parsed_answer = self.parser.parse(response)
            answer = {k:v for k, v in parsed_answer["answer"].items() if k in fields}
            need_clarification = parsed_answer["need_clarification"]
            clarification_reason = parsed_answer["clarification_reason"]

        # if not all(f in answer.keys() for f in fields) or need_clarification == True:
        #     if dimension == "time":
        #         self.clarification_list.extend(["start", "end", "period_type"])
        #     else:
        #         self.clarification_list.extend(fields)
        # need_clarification = llm_response.need_clarification
        # answer = llm_response.model_dump(exclude={"need_clarification"})

        if bool(need_clarification) or any(x not in answer for x in fields):
            if dimension == "time":
                clarification_fields = ["start", "end", "period_type"]
            else:
                clarification_fields = fields
            self.clarification_list.extend(fields)
            self.clarification_reasons[dimension] = clarification_reason

        logger.debug(f'{dimension} prompt: {prompt}')
        logger.info(f'llm answer: {answer}, need clarification: {need_clarification}, because: {clarification_reason}')

        return answer

    def run(self) -> dict:
        for dimension, dimension_questionnaire in self.nerData.questionnaire.items():
            start = time.time()
            answer_dict = {}
            for i in range(1, self.max_attempts_per_filter + 1):
                try:
                    answer_dict = self._run_single_field(dimension)
                    end = time.time()
                    self.valid_count.update({dimension: i, dimension + '_time': end - start})
                    logger.debug(f'Valid answer in {i} trials for dimension {dimension}')
                    break
                except Exception as e:
                    if i == self.max_attempts_per_filter:
                        end = time.time()
                        self.valid_count.update({dimension: i, dimension + '_time': end - start})
                        logger.info(f"Maximum number of retries exceeded for dimension {dimension}, "
                                    f"{str(e)}, adding to clarification list")

                        # raise RuntimeError(f"Maximum number of retries exceeded for dimension {dimension}, "f"{str(e)}")
                        dimension_questionnaire = self.nerData.questionnaire.get(dimension, {})
                        fields = dimension_questionnaire.get("field", [])
                        if dimension == "time":
                            self.clarification_list.extend(["start", "end", "period_type"])
                        else:
                            self.clarification_list.extend(fields)
                        raise Exception(e)
            self.ner_results.update(answer_dict)

        return self.ner_results


def get_available_values(clarification_list: list, nerData: NerData, nerDataConfig: NerDataConfig,
                         data: pd.DataFrame = None) -> dict:
    if data is None:
        data = nerData.df_bi
    data_cols = data.columns
    level_type = nerData.level_type
    hierarchy_levels = level_type.get("halo_level", []) + level_type.get("general_level", [])
    res = {}
    #
    default_vals = {"trend": ["yes", "no"],
                    # "halo": ["yes", "no"],
                    # "rank": ["top", "bottom", "na"]
                    }

    additional_options = {"period_type": []}
    additional_options.update({lvl: ["all"] for lvl in hierarchy_levels})

    for field in clarification_list:
        if field == "intention":
            vals = list(nerDataConfig.metric_info.keys()) + ["out-of-scope"]
        elif field in data_cols:
            vals = list(data[field].dropna().unique())
        else:
            vals = default_vals.get(field, [])

        if vals:
            vals = sorted(vals) + additional_options.get(field, ["all", "irrelevant"])
            res[field] = vals

    return res


def process_bert_predictions(probabilities: dict, ner_data: NerData=None) -> tuple[dict, dict]:
    pred_bd = probabilities.get('business_driver_class')
    if pred_bd:
        probabilities["business_driver"] = pred_bd
        del probabilities["business_driver_class"]

    if ner_data is not None:
        pred_period = probabilities.get('period_type')
        is_fiscal_calendar = "fiscal" in str(ner_data.df_bi["time"].iloc[0])
        if pred_period and not is_fiscal_calendar:
            probabilities["period_type"] = {
                ("half year" if (stripped := k.replace("fiscal ", "")) == "half" else stripped): v
                for k, v in pred_period.items()
            }
    predictions = {field: max(vals, key=vals.get) for field, vals in probabilities.items()}
    return predictions, probabilities


def _is_irrelevant(v) -> bool:
    if isinstance(v, list):
        return (not v) or ("irrelevant" in v)
    return v == "irrelevant"


def _time_intersects(start, end, df_bi, period_type) -> bool:
    """True if requested [start,end] overlaps df_bi rows of the given period_type.
    Overlap test mirrors data_filtering (data_start <= req_end AND data_end >= req_start)."""
    df = df_bi[df_bi["period_type"] == period_type]
    if df.empty:
        return True  # cannot disprove coverage -> don't trigger on (b)
    data_min = int(df["start"].astype(int).min())
    data_max = int(df["end"].astype(int).max())
    starts = [int(x) for x in (start if isinstance(start, list) else [start])]
    ends = [int(x) for x in (end if isinstance(end, list) else [end])]
    return not (min(starts) > data_max or max(ends) < data_min)


def _should_clarify_time(ner_results, predictions, probabilities, df_bi,
                         llm_priority, threshold=0.95) -> bool:
    start = ner_results.get("start")
    end = ner_results.get("end")

    # (a) missing start or end
    if (_is_irrelevant(start) or _is_irrelevant(end)) and predictions.get("trend") == "yes":
        return True

    # (c) period_type BERT confidence (under llm_priority the LLM handles period_type clarification)
    if not llm_priority:
        pt_scores = probabilities.get("period_type")
        if pt_scores and max(pt_scores.values()) <= threshold:
            return True

    # (b) no intersection under current period type choice
    period_type = predictions.get("period_type")
    if period_type is not None:
        try:
            if not _time_intersects(start, end, df_bi, period_type):
                return True
        except (ValueError, TypeError):
            return True  # unparsable start/end -> safest to clarify

    return False


def _should_clarify_halo(ner_results, probabilities, spacy_filter, halo_level, halo_tagging) -> bool:
    halo_ner = spacy_filter["halo_tagging"]
    ishalo_llm = False
    if halo_level and halo_tagging:
        # spacy found a halo tag not reflected in the llm ner -> clarify
        res1 = ner_results.get(halo_level, [])
        res2 = ner_results.get(halo_tagging, [])
        if isinstance(res1, str):
            res1 = [res1]
        if isinstance(res2, str):
            res2 = [res2]
        ner_res = res1 + res2
        if len(halo_ner) and not set(halo_ner).intersection(set(ner_res)):
            return True
        ishalo_llm = not (("irrelevant" in res1) or ("irrelevant" in res2))

    # bert and llm disagree on halo presence -> clarify
    ishalo_bert = probabilities.get("halo", {}).get("yes", 0) >= 0.5
    if ishalo_bert != ishalo_llm:
        return True

    return False


def update_clarification_list(clarification_list, ner_results, probabilities, spacy_filter, level_type,
                              df_bi, predictions, llm_priority, threshold=0.95):
    """
    logics to add field to clarification list
    """
    # halo: clarify when spacy halo tags aren't reflected in the llm ner, or bert/llm disagree on halo
    halo_level = level_type["halo_level"][0] if "halo_level" in level_type else None
    halo_tagging = level_type["halo_tagging"][0] if "halo_tagging" in level_type else None
    if _should_clarify_halo(ner_results, probabilities, spacy_filter, halo_level, halo_tagging):
        clarification_list.extend([halo_level, halo_tagging])

    # prediction confidence < threshold
    # for dim, dim_scores in probabilities.items():
    #     if max(dim_scores.values()) <= threshold:
    #         clarification_list.extend([dim])

    # llm ner is specific but spacy is not
    # level_type_reverse = {v: k for k, val in level_type.items() for v in val}
    # spacy_key_map = {"general_level": "custom_level",
    #                  "general_tagging": "custom_tagging",
    #                  "business_driver": "business_driver",
    #                  "diag_tagging": "diag_tagging",
    #                  }
    # for k, v in ner_results.items():
    #     if isinstance(v, str) and v == "specific":
    #         # clarification_list.extend([k])
    #         if k in level_type_reverse:
    #             type_k = spacy_key_map[level_type_reverse[k]]
    #             spacy_res = spacy_filter[type_k]
    #             if not spacy_res:
    #                 clarification_list.extend([k])

    # time: expose start/end/period_type together on missing time, out-of-range, or low period_type confidence
    # if _should_clarify_time(ner_results=ner_results, predictions=predictions, probabilities=probabilities,
    #                         df_bi=df_bi, llm_priority=llm_priority, threshold=threshold):
    #     clarification_list.extend(["start", "end", "period_type"])

    return list(set(clarification_list))


def order_clarification_list(l: list, level_type: dict) -> list:
    order = ["intention"]
    time_cols = ["period_type", "start", "end"]
    type_order = ["general_level", "general_tagging", "halo_level", "halo_tagging", "diag_tagging"]

    order += time_cols
    for t in type_order:
        order += level_type.get(t, [])

    ordered_list = [x for x in order if x in l]
    ordered_list += [x for x in l if x not in order]
    return ordered_list


def extract_ner_filter(clientCode: str, modelgroupId: int, query: str,
                       llm_priority: bool = False, enterprise_ai_ner: bool = False,
                       coarse_intent: str = None):

    # ner data config
    nerData = NerData.from_local(clientCode, modelgroupId)
    nerDataConfig = NerDataConfig.from_local(clientCode, modelgroupId)
    spacyConfig = SpacyConfig.from_local(clientCode, modelgroupId)

    # extend acronyms and special terms
    query = query.lower()
    extended_query, detected_pair = extend_acronyms(query, nerDataConfig.acronym_dict)

    # TODO: check if embeddings are in dataframe already
    nerData = calculate_embedding_similarity_scores(query, nerData)

    # multi-task bert prediction
    multitask_pred = single_request(extended_query)
    probabilities = multitask_pred["probabilities"]
    predictions, probabilities = process_bert_predictions(probabilities=probabilities,
                                                          ner_data=nerData)

    # spacy
    spacy_filter = dict()
    cleaned_variant_mapping = _clean_variant_mapping(spacyConfig.variant_mapping)
    normalized_query = normalize_query(query, cleaned_variant_mapping)
    spacy_filter = get_spacy_filter(normalized_query.replace("'", ""), spacyConfig)

    # validate intention with spacy filter
    multitask_pred, l1 = validate_intention(bert_prediction=predictions, spacy_filter=spacy_filter,
                                            core_dimension_filters=spacyConfig.core_filters,
                                            nerDataConfig=nerDataConfig)

    # validate intention with response modifier
    multitask_pred, l2 = _check_for_response_modifier(query, multitask_pred)
    predictions.update(multitask_pred)

    # llm filters
    # When llm_priority is on: send every questionnaire field to the LLM (LLM overrides BERT downstream).
    # Otherwise: only fields uncovered by BERT, with low BERT confidence, or when intention == "none".
    all_fields = [item for k, v in nerData.questionnaire.items() for item in v['field']]
    if llm_priority:
        llm_fields = set(all_fields)
    else:
        threshold = 0.95
        uncovered_fields = [item for item in all_fields if item not in predictions]
        low_score_fields = [item for item in all_fields if item in probabilities and
                            max(probabilities.get(item).values()) <= threshold]
        llm_fields = uncovered_fields + low_score_fields
        llm_fields += ["intention"] if predictions["intention"] == "none" else []
        llm_fields = set(llm_fields)

    questionnaire_dimensions = list(nerData.questionnaire.keys())
    for dim in questionnaire_dimensions:
        dim_questionnaire = nerData.questionnaire[dim]
        dim_uncovered_fields = [x for x in dim_questionnaire['field'] if x in llm_fields]
        if len(dim_uncovered_fields):
            nerData.questionnaire[dim]['field'] = dim_uncovered_fields
        else:
            nerData.questionnaire.pop(dim)

    # generate ner from llm / bert
    ner_llm = SimpleFilterExtractor(extended_query, nerData, nerDataConfig, enterprise_ai_ner)
    ner_results = ner_llm.run()
    clarification_list = ner_llm.clarification_list
    clarification_reasons = ner_llm.clarification_reasons

    # update clarification list
    # When llm_priority is on, drop questionnaire fields from probabilities so the BERT-confidence
    # loop inside update_clarification_list doesn't re-ask for fields the LLM has already answered.
    # Non-questionnaire dims (e.g. "halo") stay for the halo BERT-vs-LLM check.
    if llm_priority:
        probabilities_for_clarification = {k: v for k, v in probabilities.items() if k not in llm_fields}
    else:
        probabilities_for_clarification = probabilities

    clarification_list = update_clarification_list(clarification_list=clarification_list,
                                                   ner_results=ner_results,
                                                   probabilities=probabilities_for_clarification,
                                                   spacy_filter=spacy_filter,
                                                   level_type=nerData.level_type,
                                                   df_bi=nerData.df_bi,
                                                   predictions=predictions,
                                                   llm_priority=llm_priority)

    res = dict(predictions)
    res.update(ner_results)
    res.update(detected_pair)

    # order clarification list
    clarification_list = order_clarification_list(l=clarification_list,
                                                  level_type=nerData.level_type)
    # available values for selection when clarification
    available_values = get_available_values(clarification_list, nerData, nerDataConfig)

    # planner/BERT intent reconciliation: on disagreement surface intention for clarification (offer both)
    if coarse_intent is not None:
        resolved_intent = res.get("intention")
        if isinstance(resolved_intent, list):
            resolved_intent = resolved_intent[0] if resolved_intent else None
        if resolved_intent and coarse_intent != resolved_intent and "intention" not in available_values:
            available_values["intention"] = [coarse_intent, resolved_intent]

    return probabilities, spacy_filter, res, available_values


def map_filter_to_value(ner_results: dict,
                        nerData: NerData,
                        nerDataConfig: NerDataConfig,
                        process_indicator: ProcessIndicator) -> tuple[dict, list, list, pd.DataFrame]:
    ner_filter = dict()
    mapped_col = set()
    time_cols = ["period_type", "start", "end"]
    data_levels, ignore_fields, data = [], [], pd.DataFrame()

    # map intention
    intention = ner_results["intention"]
    # TODO: use _get_intention_answer_dict
    if isinstance(intention, str):
        intention = [intention]
    if _check_intention_rejection(intention,
                                  rejection_labels=["none", "out-of-scope"],
                                  available_labels=list(nerDataConfig.metric_info.keys())):
        raise ValueError("Intention is out-of-scope, question rejected")
    else:
        # get metrics based on intention
        intention_result = {"intention": intention}
        metric_info = get_metric_by_intention(intention_result, nerDataConfig)
        ner_filter.update(intention_result)
        ner_filter.update(metric_info)
        # get data levels and ignore fields
        data_levels, ignore_fields, data = get_data_level_by_intention(nerDataConfig, nerData, metric_info)
    mapped_col.update(["intention"])

    for k, v in ner_results.items():
        if k in mapped_col:
            continue
        if k in time_cols:
            time_dict = legacy_filtering._get_time_period_answer_dict(data,
                                                                      ner_results,
                                                                      nerDataConfig.ner_postprocess_indicators,
                                                                      intention)
            ner_filter.update(time_dict)
            mapped_col.update(time_dict)
        else:
            if process_indicator.legacy_preprocessing_enabled:
                dim_filter = legacy_filtering._get_answer_dict(data, k, fields=[k], answer_dict={k: v})
            else:
                dim_filter = general_filtering._get_answer_dict(data, k, fields=[k], answer_dict={k: v})
            ner_filter.update(dim_filter)
            mapped_col.update([k])

    return dict(ner_filter), data_levels, ignore_fields, data


def log_to_message(log):
    """Flatten the _apply_answer_dict log into a single string."""
    parts = []

    for key, vals in log["skipped_no_value"].items():
        parts.append(f"No data matches {key} = {vals} after applying keyword filtering; this filter was skipped.")

    for c in log["conflicts"]:
        if c["conflicts_with"]:
            with_str = " + ".join(c["conflicts_with"])
            parts.append(f"{c['filter']} = {c['values']} has no data in combination with {with_str}.")
        else:
            parts.append(f"{c['filter']} = {c['values']} produced no data.")

    return "\n\n".join(parts)


def generate_readoutdata(clientCode: str, modelgroupId: int, query: str,
                         ner_results: dict, spacy_filter: dict,
                         process_indicator: ProcessIndicator) -> ReadoutData:
    rejection_message = ""

    # ner data config
    nerData = NerData.from_local(clientCode, modelgroupId)
    nerDataConfig = NerDataConfig.from_local(clientCode, modelgroupId)
    spacyConfig = SpacyConfig.from_local(clientCode, modelgroupId)

    # validate ner postprocess indicators
    ner_postprocess_indicators = _validate_postprocess_indicators(indicators=nerDataConfig.ner_postprocess_indicators,
                                                                  bidata=nerData.df_bi, brdata=nerData.df_br)
    nerDataConfig.ner_postprocess_indicators = ner_postprocess_indicators

    # map model res to filter
    ner_filter, data_levels, ignore_fields, data = map_filter_to_value(ner_results, nerData, nerDataConfig, process_indicator)

    # find latest time in data
    latest_time, latest_period_type = find_latest_time(data, ner_filter, {},
                                                       nerDataConfig.ner_postprocess_indicators)

    ner_filter, ner_res = legacy_filtering._postprocess_filter_given_indicator(ner_filter,
                                                                               ner_results,
                                                                               data,
                                                                               nerDataConfig.ner_postprocess_indicators,
                                                                               nerData.level_type)
    if process_indicator.legacy_preprocessing_enabled:
        df, ner_filter, ignore_fields = legacy_filtering.map_keyword_in_query(data, query, ner_filter,
                                                                              ignore_fields)
    else:
        df, ner_filter, ignore_fields = general_filtering.map_keyword_in_query(modelgroupId, data, query,
                                                                               ner_filter,
                                                                               ignore_fields)

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
    tactic_str = ', '.join(spacy_filter.get("core_dimension", []) + spacy_filter.get("core_dimension_composite", []))
    if not len(df):
        rejection_message = PromptTemplates().no_tactic_rejection_message.format(tactic=tactic_str)
    else:
        df, filter_log = legacy_filtering._apply_answer_dict_with_log(df, ner_filter,
                                                                      res_dict=ner_res,
                                                                      ignore_fields=ignore_fields,
                                                                      prioritized_indicator=prioritized_indicator)
        rejection_message = log_to_message(filter_log)
        # client specific rules, to be generalized
        df = legacy_filtering._apply_filtering_logic(clientCode, ner_filter, data=df)
        if any(x for x in ner_filter.get("main_metric", []) if x in df["metric"].unique()):
            is_halo_tagging = any(ner_filter.get(x) for x in nerData.level_type.get("halo_tagging", []))
            data_levels = legacy_filtering.get_report_levels(df=df,
                                                             level=data_levels,
                                                             ner_filter=ner_filter,
                                                             is_halo_tagging=is_halo_tagging)
            dedup_df = legacy_filtering._dedup_df(df=df,
                                                  level=data_levels,
                                                  level_type=nerData.level_type)
            ag_df, mg_df, m_df = legacy_filtering._get_level_answer(dedup_df, level=data_levels)
        else:
            rejection_message = PromptTemplates().no_metric_rejection_message
    logger.debug(f'Generating filters on client {clientCode}, modelgroup {modelgroupId}, ner filter {ner_filter}')

    nerFilters = NerFilters(query=query, ner_filter=ner_filter, ner_res=ner_res)
    nerFilters.validate()

    logger.debug(f'ner filter: {ner_filter}')


    if ("brand_focused_marketing" in df.columns) and ("custom_aggregated" in df.columns) and (
        not nerData.level_type.get("halo_tagging")) and ("yes" in df["custom_aggregated"]):
        nerData.level_type["halo_tagging"] = ["brand_focused_marketing"]

    ner_res.update({'time_tag': ner_filter["time_tag"]})
    ner_filter.update({'level': nerData.level_type})
    ner_filter.update({"rejection_message": rejection_message})
    ner_filter.update({'time': df["time"].unique().tolist()})
    ner_filter.update({'period_type': df["period_type"].unique().tolist()})
    ner_filter.update({"latest_time": int(latest_time), "latest_period_type": latest_period_type})
    ner_filter.update({"ag_df_shape": ag_df.shape, "mg_df_shape": mg_df.shape, "m_df_shape": m_df.shape})
    readoutData = ReadoutData(ner_filters=ner_filter, ner_results=ner_res, df_activity_group=ag_df,
                          df_measure_group=mg_df, df_measure=m_df)
    return readoutData
