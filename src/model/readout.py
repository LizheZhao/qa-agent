import json
import logging
import os
import pathlib
import re
import sys
from typing import Optional, Dict, Iterator
import numpy as np
import pandas as pd
import importlib

from src.data.data_interface import ReadoutData, ProcessIndicator
from src.data.local import get_data_path
from src.integrations.llm import generate_text, stream_text
from src.integrations.llm_external import generate_analysis_response
from src.model.common import PromptTemplates, truncate_by_tokens
import src.model.readout_utils as readout_utils
from src.model.readout_utils import _get_driver_tag_default
from src.utils import load_yaml_file

logger = logging.getLogger(__name__)


# def _process_bi_detail_COLGUS_tp(client_code: str, model_group_id: int, df_dict: dict[str, pd.DataFrame],
#                                  driver_tags: tuple,
#                                  ner_filter: dict, ner_res_dict: dict, map_dict: dict, metric_df: pd.DataFrame):
#     file_mapping_bi = {'tactics': 'ag_df', 'tactics_detail': 'mg_df', 'detail_tactics': 'm_df'}
#     file_type, data_type, driver_high, driver_detail = driver_tags
#
#     intention = ner_filter['intention'][0]
#     media_channel_check = ner_res_dict.get('media_channel', 'irrelevant')
#     media_channel_ls = ner_filter.get('media_channel', [])
#     trend_check, rank_check = ner_res_dict.get('trend', 'no'), ner_res_dict.get('rank', 'na')
#     how_many = ner_res_dict.get('how_many', 'na')
#
#     hispanic_check, retailer_check, prog_check, platform_check = ner_res_dict.get('hispanic',
#                                                                                   'irrelevant'), ner_filter.get(
#         'retailer', 'no'), ner_res_dict.get('programmatic', 'irrelevant'), ner_filter.get('platform', 'irrelevant')
#     condition_extra = hispanic_check != 'irrelevant' or retailer_check != 'no' or prog_check != 'irrelevant' or platform_check != 'irrelevant'
#     rank_metric, sort_metric = map_dict[intention]['rankMetric'], map_dict[intention]['sortMetric']
#     metrics_order = ner_filter['metric']
#     # load benchmark data
#     benchmark_idx = readout_utils.load_benchmark_data(client_code, model_group_id)
#     planner_driver_data = readout_utils.load_planner_data(client_code, model_group_id)
#     period_type = ner_filter.get('period_type', ['year'])[0]
#     fixtext = ""
#     pillar_data, pillar_pivot_table, str_pillar = pd.DataFrame(), pd.DataFrame(), ""
#     core_dimension = ner_filter.get('core_dimension', {})
#     core_dimension_composite = ner_filter.get('core_dimension_composite', {})
#     product_halo_check = ner_filter.get('product_halo', 'irrelevant')
#     pivot_sort_col = ""
#     dim_cols = readout_utils._get_dim_col_in_pivot_table(df_dict, ner_filter)
#
#     if intention == 'spending' and product_halo_check == 'irrelevant' and not core_dimension_composite and not core_dimension and not condition_extra and media_channel_check == 'irrelevant' and not (
#             isinstance(media_channel_ls, list) and len(media_channel_ls) != 0):
#         str_q, pivot_table = readout_utils._overall_metric(client_code, model_group_id, ner_filter, ner_res_dict,
#                                                            driver_tags, dim_cols, metric_df)
#         # no need to include benchmark info for this type of questions
#         benchmark_data = pd.DataFrame()
#         benchmark_str = ""
#         planner_data = pd.DataFrame()
#     elif intention == 'planner':
#         str_q, fixtext, pivot_table, benchmark_data, benchmark_str, planner_data = readout_utils._planner_intention_res(
#             ner_filter, planner_driver_data, product_halo_check, media_channel_check, media_channel_ls)
#     else:
#         # read in and clean data
#         curr_df_name = file_mapping_bi[driver_detail]
#         curr_df = df_dict[curr_df_name]
#         curr_df = readout_utils._filter_paid_search(intention, curr_df)
#
#         if curr_df.empty:
#             raise ValueError('No valid data.')
#
#         # check product media
#         if (curr_df['custom_aggregated'] != 'no').all():
#             product_media_check = True
#         else:
#             product_media_check = False
#         curr_df = readout_utils._filter_pillar(product_halo_check, core_dimension, curr_df)
#         # further clean
#         data1 = readout_utils._read_and_process_csv(curr_df, ner_filter)
#
#         overall_roi_condition = intention in ['margin roi', 'performance'] and media_channel_check in ['all',
#                                                                                                        'irrelevant'] and not core_dimension and not condition_extra and not product_media_check and product_halo_check == 'irrelevant'
#         overall_spend_condition = intention in ['spending'] and media_channel_check in ['all',
#                                                                                         'irrelevant'] and not core_dimension and not condition_extra and not product_media_check and product_halo_check == 'irrelevant'
#         # provide overall roi/spend info in fixtext which is appended to LLM response
#         if overall_spend_condition or overall_roi_condition:
#             fixtext, _ = readout_utils._overall_metric(client_code, model_group_id, ner_filter, ner_res_dict,
#                                                        driver_tags, dim_cols, metric_df)
#             fixtext = readout_utils.convert_display_format(fixtext)
#             data1 = data1[~data1['media_channel'].isin(['special media', 'non-media'])]
#
#         data1 = readout_utils._filter_special_media(product_halo_check, data1)
#
#         if driver_high != 'media_channel':
#             data = data1
#         else:
#             data = readout_utils._concatenate_ag_mg_bi(data1)
#         # biz rule: pillar data to show aggregated level data
#         pillar_data, data = readout_utils._get_pillar_data(period_type, product_media_check, core_dimension, data)
#
#         ner_dict = readout_utils._get_ner_key_value(data, driver_high, file_type, data_type, driver_detail, dim_cols)
#         # biz rule: product media (in tactics_detail col) as product when condition meets
#         data, ner_dict = readout_utils._switch_dimension_col(ner_dict, data)
#         # compute percentage change by time
#         data = readout_utils._calc_growth(period_type, ner_dict, data)
#         if data.empty:
#             raise ValueError('No valid data.')
#         # further cleaned and sort by metrics
#         growth_col, sort_by_col = readout_utils._get_growth_sort_col(trend_check, data)
#         data = readout_utils._sort_filt_data(data, ner_dict, sort_metric, sort_by_col, rank_check, metric_df)
#         data = readout_utils._ignore_growth_outlier(growth_col, data)
#         # generate table display
#         pivot_table, pivot_sort_col = readout_utils._get_granular_table(data, ner_dict, metrics_order, ner_filter,
#                                                                         rank_check, sort_by_col, client_code, metric_df)
#
#         # get top performing tactics data, support user given top number
#         top_data = readout_utils._top_data(data, ner_dict, how_many)
#         top_data = readout_utils._filter_top_media_channel(driver_high, how_many, ner_filter, ner_dict, sort_by_col,
#                                                            top_data)
#
#         # get planner and benchmark info
#         planner_data = readout_utils._merge_planner(intention, media_channel_check, core_dimension,
#                                                     core_dimension_composite, data, planner_driver_data)
#         benchmark_str, benchmark_data = readout_utils._merge_benchmark(intention, media_channel_check, core_dimension,
#                                                                        core_dimension_composite, ner_dict, data,
#                                                                        top_data, benchmark_idx)
#
#         str_q = readout_utils._get_readout(intention, growth_col, trend_check, rank_check, sort_metric, how_many,
#                                            ner_dict, ner_filter, data, top_data, metric_df)
#
#         if not pillar_data.empty:
#             pillar_pivot_table, _ = readout_utils._get_granular_table(pillar_data, ner_dict, metrics_order, ner_filter,
#                                                                       rank_check, sort_by_col, client_code, metric_df)
#             ner_dict_pillar = ner_dict.copy()
#             ner_dict_pillar['driver'] = 'tactics_detail'
#             str_pillar = f"Overall {intention} of each product media is as follows:" + readout_utils._get_granular_level_pivot(
#                 pillar_data, ner_dict_pillar, ner_filter, trend_check, metric_df)
#
#         if str_pillar:
#             str_q = str_pillar + str_q
#
#     return str_q, pivot_table, benchmark_data, benchmark_str, planner_data, fixtext, pillar_pivot_table, pivot_sort_col
#
#
# def _process_bi_COLGUS_tp(client_code: str, model_group_id: int, df_dict: dict[str, pd.DataFrame], driver_tags: tuple,
#                           ner_filter: dict, ner_res_dict: dict, map_dict: dict, metric_df: pd.DataFrame):
#     core_dimension = ner_filter.get('core_dimension', {})
#     core_dimension_composite = ner_filter.get('core_dimension_composite', {})
#     product_halo_check = ner_filter.get('product_halo', 'irrelevant')
#     dim_cols = readout_utils._get_dim_col_in_pivot_table(df_dict, ner_filter)
#     try:
#         str_q, pivot_table, benchmark_data, benchmark_str, planner_data, fixtext, pillar_pivot_table, pivot_sort_col = _process_bi_detail_COLGUS_tp(
#             client_code, model_group_id, df_dict, driver_tags, ner_filter, ner_res_dict, map_dict, metric_df)
#     except Exception as e:
#         logger.warning(f"Failed to process initial BI details: {e}")
#         str_q, pivot_table, benchmark_data, benchmark_str, planner_data, fixtext, pillar_pivot_table, pivot_sort_col = "", pd.DataFrame(), pd.DataFrame(), "", pd.DataFrame(), "", pd.DataFrame(), ""
#     if (core_dimension_composite or core_dimension):
#         search_level_com = readout_utils._check_key_or_value(core_dimension_composite,
#                                                              ['activity_group', 'measure_group', 'measure'])
#         search_level = readout_utils._check_key_or_value(core_dimension, ['activity_group', 'measure_group', 'measure'])
#         driver_tags_m = 'm', 'bi', 'detail_tactics', 'detail_tactics'
#         driver_tags_ag = 'ag', 'bi', 'tactics', 'tactics'
#         driver_tags_mg = 'mg', 'bi', 'tactics_detail', 'tactics_detail'
#         file_type, _, _, _ = driver_tags
#         if file_type != 'mg':
#             try:
#                 str_q, pivot_table, benchmark_data, benchmark_str, planner_data, fixtext, pillar_pivot_table, pivot_sort_col = _process_bi_detail_COLGUS_tp(
#                     client_code, model_group_id, df_dict, driver_tags_mg, ner_filter, ner_res_dict, map_dict, metric_df)
#             except Exception as e:
#                 logger.warning(f"Failed to process measure level: {e}")
#                 str_q, pivot_table, benchmark_data, benchmark_str, planner_data, fixtext, pillar_pivot_table, pivot_sort_col = "", pd.DataFrame(), pd.DataFrame(), "", pd.DataFrame(), "", pd.DataFrame(), ""
#         try:
#             str_q_m_com, pivot_table_m_com, benchmark_data_m_com, benchmark_str_m_com, planner_data_m_com, fixtext_m_com, pillar_pivot_table_m_com, pivot_sort_col_m_com = _process_bi_detail_COLGUS_tp(
#                 client_code, model_group_id, df_dict, driver_tags_m, ner_filter, ner_res_dict, map_dict, metric_df)
#         except Exception as e:
#             logger.warning(f"Failed to process measure level: {e}")
#             str_q_m_com, pivot_table_m_com, benchmark_data_m_com, benchmark_str_m_com, planner_data_m_com, fixtext_m_com, pillar_pivot_table_m_com, pivot_sort_col_m_com = "", pd.DataFrame(), pd.DataFrame(), "", pd.DataFrame(), "", pd.DataFrame(), ""
#         try:
#             str_q_ag, pivot_table_ag, benchmark_data_ag, benchmark_str_ag, planner_data_ag, fixtext_ag, pillar_pivot_table_ag, pivot_sort_col_ag = _process_bi_detail_COLGUS_tp(
#                 client_code, model_group_id, df_dict, driver_tags_ag, ner_filter, ner_res_dict, map_dict, metric_df)
#         except Exception as e:
#             logger.warning(f"Failed to process activitygroup level: {e}")
#             str_q_ag, pivot_table_ag, benchmark_data_ag, benchmark_str_ag, planner_data_ag, fixtext_ag, pillar_pivot_table_ag, pivot_sort_col_ag = "", pd.DataFrame(), pd.DataFrame(), "", pd.DataFrame(), "", pd.DataFrame(), ""
#
#         str_q, pillar_pivot_table, pivot_table = readout_utils._core_dim_readout(search_level, search_level_com,
#                                                                                  pivot_table_m_com, pivot_table_ag,
#                                                                                  pivot_table, pillar_pivot_table,
#                                                                                  str_q_ag, str_q_m_com, str_q)
#     str_q = readout_utils._metrics_aka(str_q, ner_filter)
#     str_q = readout_utils._colgate_specific_readout_reformat(str_q)
#     pillar_pivot_table, pivot_table = readout_utils._colgate_specific_table_reformat(dim_cols, product_halo_check,
#                                                                                      pivot_sort_col,
#                                                                                      core_dimension_composite,
#                                                                                      core_dimension, pivot_table,
#                                                                                      pillar_pivot_table, metric_df)
#
#     return str_q, pivot_table, benchmark_data, benchmark_str, planner_data, fixtext, pillar_pivot_table
#
#
# def _process_br_COLGUS_tp(client_code: str, model_group_id: int, df_dict, driver_tags, ner_filter, ner_res_dict,
#                           map_dict, metric_df):
#     fixtext = ''
#     file_mapping_br = {'business_driver': 'ag_df', 'business_driver_detail': 'mg_df', 'tactics': 'm_df'}
#     intention, biz_driver_check = ner_res_dict['intention'][0], ner_res_dict.get('business_driver', 'irrelevant')
#     biz_driver_ls = ner_filter.get('business_driver', [])
#     hispanic_check, retailer_check, prog_check, platform_check = ner_res_dict.get('hispanic',
#                                                                                   'irrelevant'), ner_filter.get(
#         'retailer', 'no'), ner_res_dict.get('programmatic', 'irrelevant'), ner_filter.get('platform', 'irrelevant')
#     condition_extra = hispanic_check != 'irrelevant' or retailer_check != 'no' or prog_check != 'irrelevant' or platform_check != 'irrelevant'
#     core_dimension = ner_filter.get('core_dimension', {})
#     core_dimension_composite = ner_filter.get('core_dimension_composite', {})
#     trend_check, rank_check = ner_res_dict.get('trend', 'no'), ner_res_dict.get('rank', 'na')
#     trend_check = 'no' if intention == 'source of change' else trend_check
#     how_many = ner_res_dict.get('how_many', 'na')
#     period_type = ner_filter.get('period_type', ['year'])[0]
#     if intention != 'sales' and biz_driver_check == 'all':
#         file_type, data_type, driver_high, driver_detail = 'm', 'br', 'business_driver', 'tactics'
#     else:
#         file_type, data_type, driver_high, driver_detail = driver_tags
#     rank_metric, sort_metric = map_dict[intention]['rankMetric'], map_dict[intention]['sortMetric']
#     media_channel_check = ner_res_dict.get('media_channel', 'irrelevant')
#     metrics_order = ner_filter['metric']
#     pivot_table_biz = pd.DataFrame()
#     product_halo_check = ner_filter.get('product_halo', 'irrelevant')
#     dim_cols = readout_utils._get_dim_col_in_pivot_table(df_dict, ner_filter)
#
#     if intention == 'sales' and product_halo_check == 'irrelevant' and biz_driver_check == 'irrelevant' and not (
#             isinstance(biz_driver_ls, list) and len(
#         biz_driver_ls) != 0) and not condition_extra and not core_dimension and not core_dimension_composite:
#         str_q, pivot_table = readout_utils._overall_metric(client_code, model_group_id, ner_filter, ner_res_dict,
#                                                            driver_tags, dim_cols, metric_df)
#     elif product_halo_check != 'irrelevant':
#         raise NotImplementedError("Reject answering halo questions.")
#     else:
#         curr_df_name_biz = file_mapping_br['business_driver_detail']
#         curr_df_biz = df_dict[curr_df_name_biz]
#         data1 = readout_utils._read_and_process_csv(curr_df_biz, ner_filter)
#
#         curr_df_name = file_mapping_br[driver_detail]
#         curr_df = df_dict[curr_df_name]
#         curr_df = readout_utils._empty_mg_replace(df_dict, file_mapping_br, curr_df_name, curr_df)
#
#         if curr_df.empty:
#             raise ValueError('No valid data.')
#         data2 = readout_utils._read_and_process_csv(curr_df, ner_filter)
#         ner_dict = readout_utils._get_ner_key_value(data2, driver_high, file_type, data_type, driver_detail, dim_cols)
#         ner_dict1 = readout_utils._get_ner_key_value(data1, driver_high, 'mg', 'br', 'business_driver_detail', dim_cols)
#
#         data1 = readout_utils._filter_media_promotion(intention, biz_driver_check, media_channel_check, data1)
#         data2 = readout_utils._filter_media_promotion(intention, biz_driver_check, media_channel_check, data2)
#         if data2.empty:
#             raise ValueError('No valid data.')
#
#         data1 = readout_utils._calc_growth_br(intention, ner_res_dict, period_type, ner_dict1, data1)
#         data2 = readout_utils._calc_growth_br(intention, ner_res_dict, period_type, ner_dict, data2)
#         data1 = readout_utils._exclude_system(intention, data1)
#         data2 = readout_utils._exclude_system(intention, data2)
#         growth_col, sort_by_col = readout_utils._get_growth_sort_col(trend_check, data2)
#         data1 = readout_utils._ignore_growth_outlier(growth_col, data1)
#         data2 = readout_utils._ignore_growth_outlier(growth_col, data2)
#         data2 = readout_utils._sort_filt_data(data2, ner_dict, sort_metric, sort_by_col, rank_check, metric_df)
#
#         top_data = readout_utils._top_data(data2, ner_dict, how_many)
#
#         fixtext = readout_utils._get_fixtext_br(intention, biz_driver_check, media_channel_check, ner_dict1, ner_filter,
#                                                 trend_check, driver_detail, curr_df, data1, metric_df)
#         str_q = readout_utils._get_readout(intention, growth_col, trend_check, rank_check, sort_metric, how_many,
#                                            ner_dict, ner_filter, data2, top_data, metric_df)
#         # str_q = readout_utils._get_readout_br(intention, driver_detail, str_q)
#
#         pivot_table, pivot_sort_col = readout_utils._get_granular_table(data2, ner_dict, metrics_order, ner_filter,
#                                                                         rank_check, sort_by_col, client_code, metric_df)
#         pivot_table_biz = readout_utils._get_pivot_biz_table(intention, biz_driver_check, media_channel_check,
#                                                              ner_dict1, ner_filter, rank_check, sort_by_col,
#                                                              metrics_order, client_code, data1, metric_df)
#
#         str_q, fixtext = readout_utils._colgate_specific_soc(intention, ner_dict1, ner_filter, trend_check, data1,
#                                                              pivot_table_biz, str_q, fixtext, metric_df)
#         fixtext = readout_utils.convert_display_format(fixtext)
#         if intention == 'source of change':
#             ner_filter_copy = ner_filter.copy()
#             ner_filter_copy['main_metric'] = ['sales']
#             str_q_overall_sales, _ = readout_utils._overall_metric(client_code, model_group_id, ner_filter_copy,
#                                                                    ner_res_dict, driver_tags, dim_cols, metric_df)
#             fixtext += str_q_overall_sales
#         str_q = readout_utils._metrics_aka(str_q, ner_filter)
#         pivot_table = readout_utils._drop_single_dim(dim_cols, pivot_table)
#         pivot_table_biz = readout_utils._drop_single_dim(dim_cols, pivot_table_biz)
#         pivot_table, pivot_table_biz = readout_utils._br_table_reformat(pivot_table, pivot_table_biz)
#
#     return str_q, pivot_table, pd.DataFrame(), "", pd.DataFrame(), fixtext, pivot_table_biz
#
#
# def _process_bi_detail_COLGUS_other_category(client_code: str, model_group_id: int, df_dict: dict[str, pd.DataFrame],
#                                              driver_tags: tuple,
#                                              ner_filter: dict, ner_res_dict: dict, map_dict: dict,
#                                              metric_df: pd.DataFrame):
#     file_mapping_bi = {'tactics': 'ag_df', 'detail_tactics': 'mg_df', 'tactics_detail': 'm_df'}
#
#     file_type, data_type, driver_high, driver_detail = driver_tags
#
#     intention = ner_filter['intention'][0]
#     media_channel_check = ner_res_dict.get('media_channel', 'irrelevant')
#     media_channel_ls = ner_filter.get('media_channel', [])
#     trend_check, rank_check = ner_res_dict.get('trend', 'no'), ner_res_dict.get('rank', 'na')
#     how_many = ner_res_dict.get('how_many', 'na')
#     product_media_check = 'product_focused_media' in ner_filter.keys()  # True: custom_agg data
#     retailer_check, prog_check = ner_filter.get('retailer', 'no'), ner_res_dict.get('programmatic', 'irrelevant')
#     condition_extra = retailer_check != 'no' or prog_check != 'irrelevant'
#     rank_metric, sort_metric = map_dict[intention]['rankMetric'], map_dict[intention]['sortMetric']
#     metrics_order = ner_filter['metric']
#     period_type = ner_filter.get('period_type', ['year'])[0]
#     fixtext = ""
#     pivot_sort_col = ""
#     core_dimension = ner_filter.get('core_dimension', {})
#     dim_cols = readout_utils._get_dim_col_in_pivot_table(df_dict, ner_filter)
#
#     # for questions asking about overall spending
#     if intention == 'spending' and media_channel_check == 'irrelevant' and not (
#             isinstance(media_channel_ls, list) and len(
#         media_channel_ls) != 0) and not condition_extra and not product_media_check and not core_dimension:
#         str_q, pivot_table = readout_utils._overall_metric(client_code, model_group_id, ner_filter, ner_res_dict,
#                                                            driver_tags, dim_cols, metric_df)
#     elif intention == 'planner':
#         str_q, fixtext, pivot_table, _, _, _ = readout_utils._planner_intention_res(ner_filter, pd.DataFrame(),
#                                                                                     'irrelevant', media_channel_check,
#                                                                                     media_channel_ls)
#     else:
#         # read in and clean driver_detail dataframe
#         curr_df_name = file_mapping_bi[driver_detail]
#         curr_df = df_dict[curr_df_name]
#         curr_df = readout_utils._filter_paid_search(intention, curr_df)
#
#         if curr_df.empty:
#             raise ValueError('No valid data.')
#
#         data1 = readout_utils._read_and_process_csv_other_category(curr_df, ner_filter)
#         overall_spend_condition = intention in ['spending'] and media_channel_check in ['all',
#                                                                                         'irrelevant'] and not core_dimension and not condition_extra and not product_media_check
#         if overall_spend_condition:
#             fixtext, _ = readout_utils._overall_metric(client_code, model_group_id, ner_filter, ner_res_dict,
#                                                        driver_tags, dim_cols, metric_df)
#             fixtext = readout_utils.convert_display_format(fixtext)
#
#         if driver_high != 'media_channel':
#             data = data1
#         else:
#             data = readout_utils._concatenate_ag_mg_bi(data1, 'detail_tactics')
#
#         ner_dict = readout_utils._get_ner_key_value_other_category(data, driver_high, file_type, data_type,
#                                                                    driver_detail)
#         # compute percentage change
#         data = readout_utils._calc_growth(period_type, ner_dict, data)
#         if data.empty:
#             raise ValueError('No valid data.')
#         growth_col, sort_by_col = readout_utils._get_growth_sort_col(trend_check, data)
#         data = readout_utils._sort_filt_data(data, ner_dict, sort_metric, sort_by_col, rank_check, metric_df)
#         data = readout_utils._ignore_growth_outlier(growth_col, data)
#         # generate table display
#         pivot_table, pivot_sort_col = readout_utils._get_granular_table(data, ner_dict, metrics_order, ner_filter,
#                                                                         rank_check, sort_by_col, client_code, metric_df)
#         # get top performing tactics data, support user given top number
#         top_data = readout_utils._top_data(data, ner_dict, how_many)
#         top_data = readout_utils._filter_top_media_channel(driver_high, how_many, ner_filter, ner_dict, sort_by_col,
#                                                            top_data)
#
#         # # get planner and benchmark info
#         # planner_data = readout_utils._merge_planner(intention, media_channel_check, core_dimension, {}, data,
#         #                                             pd.DataFrame())
#         # benchmark_str, benchmark_data = readout_utils._merge_benchmark(intention, media_channel_check, core_dimension,
#         #                                                                {}, ner_dict, data, top_data, pd.DataFrame())
#
#         str_q = readout_utils._get_readout(intention, growth_col, trend_check, rank_check, sort_metric, how_many,
#                                            ner_dict, ner_filter, data, top_data, metric_df)
#
#     return str_q, pivot_table, pd.DataFrame(), "", pd.DataFrame(), fixtext, pd.DataFrame(), pivot_sort_col
#
#
# def _process_bi_COLGUS_other_category(client_code: str, model_group_id: int, df_dict: dict[str, pd.DataFrame],
#                                       driver_tags: tuple,
#                                       ner_filter: dict, ner_res_dict: dict, map_dict: dict, metric_df: pd.DataFrame):
#     core_dimension = ner_filter.get('core_dimension', {})
#     dim_cols = readout_utils._get_dim_col_in_pivot_table(df_dict, ner_filter)
#     try:
#         str_q, pivot_table, _, _, _, fixtext, pivot_biz_table, pivot_sort_col = _process_bi_detail_COLGUS_other_category(
#             client_code, model_group_id, df_dict, driver_tags, ner_filter, ner_res_dict, map_dict, metric_df)
#     except Exception as e:
#         logger.warning(f"Failed to process initial BI details: {e}")
#         str_q, pivot_table, _, _, _, fixtext, pivot_biz_table, pivot_sort_col = "", pd.DataFrame(), pd.DataFrame(), "", pd.DataFrame(), "", pd.DataFrame(), ""
#     if core_dimension:
#         search_level = readout_utils._check_key_or_value(core_dimension, ['activity_group', 'measure_group', 'measure'])
#         if search_level['measure'] and search_level['activity_group']:
#             driver_tags_ag = 'ag', 'bi', 'tactics', 'tactics'
#             try:
#                 str_q_ag, pivot_table_ag, _, _, _, fixtext_ag, pivot_biz_table_ag, _ = _process_bi_detail_COLGUS_other_category(
#                     client_code, model_group_id, df_dict, driver_tags_ag, ner_filter, ner_res_dict, map_dict, metric_df)
#             except Exception as e:
#                 logger.warning(f"Failed to process activitygroup level: {e}")
#                 str_q_ag, pivot_table_ag, fixtext_ag, pivot_biz_table_ag = "", pd.DataFrame(), "", pd.DataFrame()
#             pivot_biz_table = pivot_table_ag
#             str_q = str_q_ag + 'At Tactics level: ' + str_q
#         elif not search_level['measure'] and search_level['activity_group']:
#             driver_tags_m = 'mg', 'bi', 'media_channel', 'detail_tactics'
#             try:
#                 str_q_m, pivot_table_m, _, _, _, fixtext_m, pivot_biz_table_m, _ = _process_bi_detail_COLGUS_other_category(
#                     client_code, model_group_id, df_dict, driver_tags_m, ner_filter, ner_res_dict, map_dict, metric_df)
#             except Exception as e:
#                 logger.warning(f"Failed to process activitygroup level: {e}")
#                 str_q_m, pivot_table_m, fixtext_m, pivot_biz_table_m = "", pd.DataFrame(), "", pd.DataFrame()
#             if not pivot_table_m.empty:
#                 pivot_biz_table = pivot_table
#                 pivot_table = pivot_table_m
#             str_q = str_q + 'At Tactics level: ' + str_q_m
#     str_q = readout_utils._metrics_aka(str_q, ner_filter)
#     dim_col = pivot_table.columns[0] if not pivot_table.empty else ""
#     dim_col_biz = pivot_biz_table.columns[0] if not pivot_biz_table.empty else ""
#     pivot_table = readout_utils._dynamic_top([dim_col], pivot_sort_col, pivot_table, metric_df)
#     pivot_table = readout_utils._drop_single_dim([dim_col], pivot_table)
#     pivot_table = readout_utils._rename_product_media_other_category(ner_filter, dim_col, pivot_table)
#     pivot_biz_table = readout_utils._dynamic_top([dim_col_biz], pivot_sort_col, pivot_biz_table, metric_df)
#     pivot_biz_table = readout_utils._drop_single_dim([dim_col], pivot_biz_table)
#     pivot_biz_table = readout_utils._rename_product_media_other_category(ner_filter, dim_col_biz, pivot_biz_table)
#
#     return str_q, pivot_table, pd.DataFrame(), "", pd.DataFrame(), fixtext, pivot_biz_table
#
#
# def _process_br_COLGUS_other_category(client_code: str, model_group_id: int, df_dict, driver_tags, ner_filter,
#                                       ner_res_dict, map_dict, metric_df: pd.DataFrame):
#     fixtext = ""
#     file_mapping_br = {'business_driver_detail': 'ag_df', 'tactics': 'mg_df', 'detail_tactics': 'm_df'}
#     intention, biz_driver_check = ner_res_dict['intention'][0], ner_res_dict.get('business_driver', 'irrelevant')
#     biz_driver_ls = ner_filter.get('business_driver', [])
#     trend_check, rank_check = ner_res_dict.get('trend', 'no'), ner_res_dict.get('rank', 'na')
#     how_many = ner_res_dict.get('how_many', 'na')
#     product_media_check = 'product_focused_media' in ner_filter.keys()
#     retailer_check, prog_check = ner_filter.get('retailer', 'no'), ner_res_dict.get('programmatic', 'irrelevant')
#     condition_extra = retailer_check != 'no' or prog_check != 'irrelevant'
#     period_type = ner_filter.get('period_type', ['year'])[0]
#     if intention != 'sales' and biz_driver_check == 'all':
#         file_type, data_type, driver_high, driver_detail = 'mg', 'br', 'business_driver', 'tactics'
#     else:
#         file_type, data_type, driver_high, driver_detail = driver_tags
#     rank_metric, sort_metric = map_dict[intention]['rankMetric'], map_dict[intention]['sortMetric']
#     media_channel_check = ner_res_dict.get('media_channel', 'irrelevant')
#     metrics_order = ner_filter['metric']
#     pivot_table_biz = pd.DataFrame()
#     core_dimension = ner_filter.get('core_dimension', {})
#     dim_cols = readout_utils._get_dim_col_in_pivot_table(df_dict, ner_filter)
#
#     if intention == 'sales' and biz_driver_check == 'irrelevant' and not (isinstance(biz_driver_ls, list) and len(
#             biz_driver_ls) != 0) and not condition_extra and not core_dimension and not product_media_check:
#         str_q, pivot_table = readout_utils._overall_metric(client_code, model_group_id, ner_filter, ner_res_dict,
#                                                            driver_tags, dim_cols, metric_df)
#     else:
#         curr_df_name_biz = file_mapping_br['business_driver_detail']
#         curr_df_biz = df_dict[curr_df_name_biz]
#         data1 = readout_utils._read_and_process_csv_other_category(curr_df_biz, ner_filter)
#
#         curr_df_name = file_mapping_br[driver_detail]
#         curr_df = df_dict[curr_df_name]
#         curr_df = readout_utils._empty_mg_replace_other_category(df_dict, file_mapping_br, curr_df_name, curr_df)
#         if curr_df.empty:
#             raise ValueError('No valid data.')
#         overall_sales_condition = intention in ['sales'] and biz_driver_check in ['all',
#                                                                                   'irrelevant'] and not condition_extra
#         if overall_sales_condition:
#             fixtext, _ = readout_utils._overall_metric(client_code, model_group_id, ner_filter, ner_res_dict,
#                                                        driver_tags, dim_cols, metric_df)
#
#         data2 = readout_utils._read_and_process_csv_other_category(curr_df, ner_filter)
#         ner_dict = readout_utils._get_ner_key_value_other_category(data2, driver_high, file_type, data_type,
#                                                                    driver_detail)
#         ner_dict1 = readout_utils._get_ner_key_value_other_category(data1, driver_high, 'ag', 'br',
#                                                                     'business_driver_detail')
#
#         if intention == 'contribution' and 'other' in curr_df[
#             'business_driver'].unique().tolist() and biz_driver_check != 'all':
#             fixtext += 'We only provide contribution by marketing drivers. '
#         data1 = readout_utils._filter_media_promotion(intention, biz_driver_check, media_channel_check, data1)
#         data2 = readout_utils._filter_media_promotion(intention, biz_driver_check, media_channel_check, data2)
#         if data2.empty:
#             raise ValueError('No valid data.')
#
#         data1 = readout_utils._calc_growth_br(intention, ner_res_dict, period_type, ner_dict1, data1)
#         data2 = readout_utils._calc_growth_br(intention, ner_res_dict, period_type, ner_dict, data2)
#         data1 = readout_utils._exclude_system(intention, data1)
#         data2 = readout_utils._exclude_system(intention, data2)
#         growth_col, sort_by_col = readout_utils._get_growth_sort_col(trend_check, data2)
#         data1 = readout_utils._ignore_growth_outlier(growth_col, data1)
#         data2 = readout_utils._ignore_growth_outlier(growth_col, data2)
#         data2 = readout_utils._sort_filt_data(data2, ner_dict, sort_metric, sort_by_col, rank_check, metric_df)
#
#         top_data = readout_utils._top_data(data2, ner_dict, how_many)
#
#         fixtext = readout_utils._get_fixtext_br(intention, biz_driver_check, media_channel_check, ner_dict1, ner_filter,
#                                                 trend_check, driver_detail, curr_df, data1, metric_df)
#         str_q = readout_utils._get_readout(intention, growth_col, trend_check, rank_check, sort_metric, how_many,
#                                            ner_dict, ner_filter, data2, top_data, metric_df)
#         # str_q = readout_utils._get_readout_br(intention, driver_detail, str_q)
#
#         pivot_table, pivot_sort_col = readout_utils._get_granular_table(data2, ner_dict, metrics_order, ner_filter,
#                                                                         rank_check, sort_by_col, client_code, metric_df)
#         pivot_table_biz = readout_utils._get_pivot_biz_table(intention, biz_driver_check, media_channel_check,
#                                                              ner_dict1, ner_filter, rank_check, sort_by_col,
#                                                              metrics_order, client_code, data1, metric_df)
#
#         str_q, fixtext = readout_utils._colgate_specific_soc(intention, ner_dict1, ner_filter, trend_check, data1,
#                                                              pivot_table_biz, str_q, fixtext, metric_df)
#         if intention == 'source of change':
#             ner_filter_copy = ner_filter.copy()
#             ner_filter_copy['main_metric'] = ['sales']
#             str_q_overall_sales, _ = readout_utils._overall_metric(client_code, model_group_id, ner_filter_copy,
#                                                                    ner_res_dict, driver_tags, dim_cols, metric_df)
#             fixtext += str_q_overall_sales
#         fixtext = readout_utils.convert_display_format(fixtext)
#         str_q = readout_utils._metrics_aka(str_q, ner_filter)
#         dim_col = pivot_table.columns[0] if not pivot_table.empty else ""
#         dim_col_biz = pivot_table_biz.columns[0] if not pivot_table_biz.empty else ""
#         pivot_table = readout_utils._dynamic_top([dim_col], pivot_sort_col, pivot_table, metric_df)
#         pivot_table = readout_utils._drop_single_dim([dim_col], pivot_table)
#         pivot_table_biz = readout_utils._dynamic_top([dim_col_biz], pivot_sort_col, pivot_table_biz, metric_df)
#         pivot_table_biz = readout_utils._drop_single_dim([dim_col_biz], pivot_table_biz)
#         pivot_table, pivot_table_biz = readout_utils._br_table_reformat(pivot_table, pivot_table_biz)
#
#     return str_q, pivot_table, pd.DataFrame(), "", pd.DataFrame(), fixtext, pivot_table_biz


# def _process_bi_detail_FTR(client_code: str, model_group_id: int, df_dict: dict[str, pd.DataFrame], driver_tags: tuple,
#                            ner_filter: dict, ner_res_dict: dict, map_dict: dict, metric_df: pd.DataFrame):
#     file_mapping_bi = {'tactics': 'ag_df', 'tactics_detail': 'mg_df', 'detail_tactics': 'm_df'}
#     file_type, data_type, driver_high, driver_detail = driver_tags
#
#     intention = ner_filter['intention'][0]
#     trend_check, rank_check = ner_res_dict.get('trend', 'no'), ner_res_dict.get('rank', 'na')
#     how_many = ner_res_dict.get('how_many', 'na')
#     rank_metric, sort_metric = map_dict[intention]['rankMetric'], map_dict[intention]['sortMetric']
#     metrics_order = ner_filter['metric']
#     benchmark_idx = readout_utils.load_benchmark_data(client_code, model_group_id)
#     planner_df = readout_utils.load_planner_data(client_code, model_group_id)
#     planner_response_curve, overview = readout_utils.load_diminishing_returns_data(client_code, model_group_id)
#     period_type = ner_filter.get('period_type', ['year'])[0]
#     fixtext = ""
#     core_dimension = ner_filter.get('core_dimension', {})
#     # response modifier: list
#     response_mod = ner_filter.get('response_modifier', [])
#     pivot_sort_col = ""
#     pivot_table, pivot_biz_table = pd.DataFrame(), pd.DataFrame()
#     benchmark_data, benchmark_str = pd.DataFrame(), ""
#     planner_table = pd.DataFrame()
#     product_media_check = False
#     dim_cols = readout_utils._get_dim_col_in_pivot_table(df_dict, ner_filter)
#
#     if intention == 'planner':
#         m_df = df_dict['m_df']
#         fixtext, planner_table = (
#             readout_utils._planner_output(planner_df, m_df, metric_df, core_dimension, response_mod, planner_response_curve,
#                                           overview, "Short-Term Gross Adds")
#         )
#         str_q, fixtext_end, pivot_table_end, _, _, _ = readout_utils._planner_intention_res(
#             ner_filter, pd.DataFrame(), 'irrelevant', 'irrelevant', [])
#         fixtext += fixtext_end
#     else:
#         # read in and clean data
#         curr_df_name = file_mapping_bi[driver_detail]
#         curr_df = df_dict[curr_df_name]
#
#         if curr_df.empty:
#             raise ValueError('No valid data.')
#
#         # check product media
#         if (curr_df['custom_aggregated'] != 'no').all():
#             product_media_check = True
#         # further clean
#         data1 = readout_utils._read_and_process_csv(curr_df, ner_filter)
#
#         overall_roi_condition = intention in ['margin roi'] and not core_dimension
#         overall_spend_condition = intention in ['spending'] and not core_dimension
#         overall_perfom_condition = intention in ['performance'] and not core_dimension
#         # provide overall roi/spend info in fixtext which is appended to LLM response
#         if overall_spend_condition or overall_perfom_condition or overall_roi_condition:
#             fixtext, _ = readout_utils._overall_metric(client_code, model_group_id, ner_filter, ner_res_dict,
#                                                        driver_tags, dim_cols, metric_df)
#             fixtext = readout_utils.convert_display_format(fixtext)
#
#         data = data1
#
#         ner_dict = readout_utils._get_ner_key_value(data, driver_high, file_type, data_type, driver_detail, dim_cols)
#         # compute percentage change by time
#         data = readout_utils._calc_growth(period_type, ner_dict, data)
#         if data.empty:
#             raise ValueError('No valid data.')
#         # further cleaned and sort by metrics
#         growth_col, sort_by_col = readout_utils._get_growth_sort_col(trend_check, data)
#         data = readout_utils._sort_filt_data(data, ner_dict, sort_metric, sort_by_col, rank_check, metric_df)
#         data = readout_utils._ignore_growth_outlier(growth_col, data)
#         benchmark_str, benchmark_data = readout_utils._benchmark_output(data, ner_dict, how_many, intention, core_dimension, benchmark_idx)
#         # generate table display
#         pivot_table, pivot_sort_col = readout_utils._get_granular_table(data, ner_dict, metrics_order, ner_filter,
#                                                                         rank_check, sort_by_col, client_code, metric_df)
#
#         # get top performing tactics data, support user given top number
#         top_data = readout_utils._top_data(data, ner_dict, how_many)
#
#         str_q = readout_utils._get_readout(intention, growth_col, trend_check, rank_check, sort_metric, how_many,
#                                            ner_dict, ner_filter, data, top_data, metric_df)
#
#     return str_q, pivot_table, benchmark_data, benchmark_str, planner_table, fixtext, pivot_biz_table, pivot_sort_col, product_media_check
#
#
# def _process_bi_FTR(client_code: str, model_group_id: int, df_dict: dict[str, pd.DataFrame],
#                     driver_tags: tuple,
#                     ner_filter: dict, ner_res_dict: dict, map_dict: dict, metric_df: pd.DataFrame):
#     core_dimension = ner_filter.get('core_dimension', {})
#     planner_table = pd.DataFrame()
#     dim_cols = readout_utils._get_dim_col_in_pivot_table(df_dict, ner_filter)
#     try:
#         str_q, pivot_table, _, _, planner_table, fixtext, pivot_biz_table, pivot_sort_col, product_media_check = _process_bi_detail_FTR(
#             client_code, model_group_id, df_dict, driver_tags, ner_filter, ner_res_dict, map_dict, metric_df)
#     except Exception as e:
#         logger.warning(f"Failed to process initial BI details: {e}")
#         str_q, pivot_table, fixtext, pivot_biz_table, pivot_sort_col, product_media_check = "", pd.DataFrame(), "", pd.DataFrame(), "", False
#     if core_dimension:
#         search_level = readout_utils._check_key_or_value(core_dimension, ['activity_group', 'measure_group', 'measure'])
#         if search_level['measure_group'] and search_level['activity_group']:
#             driver_tags_ag = 'ag', 'bi', 'tactics', 'tactics'
#             try:
#                 str_q_ag, pivot_table_ag, _, _, planner_table, fixtext_ag, pivot_biz_table_ag, _, _ = _process_bi_detail_FTR(
#                     client_code, model_group_id, df_dict, driver_tags_ag, ner_filter, ner_res_dict, map_dict, metric_df)
#             except Exception as e:
#                 logger.warning(f"Failed to process activitygroup level: {e}")
#                 str_q_ag, pivot_table_ag, fixtext_ag, pivot_biz_table_ag = "", pd.DataFrame(), "", pd.DataFrame()
#             if not product_media_check:
#                 pivot_biz_table = pivot_table_ag
#                 if str_q:
#                     str_q = str_q_ag + 'At Tactics level: ' + str_q
#                 elif str_q_ag and not str_q:
#                     str_q = str_q_ag
#             else:
#                 pivot_biz_table = pivot_table
#                 pivot_table = pivot_table_ag
#                 if str_q_ag:
#                     str_q = str_q + 'At Tactics level: ' + str_q_ag
#         elif not search_level['measure'] and not search_level['measure_group'] and search_level['activity_group']:
#             driver_tags_m = 'm', 'bi', 'detail_tactics', 'detail_tactics'
#             try:
#                 str_q_m, pivot_table_m, _, _, planner_table, fixtext_m, pivot_biz_table_m, _, _ = _process_bi_detail_FTR(
#                     client_code, model_group_id, df_dict, driver_tags_m, ner_filter, ner_res_dict, map_dict, metric_df)
#             except Exception as e:
#                 logger.warning(f"Failed to process activitygroup level: {e}")
#                 str_q_m, pivot_table_m, fixtext_m, pivot_biz_table_m = "", pd.DataFrame(), "", pd.DataFrame()
#             if not pivot_table_m.empty:
#                 pivot_biz_table = pivot_table
#                 pivot_table = pivot_table_m
#             if str_q_m:
#                 str_q = readout_utils._remove_noshow_readout(str_q)
#                 str_q = str_q + 'At Tactics level: ' + str_q_m
#         elif search_level['measure'] and search_level['measure_group']:
#             driver_tags_mg = 'mg', 'bi', 'tactics_detail', 'tactics_detail'
#             try:
#                 str_q_mg, pivot_table_mg, _, _, planner_table, fixtext_mg, pivot_biz_table_mg, _, _ = _process_bi_detail_FTR(
#                     client_code, model_group_id, df_dict, driver_tags_mg, ner_filter, ner_res_dict, map_dict, metric_df)
#             except Exception as e:
#                 logger.warning(f"Failed to process activitygroup level: {e}")
#                 str_q_mg, pivot_table_mg, fixtext_mg, pivot_biz_table_mg = "", pd.DataFrame(), "", pd.DataFrame()
#             pivot_biz_table = pivot_table_mg
#             if str_q_mg:
#                 str_q = str_q_mg + 'At Tactics level: ' + str_q
#         if str_q:
#             no_data_term = readout_utils.find_top_level_empty_keys(core_dimension)
#             str_q += ', '.join(no_data_term) + " not available in data." if no_data_term else ""
#     str_q = readout_utils._metrics_aka(str_q, ner_filter)
#     pivot_table = readout_utils._dynamic_top(dim_cols, pivot_sort_col, pivot_table, metric_df)
#     pivot_table = readout_utils._drop_single_dim(dim_cols, pivot_table)
#     pivot_biz_table = readout_utils._dynamic_top(dim_cols, pivot_sort_col, pivot_biz_table, metric_df)
#     pivot_biz_table = readout_utils._drop_single_dim(dim_cols, pivot_biz_table)
#     if pivot_sort_col:
#         matching_rows = metric_df[metric_df['metric'].apply(lambda x: pivot_sort_col.lower().startswith(x))]
#         rename = matching_rows['rename'].iloc[0] if not matching_rows.empty else ""
#         rename = "" if pd.isna(rename) else rename
#         original_metric = matching_rows['metric'].iloc[0] if not matching_rows.empty else ""
#         if not pivot_table.empty and rename:
#             pivot_table.columns = pivot_table.columns.str.replace(original_metric, rename.title(), case=False,
#                                                                   regex=True)
#         if not pivot_biz_table.empty and rename:
#             pivot_biz_table.columns = pivot_biz_table.columns.str.replace(original_metric, rename.title(), case=False,
#                                                                           regex=True)
#
#     return str_q, pivot_table, pd.DataFrame(), "", planner_table, fixtext, pivot_biz_table
#
#
# def _process_br_FTR(client_code: str, model_group_id: int, df_dict, driver_tags, ner_filter, ner_res_dict, map_dict,
#                     metric_df):
#     fixtext = ''
#     file_mapping_br = {'business_driver_detail': 'ag', 'tactics': 'mg', 'tactics_detail': 'm'}
#     file_type, data_type, driver_high, driver_detail = driver_tags
#     intention, biz_driver_check = ner_res_dict['intention'][0], ner_filter.get('business_driver', [])
#     trend_check, rank_check = ner_res_dict.get('trend', 'no'), ner_res_dict.get('rank', 'na')
#     trend_check = 'no' if intention == 'source of change' else trend_check
#     how_many = ner_res_dict.get('how_many', 'na')
#     period_type = ner_filter.get('period_type', ['year'])[0]
#     rank_metric, sort_metric = map_dict[intention]['rankMetric'], map_dict[intention]['sortMetric']
#     metrics_order = ner_filter['metric']
#     pivot_table, pivot_table_biz = pd.DataFrame(), pd.DataFrame()
#     pivot_sort_col, pivot_sort_col_biz = "", ""
#     core_dimension = ner_filter.get('core_dimension', {})
#     dim_cols = readout_utils._get_dim_col_in_pivot_table(df_dict, ner_filter)
#     str_q = ""
#
#     if intention == 'sales':
#         str_q, pivot_table = readout_utils._overall_metric(client_code, model_group_id, ner_filter, ner_res_dict,
#                                                            driver_tags, dim_cols, metric_df)
#     else:
#         curr_df_name_biz = file_mapping_br[driver_high] + '_df'
#         curr_df_biz = df_dict[curr_df_name_biz]
#         data1 = readout_utils._read_and_process_csv(curr_df_biz, ner_filter)
#
#         curr_df_name = file_type + '_df'
#         curr_df = df_dict[curr_df_name]
#         # curr_df = readout_utils._empty_mg_replace(df_dict, file_mapping_br, curr_df_name, curr_df)
#
#         if curr_df.empty:
#             if not curr_df_biz.empty:
#                 curr_df = curr_df_biz
#                 driver_detail = driver_high
#             else:
#                 raise ValueError('No valid data.')
#         data2 = readout_utils._read_and_process_csv(curr_df, ner_filter)
#         ner_dict = readout_utils._get_ner_key_value(data2, driver_high, file_type, data_type, driver_detail, dim_cols)
#         ner_dict['driver'] = driver_detail
#         ner_dict1 = readout_utils._get_ner_key_value(data1, driver_high, file_mapping_br[driver_high], data_type,
#                                                      driver_high, dim_cols)
#         ner_dict1['driver'] = driver_high
#
#         # data1 = readout_utils._filter_media_promotion(intention, biz_driver_check, media_channel_check, data1)
#         # data2 = readout_utils._filter_media_promotion(intention, biz_driver_check, media_channel_check, data2)
#         if data2.empty:
#             raise ValueError('No valid data.')
#
#         data1 = readout_utils._calc_growth_br(intention, ner_res_dict, period_type, ner_dict1, data1)
#         data2 = readout_utils._calc_growth_br(intention, ner_res_dict, period_type, ner_dict, data2)
#         growth_col, sort_by_col = readout_utils._get_growth_sort_col(trend_check, data2)
#         data1 = readout_utils._ignore_growth_outlier(growth_col, data1)
#         data2 = readout_utils._ignore_growth_outlier(growth_col, data2)
#         data2 = readout_utils._exclude_system(intention, data2)
#         data2 = readout_utils._filter_contribution_driver(intention, data2)
#         if not data2.empty:
#             data2 = readout_utils._sort_filt_data(data2, ner_dict, sort_metric, sort_by_col, rank_check, metric_df)
#             top_data = readout_utils._top_data(data2, ner_dict, how_many)
#             str_q = readout_utils._get_readout(intention, growth_col, trend_check, rank_check, sort_metric, how_many,
#                                                ner_dict, ner_filter, data2, top_data, metric_df)
#             str_q = readout_utils._get_readout_br(intention, driver_detail, str_q)
#
#         top_data1 = pd.DataFrame()
#         if not data1.empty:
#             data1 = readout_utils._exclude_system(intention, data1)
#             data1 = readout_utils._filter_contribution_driver(intention, data1)
#             if not data1.empty:
#                 data1 = readout_utils._sort_filt_data(data1, ner_dict1, sort_metric, sort_by_col, rank_check, metric_df)
#                 top_data1 = readout_utils._top_data(data1, ner_dict1, how_many)
#
#         if driver_high != driver_detail and not data1.empty:
#             fixtext = readout_utils._get_readout(intention, growth_col, trend_check, rank_check, sort_metric, how_many,
#                                                  ner_dict1, ner_filter, data1, top_data1, metric_df)
#             fixtext = readout_utils._get_readout_br(intention, driver_high, fixtext)
#
#         if not str_q:
#             data2 = pd.DataFrame()
#
#         if fixtext:
#             str_q = fixtext + "More granularly: " + str_q if str_q else fixtext
#             fixtext = ""
#         if not data2.empty:
#             pivot_table, pivot_sort_col = readout_utils._get_granular_table(data2, ner_dict, metrics_order, ner_filter,
#                                                                             rank_check, sort_by_col, client_code,
#                                                                             metric_df)
#         if driver_high != driver_detail and not data1.empty:
#             pivot_table_biz, pivot_sort_col_biz = readout_utils._get_granular_table(data1, ner_dict1, metrics_order,
#                                                                                     ner_filter,
#                                                                                     rank_check, sort_by_col,
#                                                                                     client_code, metric_df)
#         if intention == 'source of change':
#             ner_filter_copy = ner_filter.copy()
#             ner_filter_copy['main_metric'] = ['sales']
#             str_q_overall_sales, _ = readout_utils._overall_metric(client_code, model_group_id, ner_filter_copy,
#                                                                    ner_res_dict, driver_tags, dim_cols, metric_df)
#             fixtext += str_q_overall_sales
#
#         if intention == 'contribution':
#             if biz_driver_check and (
#                     biz_driver_check == 'all' or (isinstance(biz_driver_check, list) and 'other' in biz_driver_check)):
#                 fixtext = "We only provide contribution by marketing drivers. "
#             if not core_dimension:
#                 fixtext_overall, _ = readout_utils._overall_metric(client_code, model_group_id, ner_filter,
#                                                                    ner_res_dict,
#                                                                    driver_tags, dim_cols, metric_df)
#                 fixtext += fixtext_overall
#         str_q = readout_utils._metrics_aka(str_q, ner_filter)
#         pivot_table = readout_utils._dynamic_top(dim_cols, pivot_sort_col, pivot_table, metric_df)
#         pivot_table = readout_utils._drop_single_dim(dim_cols, pivot_table)
#         pivot_table_biz = readout_utils._dynamic_top(dim_cols, pivot_sort_col_biz, pivot_table_biz, metric_df)
#         pivot_table_biz = readout_utils._drop_single_dim(dim_cols, pivot_table_biz)
#         pivot_table, pivot_table_biz = readout_utils._br_table_reformat(pivot_table, pivot_table_biz)
#
#     return str_q, pivot_table, pd.DataFrame(), "", pd.DataFrame(), fixtext, pivot_table_biz
#
#
# def _process_bi_detail_HILSP(client_code: str, model_group_id: int, df_dict: dict[str, pd.DataFrame],
#                              driver_tags: tuple,
#                              ner_filter: dict, ner_res_dict: dict, map_dict: dict, metric_df: pd.DataFrame, agg_view_df: pd.DataFrame, tactics_level: pd.DataFrame):
#     file_mapping_bi = {'tactics': 'ag_df', 'tactics_detail': 'mg_df', 'detail_tactics': 'm_df'}
#     file_type, data_type, driver_high, driver_detail = driver_tags
#
#     intention = ner_filter['intention'][0]
#     trend_check, rank_check = ner_res_dict.get('trend', 'no'), ner_res_dict.get('rank', 'na')
#     how_many = ner_res_dict.get('how_many', 'na')
#     rank_metric, sort_metric = map_dict[intention]['rankMetric'], map_dict[intention]['sortMetric']
#     metrics_order = ner_filter['metric']
#     benchmark_idx = readout_utils.load_benchmark_data(client_code, model_group_id)
#     planner_df = readout_utils.load_planner_data(client_code, model_group_id)
#     planner_response_curve, overview = readout_utils.load_diminishing_returns_data(client_code, model_group_id)
#     period_type = ner_filter.get('period_type', ['year'])[0]
#     fixtext = ""
#     core_dimension = ner_filter.get('core_dimension', {})
#     response_mod = ner_filter.get('response_modifier', [])
#     pivot_sort_col = ""
#     pivot_table, pivot_biz_table = pd.DataFrame(), pd.DataFrame()
#     planner_table = pd.DataFrame()
#     benchmark_data, benchmark_str = pd.DataFrame(), ""
#     product_media_check = False
#     dim_cols = readout_utils._get_dim_col_in_pivot_table(df_dict, ner_filter)
#     brand_focused_mkt, campaign = ner_filter.get('brand_focused_marketing', []), ner_filter.get('campaign', [])
#
#     if intention == 'planner':
#         m_df = df_dict['m_df']
#         fixtext, planner_table = (
#             readout_utils._planner_output(planner_df, m_df, metric_df, core_dimension, response_mod, planner_response_curve, overview, "Margin ROI")
#         )
#         str_q, fixtext_end, pivot_table_end, _, _, _ = readout_utils._planner_intention_res(
#             ner_filter, pd.DataFrame(), 'irrelevant', 'irrelevant', [])
#         fixtext += fixtext_end
#     else:
#         # read in and clean data
#         curr_df_name = file_mapping_bi[driver_detail]
#         curr_df = df_dict[curr_df_name]
#         if brand_focused_mkt:
#             curr_df = curr_df[(curr_df['brand_focused_marketing'].notna())&(curr_df['custom_aggregated']!='no')]
#         if curr_df.empty:
#             raise ValueError('No valid data.')
#
#         # check product media
#         if (curr_df['custom_aggregated'] != 'no').all():
#             product_media_check = True
#         # further clean
#         data1 = readout_utils._read_and_process_csv(curr_df, ner_filter)
#
#         # data1 = readout_utils._get_margin_roi_index(data1, 1.24, 1.31)
#
#         # provide overall roi/spend info in fixtext which is appended to LLM response
#         if intention in ['margin roi', 'performance', 'spending', 'cost per', 'activity'] and not core_dimension and not brand_focused_mkt and not campaign:
#             fixtext, _ = readout_utils._overall_metric_w_driver(client_code, model_group_id, ner_filter, ner_res_dict,
#                                                              driver_tags, metric_df)
#             fixtext = readout_utils.convert_display_format(fixtext)
#
#         concatenate = tactics_level[(tactics_level['dataType'] == 'bi') &
#                                     (tactics_level['tactics_level'] == driver_detail) &
#                                     (tactics_level['file_type'] == file_type)]['concatenate'].values[0]
#         if concatenate == 'yes':
#             data = readout_utils._concatenate_ag_mg_bi(data1)
#             driver_high = 'tactics_concat'
#         else:
#             data = data1
#
#         ner_dict = readout_utils._get_ner_key_value(data, driver_high, file_type, data_type, driver_detail, dim_cols)
#         data = readout_utils._calc_growth(period_type, ner_dict, data)
#         if data.empty:
#             raise ValueError('No valid data.')
#
#         growth_col, sort_by_col = readout_utils._get_growth_sort_col(trend_check, data)
#         data = readout_utils._sort_filt_data(data, ner_dict, sort_metric, sort_by_col, rank_check, metric_df)
#         data = readout_utils._ignore_growth_outlier(growth_col, data)
#
#         # top_data_benchmark = readout_utils._top_data(data, ner_dict, how_many)
#         # benchmark_str, benchmark_data = readout_utils._merge_benchmark(intention, None, core_dimension, None,
#         #                                                                ner_dict, data,
#         #                                                                top_data_benchmark, benchmark_idx)
#         benchmark_str, benchmark_data = readout_utils._benchmark_output(data, ner_dict, how_many, intention, core_dimension, benchmark_idx)
#
#         # agg_data, data = readout_utils._get_agg_data_hilsp(product_media_check, data)
#         agg_data, data, views = readout_utils._get_agg_data(data, agg_view_df)
#         pivot_table, pivot_sort_col = readout_utils._get_granular_table(data, ner_dict, metrics_order, ner_filter,
#                                                                         rank_check, sort_by_col, client_code, metric_df)
#         top_data = readout_utils._top_data(data, ner_dict, how_many)
#         str_q = readout_utils._get_readout(intention, growth_col, trend_check, rank_check, sort_metric, how_many,
#                                            ner_dict, ner_filter, data, top_data, metric_df)
#
#         if not agg_data.empty:
#             pivot_biz_table, _ = readout_utils._get_granular_table(agg_data, ner_dict, metrics_order, ner_filter,
#                                                                    rank_check, sort_by_col, client_code, metric_df)
#             top_data_agg = agg_data
#             str_q_agg = readout_utils._get_readout(intention, growth_col, trend_check, rank_check, sort_metric,
#                                                    how_many,
#                                                    ner_dict, ner_filter, agg_data, top_data_agg, metric_df)
#             str_q = str_q_agg + str_q
#
#         str_q = readout_utils._readout_reformat(str_q, agg_view_df)
#         pivot_table, pivot_biz_table = readout_utils._agg_table_format(pivot_table, pivot_biz_table, agg_view_df, views, concatenate)
#
#     return str_q, pivot_table, benchmark_data, benchmark_str, planner_table, fixtext, pivot_biz_table, pivot_sort_col, product_media_check
#
#
# def _process_bi_HILSP(client_code: str, model_group_id: int, df_dict: dict[str, pd.DataFrame],
#                       driver_tags: tuple,
#                       ner_filter: dict, ner_res_dict: dict, map_dict: dict, metric_df: pd.DataFrame):
#     core_dimension = ner_filter.get('core_dimension', {})
#     planner_table = pd.DataFrame()
#     dim_cols = readout_utils._get_dim_col_in_pivot_table(df_dict, ner_filter)
#     mg_m_tagging = ner_res_dict.get('campaign', 'irrelevant')
#     try:
#         str_q, pivot_table, benchmark_data, benchmark_str, planner_table, fixtext, pivot_biz_table, pivot_sort_col, product_media_check = _process_bi_detail_HILSP(
#             client_code, model_group_id, df_dict, driver_tags, ner_filter, ner_res_dict, map_dict, metric_df)
#     except Exception as e:
#         logger.warning(f"Failed to process initial BI details: {e}")
#         str_q, pivot_table, benchmark_data, benchmark_str, fixtext, pivot_biz_table, pivot_sort_col, product_media_check = "", pd.DataFrame(), pd.DataFrame(), "", "", pd.DataFrame(), "", False
#     if core_dimension or (mg_m_tagging != 'irrelevant' and not core_dimension):
#         search_level = readout_utils._check_key_or_value(core_dimension, ['activity_group', 'measure_group', 'measure'])
#         if (search_level['measure_group'] or search_level['activity_group']) or (mg_m_tagging != 'irrelevant' and not core_dimension):
#             driver_tags_m = 'm', 'bi', 'detail_tactics', 'detail_tactics'
#             try:
#                 str_q_m, pivot_table_m, benchmark_data_m, benchmark_str_m, planner_table, fixtext_m, pivot_biz_table_m, _, _ = _process_bi_detail_HILSP(
#                     client_code, model_group_id, df_dict, driver_tags_m, ner_filter, ner_res_dict, map_dict, metric_df)
#             except Exception as e:
#                 logger.warning(f"Failed to process measure level: {e}")
#                 str_q_m, pivot_table_m, benchmark_data_m, benchmark_str_m, fixtext_m, pivot_biz_table_m = "", pd.DataFrame(), pd.DataFrame(), "", "", pd.DataFrame()
#             if not pivot_table_m.empty:
#                 if pivot_biz_table.empty:
#                     pivot_biz_table = pivot_table
#                 pivot_table = pivot_table_m
#             if str_q_m:
#                 str_q = readout_utils._remove_noshow_readout(str_q)
#                 str_q = str_q + 'At Tactics level: ' + str_q_m
#             if benchmark_data.empty:
#                 benchmark_data = benchmark_data_m
#                 benchmark_str = benchmark_str_m
#         if str_q:
#             no_data_term = readout_utils.find_top_level_empty_keys(core_dimension)
#             str_q += ', '.join(no_data_term) + " not available in data." if no_data_term else ""
#     str_q = readout_utils._metrics_aka(str_q, ner_filter)
#     pivot_table = readout_utils._dynamic_top(dim_cols, pivot_sort_col, pivot_table, metric_df)
#     pivot_table = readout_utils._drop_single_dim(dim_cols, pivot_table)
#     pivot_biz_table = readout_utils._dynamic_top(dim_cols, pivot_sort_col, pivot_biz_table, metric_df)
#     pivot_biz_table = readout_utils._drop_single_dim(dim_cols, pivot_biz_table)
#
#     return str_q, pivot_table, benchmark_data, benchmark_str, planner_table, fixtext, pivot_biz_table
#
#
# def _process_br_HILSP(client_code: str, model_group_id: int, df_dict, driver_tags, ner_filter, ner_res_dict, map_dict,
#                       metric_df):
#     fixtext = ''
#     file_mapping_br = {'business_driver_detail': 'mg', 'tactics': 'm', 'business_driver': 'ag'}
#     file_type, data_type, driver_high, driver_detail = driver_tags
#     intention, biz_driver_check = ner_res_dict['intention'][0], ner_filter.get('business_driver', [])
#     trend_check, rank_check = ner_res_dict.get('trend', 'no'), ner_res_dict.get('rank', 'na')
#     trend_check = 'no' if intention == 'source of change' else trend_check
#     how_many = ner_res_dict.get('how_many', 'na')
#     period_type = ner_filter.get('period_type', ['year'])[0]
#     rank_metric, sort_metric = map_dict[intention]['rankMetric'], map_dict[intention]['sortMetric']
#     metrics_order = ner_filter['metric']
#     pivot_table, pivot_table_biz = pd.DataFrame(), pd.DataFrame()
#     pivot_sort_col, pivot_sort_col_biz = "", ""
#     core_dimension = ner_filter.get('core_dimension', {})
#     dim_cols = readout_utils._get_dim_col_in_pivot_table(df_dict, ner_filter)
#     brand_focused_mkt, campaign = ner_filter.get('brand_focused_marketing', []), ner_filter.get('campaign', [])
#     str_q = ""
#
#     if intention == 'sales':
#         str_q, pivot_table = readout_utils._overall_metric_w_driver(client_code, model_group_id, ner_filter,
#                                                                     ner_res_dict,
#                                                                     driver_tags, metric_df)
#     else:
#         curr_df_name_biz = file_mapping_br[driver_high] + '_df'
#         curr_df_biz = df_dict[curr_df_name_biz]
#         data1 = readout_utils._read_and_process_csv(curr_df_biz, ner_filter)
#
#         curr_df_name = file_type + '_df'
#         curr_df = df_dict[curr_df_name]
#
#         if curr_df.empty:
#             raise ValueError('No valid data.')
#         data2 = readout_utils._read_and_process_csv(curr_df, ner_filter)
#         ner_dict = readout_utils._get_ner_key_value(data2, driver_high, file_type, data_type, driver_detail, dim_cols)
#         ner_dict['driver'] = driver_detail
#         ner_dict1 = readout_utils._get_ner_key_value(data1, driver_high, file_mapping_br[driver_high], data_type,
#                                                      driver_high, dim_cols)
#         ner_dict1['driver'] = driver_high
#
#         if data2.empty:
#             raise ValueError('No valid data.')
#
#         data1 = readout_utils._calc_growth_br(intention, ner_res_dict, period_type, ner_dict1, data1)
#         data2 = readout_utils._calc_growth_br(intention, ner_res_dict, period_type, ner_dict, data2)
#         growth_col, sort_by_col = readout_utils._get_growth_sort_col(trend_check, data2)
#         data1 = readout_utils._ignore_growth_outlier(growth_col, data1)
#         data2 = readout_utils._ignore_growth_outlier(growth_col, data2)
#         data2 = readout_utils._exclude_system(intention, data2)
#         data2 = readout_utils._filter_contribution_driver(intention, data2)
#         if not data2.empty:
#             data2 = readout_utils._sort_filt_data(data2, ner_dict, sort_metric, sort_by_col, rank_check, metric_df)
#             top_data = readout_utils._top_data(data2, ner_dict, how_many)
#             str_q = readout_utils._get_readout(intention, growth_col, trend_check, rank_check, sort_metric, how_many,
#                                                ner_dict, ner_filter, data2, top_data, metric_df)
#             str_q = readout_utils._get_readout_br(intention, driver_detail, str_q)
#
#         top_data1 = pd.DataFrame()
#         if not data1.empty:
#             data1 = readout_utils._exclude_system(intention, data1)
#             data1 = readout_utils._filter_contribution_driver(intention, data1)
#             if not data1.empty:
#                 data1 = readout_utils._sort_filt_data(data1, ner_dict1, sort_metric, sort_by_col, rank_check, metric_df)
#                 top_data1 = readout_utils._top_data(data1, ner_dict1, how_many)
#
#         if driver_high != driver_detail and not data1.empty:
#             fixtext = readout_utils._get_readout(intention, growth_col, trend_check, rank_check, sort_metric, how_many,
#                                                  ner_dict1, ner_filter, data1, top_data1, metric_df)
#             fixtext = readout_utils._get_readout_br(intention, driver_high, fixtext)
#         if not str_q:
#             data2 = pd.DataFrame()
#         if fixtext:
#             str_q = fixtext + "More granularly: " + str_q if str_q else fixtext
#             fixtext = ""
#         if not data2.empty:
#             pivot_table, pivot_sort_col = readout_utils._get_granular_table(data2, ner_dict, metrics_order, ner_filter,
#                                                                             rank_check, sort_by_col, client_code,
#                                                                             metric_df)
#         if driver_high != driver_detail and not data1.empty:
#             pivot_table_biz, pivot_sort_col_biz = readout_utils._get_granular_table(data1, ner_dict1, metrics_order,
#                                                                                     ner_filter,
#                                                                                     rank_check, sort_by_col,
#                                                                                     client_code, metric_df)
#         if intention == 'source of change':
#             ner_filter_copy = ner_filter.copy()
#             ner_filter_copy['main_metric'] = ['sales']
#             str_q_overall_sales, _ = readout_utils._overall_metric_w_driver(client_code, model_group_id,
#                                                                             ner_filter_copy,
#                                                                             ner_res_dict, driver_tags,
#                                                                             metric_df)
#             fixtext += str_q_overall_sales
#
#         if intention == 'contribution':
#             if biz_driver_check and (
#                     biz_driver_check == 'all' or (isinstance(biz_driver_check, list) and 'base' in biz_driver_check)):
#                 fixtext = "We only provide contribution by marketing drivers. "
#             if not core_dimension and not brand_focused_mkt and not campaign:
#                 fixtext_overall, _ = readout_utils._overall_metric_w_driver(client_code, model_group_id, ner_filter,
#                                                                             ner_res_dict,
#                                                                             driver_tags, metric_df)
#                 fixtext += fixtext_overall
#
#         if str_q:
#             no_data_term = readout_utils.find_top_level_empty_keys(core_dimension)
#             str_q += ', '.join(no_data_term) + " not available in data." if no_data_term else ""
#         str_q = readout_utils._metrics_aka(str_q, ner_filter)
#         pivot_table = readout_utils._dynamic_top(dim_cols, pivot_sort_col, pivot_table, metric_df)
#         pivot_table = readout_utils._drop_single_dim(dim_cols, pivot_table)
#         pivot_table_biz = readout_utils._dynamic_top(dim_cols, pivot_sort_col_biz, pivot_table_biz, metric_df)
#         pivot_table_biz = readout_utils._drop_single_dim(dim_cols, pivot_table_biz)
#         pivot_table, pivot_table_biz = readout_utils._br_table_reformat(pivot_table, pivot_table_biz)
#
#     return str_q, pivot_table, pd.DataFrame(), "", pd.DataFrame(), fixtext, pivot_table_biz


def _process_bi_detail_share(client_code: str, model_group_id: int, df_dict: dict[str, pd.DataFrame],
                              driver_tags: tuple,
                              ner_filter: dict, ner_res_dict: dict, map_dict: dict, metric_df: pd.DataFrame,
                              agg_view_df: pd.DataFrame, tactics_level: pd.DataFrame):
    file_type, data_type, driver_high, driver_detail = driver_tags
    intention = ner_filter['intention'][0]
    trend_check, rank_check = ner_res_dict.get('trend', 'no'), ner_res_dict.get('rank', 'na')
    how_many = ner_filter.get('how_many', 'na')
    rank_metric, sort_metric = map_dict[intention]['rankMetric'], map_dict[intention]['sortMetric']
    metrics_order = ner_filter['metric']
    benchmark_idx = readout_utils.load_benchmark_data(client_code, model_group_id)
    planner_df = readout_utils.load_planner_data(client_code, model_group_id)
    planner_response_curve, overview = readout_utils.load_diminishing_returns_data(client_code, model_group_id)
    planner_filter = readout_utils.load_planner_filter(client_code, model_group_id)
    planner_principle = readout_utils.load_planner_principle(client_code, model_group_id)
    genome_principle = readout_utils.load_genome_principle(client_code, model_group_id)
    period_type = (ner_filter.get('period_type') or ['year'])[0]
    fixtext = ""
    core_dimension = ner_filter.get('core_dimension', {})
    response_mod = ner_res_dict.get('response_modifier', [])
    pivot_sort_col = ""
    pivot_table, pivot_biz_table = pd.DataFrame(), pd.DataFrame()
    planner_table = pd.DataFrame()
    spend_share_principle = pd.DataFrame()
    benchmark_data, benchmark_str = pd.DataFrame(), ""
    product_media_check = False
    dim_cols = readout_utils._get_dim_col_in_pivot_table(df_dict, ner_filter)
    level, tagging = readout_utils._get_level_tagging(ner_filter, client_code)
    custom_tagging_condition = readout_utils._custom_tagging_condition(ner_filter, tagging)
    overall_ag_only = False
    overall_table = pd.DataFrame()
    principle_pretext = ''

    # for colgate:
    media_channel_check = ner_res_dict.get('media_channel', 'irrelevant')
    media_channel_ls = ner_filter.get('media_channel', [])
    product_halo_check = ner_filter.get('product_halo', 'irrelevant')
    core_dimension_composite = ner_filter.get('core_dimension_composite', {})
    diag_tagging = ner_filter.get('diag_tagging', {})

    if intention == 'planner':
        if client_code == 'COLGUS':
            str_q, fixtext, pivot_table, benchmark_data, benchmark_str, planner_data = readout_utils._planner_intention_res(
                ner_filter, planner_df, product_halo_check, media_channel_check, media_channel_ls)
        else:
            if 'activity_group' in planner_df.columns:
                m_df = pd.concat([df_dict['ag_df'], df_dict['mg_df'], df_dict['m_df']])
            else:
                m_df = df_dict['m_df']
            ### filter kpi
            planner_df, planner_filter = readout_utils.planner_kpi_filter(planner_filter, planner_df, ner_res_dict, ner_filter)
            ### filter scenario
            planner_data = readout_utils._planner_filter(planner_filter, dim_cols, ner_filter, planner_df)
            planner_filter = planner_filter[planner_filter['Scenario ID'].isin(planner_data['Scenario ID'].unique().tolist())]
            if planner_data.empty and not planner_df.empty:
                planner_data = planner_df

            fixtext, planner_table, reduce_budget, call_llm, head, spend_share_principle = (
                readout_utils._planner_output(planner_data, m_df, metric_df, planner_principle, genome_principle,
                                              core_dimension, response_mod, planner_response_curve,
                                              overview, planner_filter))
            str_q, fixtext_end, pivot_table_end, _, _, _ = readout_utils._planner_intention_res(
                ner_filter, pd.DataFrame(), 'irrelevant', 'irrelevant', [])
            genome_str = "" if fixtext == "" else readout_utils._genome_fixtext(genome_principle, planner_data, intention,
                                                                                reduce_budget)
            if client_code == 'LINKEDIN' and fixtext != '':
                call_llm = True
                fixtext = 'LINKEDIN no table readout. ' + fixtext
            if call_llm:
                str_q = fixtext
                fixtext = "\n\n".join(s for s in [head, genome_str] if s and s.strip())
            else:
                fixtext = head + "\n\n" + fixtext
                # fixtext = "\n\n".join(s for s in [fixtext, genome_str, fixtext_end] if s and s.strip())
                fixtext = "\n\n".join(s for s in [fixtext, genome_str] if s and s.strip())
    else:
        # read in and clean data
        curr_df_name = file_type + "_df"
        curr_df = df_dict[curr_df_name]
        curr_df = readout_utils._halo_tagging_filter(curr_df, ner_filter, core_dimension_composite)
        if curr_df.empty:
            raise ValueError('No valid data.')
        dim_cols = readout_utils._get_dim_col_in_pivot_table_unique(df_dict, ner_filter, client_code)
        # check product media
        if ("custom_aggregated" in curr_df.columns
                and (~curr_df["custom_aggregated"].isin(["no", "diagnostic"])).all()):
            product_media_check = True
        if ("custom_aggregated" in curr_df.columns
                and (curr_df["custom_aggregated"] == 'diagnostic').all()):
            diagnostic_check = True
        else:
            diagnostic_check = False
        if client_code == 'COLGUS':
            curr_df = readout_utils._filter_paid_search(intention, curr_df)
            if model_group_id == 3041:
                curr_df = readout_utils._filter_pillar(product_halo_check, core_dimension, curr_df)
                curr_df = readout_utils._filter_special_media(product_halo_check, curr_df)

        if not (client_code == 'COLGUS' and model_group_id != 3041):
            data1 = readout_utils._read_and_process_csv(curr_df, ner_filter)
        else:
            data1, dim_cols = readout_utils._read_and_process_csv_other_category(curr_df, ner_filter,dim_cols)
        data1, dim_cols = readout_utils._level_to_drop(level, dim_cols, data1)
        if 'core_dimension_term' in data1.columns:
            dim_cols = dim_cols + ['core_dimension_term']
        dim_dict = {col: data1[col].unique().tolist() for col in dim_cols}
        uniq_dim_ner = 1
        uniq_dim_data = data1[dim_cols].drop_duplicates().shape[0] if not data1.empty else 1
        for d in dim_cols:
            d_val = ner_filter.get(d, [])
            if len(d_val) > 1:
                uniq_dim_ner *= len(d_val)
        uniq_dim = max(uniq_dim_ner, uniq_dim_data)
        # provide overall roi/spend info in fixtext which is appended to LLM response
        if not core_dimension and custom_tagging_condition and not core_dimension_composite and not diag_tagging:
            fixtext, overall_table = readout_utils._overall_metric_w_driver(client_code, model_group_id, ner_filter, ner_res_dict,
                                                                driver_tags, metric_df, dim_dict)
            fixtext = readout_utils.convert_display_format(fixtext)
            if uniq_dim>=4:
                fixtext = ''
            overall_ag_only = True
            # if intention not in ['margin roi', 'performance']:
            #     principle_str = readout_utils._genome_fixtext(genome_principle, data1, intention)
            #     principle_pretext += "**Additional Insights from ROI Genome.**\n\n" + principle_str if principle_str else ''
            if 'custom_aggregated' in data1.columns and not product_media_check:
                data1 = data1[data1['custom_aggregated'].isin(['no', 'diagnostic'])]
            if client_code == 'COLGUS' and model_group_id == 3041:
                data1 = data1[~data1['media_channel'].isin(['special media', 'non-media'])]
        principle_pretext = readout_utils.genome_principle_test_new(client_code, model_group_id, data1, dim_cols, intention)

        concatenate = tactics_level[(tactics_level['dataType'] == 'bi') &
                                    (tactics_level['file_type'] == file_type)]['concatenate'].values[0]
        if concatenate == 'yes' and not diagnostic_check:
            concat_col = driver_detail
            data = readout_utils._concatenate_ag_mg_bi(data1, concat_col)
            driver_high = 'media_channel' if client_code == 'COLGUS' else 'tactics_concat'
        else:
            data = data1

        ner_dict = readout_utils._get_ner_key_value(data, driver_high, file_type, data_type, driver_detail, dim_cols)
        if data.empty:
            raise ValueError('No valid data.')
        if intention == 'spending':
            summary = readout_utils._bi_spend_range(client_code, model_group_id, planner_principle, genome_principle,
                                                    data, dim_cols)
            principle_pretext += "**Additional Insights from ROI Genome.**\n\n" if summary and not principle_pretext else '\n\n'
            principle_pretext += summary if summary else ''
        time_message = readout_utils._validate_time_range(ner_res_dict, ner_filter, data)
        if time_message:
            fixtext = time_message + '\n\n' + fixtext
        diag_agg_data, diag_agg = readout_utils._get_diag_agg(data, file_type, diag_tagging)
        agg_data, data, views, ner_dict = readout_utils._get_agg_data(data, agg_view_df, ner_dict)
        if diag_agg:
            agg_data = diag_agg_data
        data = readout_utils._calc_growth(period_type, ner_dict, data)
        growth_col, sort_by_col = readout_utils._get_growth_sort_col(trend_check, data)

        if not data.empty:
            data = readout_utils._sort_filt_data(data, ner_dict, sort_metric, sort_by_col, rank_check, metric_df)
            data = readout_utils._ignore_growth_outlier(growth_col, data)

            top_data = readout_utils._top_data(data, ner_dict, how_many)
            top_data = readout_utils._filter_top_media_channel(driver_high, how_many, ner_filter, ner_dict, sort_by_col,
                                                               top_data)
        else:
            top_data = data

        if client_code == 'COLGUS':
            benchmark_str, benchmark_data = readout_utils._merge_benchmark(intention, media_channel_check,
                                                                           core_dimension,
                                                                           core_dimension_composite, ner_dict, data,
                                                                           top_data, benchmark_idx)
        else:
            benchmark_str, benchmark_data = readout_utils._benchmark_output(data, ner_dict, how_many, intention, core_dimension, benchmark_idx)

        str_q = readout_utils._get_readout(intention, growth_col, trend_check, rank_check, sort_metric, how_many,
                                           ner_dict, ner_filter, data, top_data, metric_df)
        str_q, data, ner_dict = readout_utils.readout_adjust(client_code, model_group_id, data, str_q, ner_res_dict, ner_dict, how_many, driver_high,
                       ner_filter, sort_by_col, intention, growth_col, trend_check, rank_check, sort_metric, metric_df)
        pivot_table, pivot_sort_col = readout_utils._get_granular_table(data, ner_dict, metrics_order, ner_filter,
                                                                        rank_check, sort_by_col, metric_df,
                                                                        client_code, model_group_id)

        if not agg_data.empty:
            agg_data = readout_utils._calc_growth(period_type, ner_dict, agg_data)
            agg_data = readout_utils._sort_filt_data(agg_data, ner_dict, sort_metric, sort_by_col, rank_check,
                                                     metric_df)
            agg_data = readout_utils._ignore_growth_outlier(growth_col, agg_data)
            pivot_biz_table, _ = readout_utils._get_granular_table(agg_data, ner_dict, metrics_order, ner_filter,
                                                                   rank_check, sort_by_col, metric_df,
                                                                   client_code, model_group_id)
            top_data_agg = agg_data
            if client_code == 'COLGUS':
                ner_dict_pillar = ner_dict.copy()
                ner_dict_pillar['driver'] = 'tactics_detail'
                # special
                str_q_agg = f"Overall {intention} of each product media is as follows:" + readout_utils._get_granular_level_pivot(
                    agg_data, ner_dict_pillar, ner_filter, trend_check, metric_df)
            else:
                str_q_agg = readout_utils._get_readout(intention, growth_col, trend_check, rank_check, sort_metric,
                                                       how_many,
                                                       ner_dict, ner_filter, agg_data, top_data_agg, metric_df)
            str_q = str_q_agg + str_q

        str_q = readout_utils._readout_reformat(str_q, agg_view_df)
        pivot_table = readout_utils._dynamic_top(ner_dict['dimension'], pivot_sort_col, pivot_table, metric_df)
        pivot_table = readout_utils._drop_single_dim(ner_dict['dimension'], pivot_table)
        pivot_biz_table = readout_utils._dynamic_top(ner_dict['dimension'], pivot_sort_col, pivot_biz_table, metric_df)
        pivot_biz_table = readout_utils._drop_single_dim(ner_dict['dimension'], pivot_biz_table)

        if client_code == 'COLGUS' and model_group_id == 3041:
            pivot_table, pivot_biz_table = readout_utils._agg_table_format(pivot_table, pivot_biz_table,
                                                                           agg_view_df, views,"", 'Product')
            pivot_table = readout_utils._rename_unwanted_tactics(pivot_table)
            pivot_biz_table = readout_utils._rename_unwanted_tactics(pivot_biz_table)
            pivot_biz_table = readout_utils._pillar_pivot_table_rename(core_dimension_composite, core_dimension,
                                                                       pivot_biz_table)
        elif client_code == 'COLGUS' and model_group_id > 3041:
            pivot_table = readout_utils._rename_product_media_other_category(ner_filter, dim_cols[0], pivot_table)
        else:
            pivot_table, pivot_biz_table = readout_utils._agg_table_format(pivot_table, pivot_biz_table, agg_view_df,
                                                                           views,concatenate)
        if diag_agg:
            pivot_biz_table = pivot_biz_table.drop(columns=['Marketing Focus'], errors='ignore')

    return str_q, pivot_table, benchmark_data, benchmark_str, planner_table, spend_share_principle, fixtext, pivot_biz_table, pivot_sort_col, product_media_check, overall_ag_only, overall_table, principle_pretext


def _process_br_detail_share(client_code: str, model_group_id: int, df_dict, driver_tags, ner_filter, ner_res_dict, map_dict,
                      metric_df):
    fixtext = ''
    file_type, data_type, driver_high, driver_detail = driver_tags
    intention, biz_driver_check = ner_filter['intention'][0], ner_res_dict.get('business_driver', 'irrelevant')
    trend_check, rank_check = ner_res_dict.get('trend', 'no'), ner_res_dict.get('rank', 'na')
    trend_check = 'no' if intention == 'source of change' else trend_check
    how_many = ner_res_dict.get('how_many', 'na')
    period_type = (ner_filter.get('period_type') or ['year'])[0]
    rank_metric, sort_metric = map_dict[intention]['rankMetric'], map_dict[intention]['sortMetric']
    metrics_order = ner_filter['metric']
    pivot_table, pivot_table_biz = pd.DataFrame(), pd.DataFrame()
    pivot_sort_col = ""
    core_dimension = ner_filter.get('core_dimension', {})
    dim_cols = readout_utils._get_dim_col_in_pivot_table(df_dict, ner_filter)
    level, tagging = readout_utils._get_level_tagging(ner_filter, client_code)
    custom_tagging_condition = readout_utils._custom_tagging_condition(ner_filter, tagging)
    str_q = ""
    overall_table = pd.DataFrame()

    pivot_biz_table = pd.DataFrame()
    if intention == 'sales':
        str_q, pivot_table = readout_utils._overall_metric_w_driver(client_code, model_group_id, ner_filter,
                                                                    ner_res_dict,
                                                                    driver_tags, metric_df, None)
        time_message = readout_utils._validate_time_range(ner_res_dict, ner_filter)
        if time_message:
            fixtext = time_message + '\n\n' + fixtext
    else:
        curr_df_name = file_type + '_df'
        curr_df = df_dict[curr_df_name]
        curr_df = readout_utils._br_custom_agg_filter(core_dimension, curr_df, ner_filter)
        if curr_df.empty:
            raise ValueError('No valid data.')
        dim_cols = readout_utils._get_dim_col_in_pivot_table_unique(df_dict, ner_filter, client_code)
        if not (client_code == 'COLGUS' and model_group_id != 3041):
            data = readout_utils._read_and_process_csv(curr_df, ner_filter)
        else:
            data, dim_cols = readout_utils._read_and_process_csv_other_category(curr_df, ner_filter,dim_cols)
        data, dim_cols = readout_utils._level_to_drop(level, dim_cols, data)
        if 'core_dimension_term' in data.columns:
            dim_cols = dim_cols + ['core_dimension_term']
        dim_dict = {col: data[col].unique().tolist() for col in dim_cols}
        data, driver_detail = readout_utils._concatenate_br(data, driver_detail)
        ner_dict = readout_utils._get_ner_key_value(data, driver_high, file_type, data_type, driver_detail,
                                                    dim_cols)
        ner_dict['driver'] = driver_detail

        if data.empty:
            raise ValueError('No valid data.')

        time_message = readout_utils._validate_time_range(ner_res_dict, ner_filter, data)
        if time_message:
            fixtext = time_message + '\n\n' + fixtext
        data = readout_utils._calc_growth_br(intention, ner_res_dict, period_type, ner_dict, data)
        growth_col, sort_by_col = readout_utils._get_growth_sort_col(trend_check, data)
        data = readout_utils._ignore_growth_outlier(growth_col, data)
        data = readout_utils._exclude_system(intention, data)
        data, trigger_pretext = readout_utils._filter_contribution_driver(intention, data)
        if 'custom_aggregated' in data.columns:
            agg_data = data[data['custom_aggregated']=='yes']
            data = data[data['custom_aggregated']!='yes']
        else:
            agg_data = pd.DataFrame()
        if not data.empty:
            data = readout_utils._sort_filt_data(data, ner_dict, sort_metric, sort_by_col, rank_check,
                                                 metric_df)
            top_data = readout_utils._top_data(data, ner_dict, how_many)
            str_q = readout_utils._get_readout(intention, growth_col, trend_check, rank_check, sort_metric,
                                               how_many,
                                               ner_dict, ner_filter, data, top_data, metric_df)
            # str_q = readout_utils._get_readout_br(intention, driver_detail, str_q)
        # if intention == 'contribution' and (((isinstance(biz_driver_check, str) and biz_driver_check in ['all', 'irrelevant']) and not core_dimension) or data.empty):
        if intention == 'contribution' and ((trigger_pretext and not core_dimension) or data.empty or any(k in ner_filter.get('business_driver_captured', {}) for k in ['base', 'other'])):
            fixtext += "We only provide contribution by marketing drivers. "
            trigger_pretext = True
        if not str_q:
            data = pd.DataFrame()
        ner_dict_cp = ner_dict.copy()
        str_q, data, ner_dict = readout_utils.readout_adjust(client_code, model_group_id, data, str_q, ner_res_dict, ner_dict, how_many,
                                             driver_high, ner_filter, sort_by_col, intention, growth_col, trend_check, rank_check,
                                             sort_metric, metric_df)
        if not agg_data.empty:
            ner_dict_cp['driver'] = 'tactics_detail'
            ner_dict_cp['driver_detail'] = 'tactics_detail'
            agg_data = readout_utils._sort_filt_data(agg_data, ner_dict_cp, sort_metric, sort_by_col, rank_check,
                                                     metric_df)
            top_data_agg = agg_data
            str_q_agg = readout_utils._get_readout(intention, growth_col, trend_check, rank_check, sort_metric,
                                               how_many,
                                               ner_dict_cp, ner_filter, data, top_data_agg, metric_df)
            str_q_agg, agg_data, ner_dict_cp = readout_utils.readout_adjust(client_code, model_group_id, agg_data, str_q_agg, ner_res_dict,
                                                                 ner_dict_cp, how_many,
                                                                 driver_high, ner_filter, sort_by_col, intention,
                                                                 growth_col, trend_check, rank_check,
                                                                 sort_metric, metric_df)
            str_q = str_q_agg + str_q
            pivot_biz_table, _ = readout_utils._get_granular_table(agg_data, ner_dict_cp, metrics_order, ner_filter,
                                                                   rank_check, sort_by_col, metric_df,
                                                                   client_code, model_group_id)
        if trigger_pretext:
            str_q = 'LINKEDIN no table readout. ' + str_q
        if not data.empty:
            pivot_table, pivot_sort_col = readout_utils._get_granular_table(data, ner_dict, metrics_order,
                                                                            ner_filter, rank_check, sort_by_col,
                                                                            metric_df, client_code, model_group_id)
        uniq_dim_ner = 1
        uniq_dim_data = data[ner_dict['dimension']].drop_duplicates().shape[0] if not data.empty else 1
        for d in ner_dict['dimension']:
            d_val = ner_filter.get(d, [])
            if len(d_val) >1:
                uniq_dim_ner *= len(d_val)
        uniq_dim = max(uniq_dim_ner, uniq_dim_data)
        if intention == 'source of change':
            ner_filter_copy = ner_filter.copy()
            sales_metric = metric_df['metric'][metric_df['metric'].str.contains('sales', case=False, na=False)].unique().tolist()
            ner_filter_copy['main_metric'] = sales_metric
            str_q_overall_sales, overall_table = readout_utils._overall_metric_w_driver(client_code, model_group_id,
                                                                            ner_filter_copy,
                                                                            ner_res_dict, driver_tags,
                                                                            metric_df, dim_dict)
            str_q_overall_sales = readout_utils.convert_display_format(str_q_overall_sales)
            # fixtext += str_q_overall_sales
            if uniq_dim < 4:
                fixtext += str_q_overall_sales

        if intention == 'contribution':
            if not core_dimension and custom_tagging_condition:
                fixtext_overall, overall_table = readout_utils._overall_metric_w_driver(client_code, model_group_id,
                                                                            ner_filter, ner_res_dict,
                                                                            driver_tags, metric_df, dim_dict)
                # fixtext += fixtext_overall
                if uniq_dim < 4:
                    fixtext += fixtext_overall
        pivot_table = readout_utils._dynamic_top(ner_dict['dimension'], pivot_sort_col, pivot_table, metric_df, rank_check)
        pivot_table = readout_utils._drop_single_dim(ner_dict['dimension'], pivot_table)
        pivot_biz_table = readout_utils._dynamic_top(ner_dict['dimension'], pivot_sort_col, pivot_biz_table, metric_df,
                                                 rank_check)
        pivot_biz_table = readout_utils._drop_single_dim(ner_dict['dimension'], pivot_biz_table)

    return str_q, pivot_table, pd.DataFrame(), "", pd.DataFrame(), fixtext, pivot_biz_table, pivot_sort_col, overall_table


class Postprocess:

    def __init__(self,
                 client_code, model_group_id,
                 df_dict, driver_tags,
                 ner_filter_dict, ner_res_dict, map_dict, metric_df):

        self.client_code = client_code
        self.model_group_id = model_group_id
        self.function_bi_detail = _get_function_name_bi_detail(client_code, model_group_id)
        self.function_br_detail = _get_function_name_br_detail(client_code, model_group_id)

        # basic info from ner
        self.ner_filter_dict = ner_filter_dict
        self.ner_res_dict = ner_res_dict
        self.intention = self.ner_res_dict.get('intention', ['margin roi'])[0]
        self.business_driver = self.ner_res_dict.get('business_driver', 'irrelevant')
        self.rank_check = self.ner_res_dict.get('rank', 'na')
        self.main_metric = self.ner_filter_dict.get('main_metric', [])
        self.data_type = self.ner_filter_dict['data']
        self.core_dimension = self.ner_filter_dict.get('core_dimension', {})
        self.time = self.ner_filter_dict.get('time', [])
        self.search_level = readout_utils._check_key_or_value(self.core_dimension,
                                                              ['activity_group', 'measure_group', 'measure'])

        self.df_dict = df_dict

        self.dim_cols = readout_utils._get_dim_col_in_pivot_table(self.df_dict, self.ner_filter_dict)

        # configuration files
        self.map_dict = map_dict
        self.tactics_level = readout_utils.load_tactics_level(client_code, model_group_id)
        self.driver_tags, self.driver_tags_br = readout_utils._generate_driver_tags(self.tactics_level, df_dict)
        self.df_lookup_bi = readout_utils.load_result_lookup_bi(client_code, model_group_id)
        self.df_lookup_br = readout_utils.load_result_lookup_br(client_code, model_group_id)
        self.granular_level = any(
            tag in ('tactics_detail', 'detail_tactics')
            for value in self.driver_tags_br.values()
            for tag in value)
        granular_flag = 'Y' if self.granular_level else 'N'
        if 'granular_level' in self.df_lookup_br.columns:
            self.df_lookup_br = self.df_lookup_br[self.df_lookup_br['granular_level'] == granular_flag]
        self.df_lookup_bi['search_key'] = self.df_lookup_bi[
            ['agg_ag', 'detail_ag', 'agg_mg', 'detail_mg', 'agg_m', 'detail_m']].astype(str).agg('~'.join, axis=1)
        if 'agg_mg' in self.df_lookup_br.columns:
            self.df_lookup_br['search_key'] = self.df_lookup_br[['detail_ag', 'agg_mg', 'detail_mg', 'detail_m']].astype(str).agg(
                '~'.join, axis=1)
        else:
            self.df_lookup_br['search_key'] = self.df_lookup_br[['detail_ag', 'detail_mg', 'detail_m']].astype(str).agg(
            '~'.join, axis=1)
        self.metric_df = metric_df
        self.agg_view_df = readout_utils.load_agg_view_logic(client_code, model_group_id)

        # results
        self.BI_readout_dict = {'ag': {}, 'mg': {}, 'm': {}}
        self.BR_readout_dict = {'ag': {}, 'mg': {}, 'm': {}}

        self.result_combo_match = {}
        self.readout = ""
        self.detail_view = pd.DataFrame()
        self.benchmark_data = pd.DataFrame()
        self.planner_data = pd.DataFrame()
        self.spend_share_principle = pd.DataFrame()
        self.agg_view = pd.DataFrame()
        self.benchmark_str = ""
        self.pretext = ""
        self.pivot_sort_col = ""
        self.pretext_table = pd.DataFrame()
        self.pretext_table_trend = pd.DataFrame()
        self.principle_pretext = ""

    def _process_bi_share(self, client_code: str, model_group_id: int, df_dict: dict[str, pd.DataFrame],
                          driver_tags: tuple, ner_filter_dict: dict, ner_res_dict: dict,
                          map_dict: dict, metric_df: pd.DataFrame):
        for level in ['ag', 'mg', 'm']:
            driver_tag = self.driver_tags[level]
            try:
                (readout, detail_df, benchmark_data, benchmark_str, planner_table,
                 spend_share_principle, pretext, agg_df, pivot_sort_col, all_custom_agg,
                 overall_ag_only, overall_table, principle_pretext) = self.function_bi_detail(
                    client_code, model_group_id, df_dict, driver_tag, ner_filter_dict, ner_res_dict, map_dict,
                    metric_df, self.agg_view_df, self.tactics_level)
                self.BI_readout_dict[level] = dict(readout=readout, detail=detail_df, benchmark_data=benchmark_data,
                                                   benchmark_str=benchmark_str,
                                                   planner_table=planner_table, spend_share_principle = spend_share_principle,
                                                   pretext=pretext, agg=agg_df,
                                                   pivot_sort_col=pivot_sort_col, all_custom_agg=all_custom_agg,
                                                   overall_ag_only=overall_ag_only, overall_table=overall_table,
                                                   principle_pretext=principle_pretext)
                planner_readout = readout if readout and not planner_table.empty else ''
            except Exception as e:
                logger.warning(f"Failed to process BI {level} results: {e}")
                self.BI_readout_dict[level] = dict(readout="", detail=pd.DataFrame(), benchmark_data=pd.DataFrame(),
                                                   benchmark_str="", planner_table=pd.DataFrame(), spend_share_principle = pd.DataFrame(),
                                                   pretext="", agg=pd.DataFrame(), pivot_sort_col="", all_custom_agg="",
                                                   overall_ag_only=False, overall_table=pd.DataFrame(), principle_pretext='')
                planner_readout = ''

        df_ls = [self.BI_readout_dict['ag']['agg'], self.BI_readout_dict['ag']['detail'],
                 self.BI_readout_dict['mg']['agg'], self.BI_readout_dict['mg']['detail'],
                 self.BI_readout_dict['m']['agg'], self.BI_readout_dict['m']['detail']]
        df_states = ['N' if df.empty else 'Y' for df in df_ls]
        df_states.append('Y' if self.BI_readout_dict['mg']['all_custom_agg'] else 'N')

        match = self.df_lookup_bi[self.df_lookup_bi['search_key'] == '~'.join(df_states[:6])]
        if len(match) > 1:
            match = match[(match['all_custom_agg'] == df_states[6])]
        if not match.empty:
            self.result_combo_match = match.reset_index().iloc[0, :].fillna("").to_dict()
            readout = self.result_combo_match['readout']
            self.readout = readout_utils._get_readout_com(readout, self.BI_readout_dict, self.result_combo_match,
                                                          self.data_type)

            self.detail_view = self.BI_readout_dict.get(self.result_combo_match['detail_view'].split('_')[-1], {}).get(
                self.result_combo_match['detail_view'].split('_')[0], pd.DataFrame())
            self.benchmark_data = self.BI_readout_dict.get(self.result_combo_match['benchmark_data'].split('_')[-1],
                                                           {}).get('benchmark_data', pd.DataFrame())
            self.planner_data = self.BI_readout_dict.get(self.result_combo_match['planner_data'].split('_')[-1],
                                                         {}).get('planner_table', pd.DataFrame())
            self.spend_share_principle = self.BI_readout_dict.get(self.result_combo_match['planner_data'].split('_')[-1],
                                                         {}).get('spend_share_principle', pd.DataFrame())
            self.agg_view = self.BI_readout_dict.get(self.result_combo_match['agg_view'].split('_')[-1], {}).get(
                self.result_combo_match['agg_view'].split('_')[0], pd.DataFrame())
            self.benchmark_str = self.BI_readout_dict.get(self.result_combo_match['benchmark_str'].split('_')[-1],
                                                          {}).get('benchmark_str', "")
            self.pretext = self.BI_readout_dict.get(self.result_combo_match['pretext'].split('_')[-1], {}).get(
                'pretext', "")
            self.principle_pretext = self.BI_readout_dict.get(self.result_combo_match['pretext'].split('_')[-1], {}).get(
                'principle_pretext', "")
            self.pivot_sort_col = self.BI_readout_dict.get(self.result_combo_match['pivot_sort_col'].split('_')[-1],
                                                           {}).get('pivot_sort_col', "")
            self.pretext_table = self.BI_readout_dict['ag']['overall_table']
            overall_view_value = self.result_combo_match.get("overall_view")
            if overall_view_value:
                prefix, suffix = overall_view_value.split("_")[0], overall_view_value.split("_")[-1]
                self.overall_view = self.BI_readout_dict.get(suffix, {}).get(prefix, pd.DataFrame())
            else:
                self.overall_view = pd.DataFrame()
            if self.BI_readout_dict['ag']['overall_ag_only'] and self.result_combo_match['readout_ag']=='Y':
                self.readout = readout_utils._get_readout_com('readout_ag', self.BI_readout_dict, self.result_combo_match,
                                                          self.data_type)
                self.agg_view = pd.DataFrame()
                self.overall_view = pd.DataFrame()
                self.detail_view = self.BI_readout_dict['ag'].get('detail', pd.DataFrame())
                self.benchmark_data = self.BI_readout_dict['ag'].get('benchmark_data', pd.DataFrame())
                self.benchmark_str = self.BI_readout_dict['ag'].get('benchmark_str', "")

        self.readout = readout_utils._add_na_term(self.core_dimension, self.readout)
        self.readout = readout_utils._metrics_aka(self.readout, self.ner_filter_dict)
        self.detail_view, self.agg_view = readout_utils.dedup_post(self.detail_view, self.agg_view, self.df_dict,
                                                     self.result_combo_match['detail_view'].split('_')[-1],
                                                     self.result_combo_match['agg_view'].split('_')[-1],
                                                                   self.client_code, self.main_metric)
        if not self.overall_view.empty:
            self.detail_view, self.overall_view = readout_utils.dedup_post(self.detail_view, self.overall_view,
                                                                           self.df_dict,
                                                                       self.result_combo_match['detail_view'].split(
                                                                           '_')[-1],
                                                                       self.result_combo_match['overall_view'].split(
                                                                           '_')[-1], self.client_code, self.main_metric)
            self.agg_view, self.overall_view = readout_utils.dedup_post(self.agg_view, self.overall_view, self.df_dict,
                                                                       self.result_combo_match['agg_view'].split(
                                                                           '_')[-1],
                                                                       self.result_combo_match['overall_view'].split(
                                                                           '_')[-1], self.client_code, self.main_metric)

        if 'Core_Dimension_Term' in self.detail_view.columns:
            self.dim_cols = self.dim_cols + ['Core_Dimension_Term']
        self.detail_view = readout_utils._get_metrics_rename(self.pivot_sort_col, self.metric_df, self.detail_view)
        self.agg_view = readout_utils._get_metrics_rename(self.pivot_sort_col, self.metric_df, self.agg_view)
        self.overall_view = readout_utils._get_metrics_rename(self.pivot_sort_col, self.metric_df, self.overall_view)
        self.detail_view = readout_utils.apply_level_mapping_display(self.client_code, self.model_group_id,
                                                                     self.detail_view)
        self.agg_view = readout_utils.apply_level_mapping_display(self.client_code, self.model_group_id, self.agg_view)
        self.overall_view = readout_utils.apply_level_mapping_display(self.client_code, self.model_group_id,
                                                                      self.overall_view)
        self.planner_data = readout_utils.apply_level_mapping_display(self.client_code, self.model_group_id, self.planner_data)
        if not self.overall_view.empty:
            agg_table_str, detail_table_str, self.overall_view, self.detail_view = readout_utils.table_to_text(
                self.overall_view, self.detail_view, take_head=False)
        else:
            agg_table_str, detail_table_str, self.agg_view, self.detail_view = readout_utils.table_to_text(
                self.agg_view, self.detail_view, take_head=False)

        if planner_readout:
            self.readout = planner_readout
        readout_adj = 'LINKEDIN no table readout' in self.readout
        if self.BI_readout_dict['ag']['overall_ag_only'] and self.rank_check=='na' and not readout_adj:
            instruction = readout_utils.load_prompt_instruction(self.client_code, self.model_group_id)
            if not instruction.empty:
                extra_inst_series = instruction.loc[instruction['intentionName'] == self.intention, 'instruction']
                extra_instruction = extra_inst_series.iloc[0] if not extra_inst_series.empty else ''
                custom_inst, outlier = readout_utils._bi_custom_instruction(self.intention, self.time, self.detail_view,
                                                                            client_code, model_group_id,
                                                                            self.ner_filter_dict,
                                                                            instruction)
                (overall_trend_str, self.pretext_table_trend) \
                    = readout_utils._get_pretext_overall_trend(client_code, model_group_id, ner_filter_dict,
                                                               ner_res_dict, self.driver_tags['ag'], metric_df,
                                                               self.detail_view)
                for text in [overall_trend_str, outlier]:
                    if text:
                        self.pretext += f"\n\n{text}"
                parts = []
                if isinstance(extra_instruction, str) and extra_instruction.strip():
                    parts.append(extra_instruction)
                if custom_inst.strip():
                    parts.append(custom_inst)
                # else:
                parts.append(self.readout)
                self.readout = '\n'.join(parts)
        elif not readout_adj:
            self.readout = readout_utils._table_readout_prompt(self.readout, agg_table_str, detail_table_str)
        else:
            self.readout =self.readout.replace('LINKEDIN no table readout. ', '')
        if self.intention=='spending' and self.readout:
            self.readout = ("Use passive tense when describing changes (e.g., 'was increased' / 'was reduced'). \n\n"
                            + self.readout)
        self.detail_view, self.agg_view = readout_utils._table_month_transform(self.detail_view), readout_utils._table_month_transform(self.agg_view)
        self.overall_view = readout_utils._table_month_transform(self.overall_view)
        self.detail_view.columns = [readout_utils._rename_change_columns(col) for col in self.detail_view.columns]
        self.agg_view.columns = [readout_utils._rename_change_columns(col) for col in self.agg_view.columns]
        self.overall_view.columns = [readout_utils._rename_change_columns(col) for col in self.overall_view.columns]
        self.pretext_table_trend = readout_utils._table_month_transform(self.pretext_table_trend)
        self.result_combo_match['is_specific'] = not self.BI_readout_dict['ag']['overall_ag_only']
        return (self.readout, self.detail_view, self.benchmark_data, self.benchmark_str, self.planner_data,
                self.spend_share_principle, self.pretext, self.agg_view, self.pretext_table, self.pretext_table_trend,
                self.result_combo_match, self.principle_pretext, self.overall_view, readout_adj)

    def _process_br_share(self, client_code: str, model_group_id: int, df_dict: dict[str, pd.DataFrame],
                          driver_tags: tuple, ner_filter_dict: dict, ner_res_dict: dict,
                          map_dict: dict, metric_df: pd.DataFrame):
        for level in ['ag', 'mg', 'm']:
            driver_tag = self.driver_tags_br[level]
            try:
                (readout, detail_df, benchmark_data, benchmark_str, planner_table, pretext, agg_df,
                 pivot_sort_col, overall_table) = self.function_br_detail(
                    client_code, model_group_id, df_dict, driver_tag, ner_filter_dict, ner_res_dict, map_dict,
                    metric_df)
                self.BR_readout_dict[level] = dict(readout=readout, detail=detail_df, benchmark_data=benchmark_data,
                                                   benchmark_str=benchmark_str,
                                                   planner_table=planner_table, pretext=pretext, agg=agg_df,
                                                   pivot_sort_col=pivot_sort_col, overall_table=overall_table)
            except Exception as e:
                logger.warning(f"Failed to process BR {level} results: {e}")
                self.BR_readout_dict[level] = dict(readout="", detail=pd.DataFrame(), benchmark_data=pd.DataFrame(),
                                                   benchmark_str="", planner_table=pd.DataFrame(), pretext="",
                                                   agg=pd.DataFrame(), pivot_sort_col="", overall_table=pd.DataFrame())

        if 'agg_mg' in self.df_lookup_br.columns:
            df_ls = [self.BR_readout_dict['ag']['detail'], self.BR_readout_dict['mg']['agg'], self.BR_readout_dict['mg']['detail'],
                     self.BR_readout_dict['m']['detail']]
            df_states = ['N' if df.empty else 'Y' for df in df_ls]
            df_states.append('Y' if self.BR_readout_dict['mg'].get('all_custom_agg', None) else 'N')
            k = 4
        else:
            df_ls = [self.BR_readout_dict['ag']['detail'], self.BR_readout_dict['mg']['detail'],
                 self.BR_readout_dict['m']['detail']]
            df_states = ['N' if df.empty else 'Y' for df in df_ls]
            k=3

        match = self.df_lookup_br[self.df_lookup_br['search_key'] == '~'.join(df_states[:k])]
        if len(match) > 1 and 'all_custom_agg' in match:
            match = match[(match['all_custom_agg'] == df_states[4])]
        if not match.empty:
            self.result_combo_match = match.reset_index().iloc[0, :].fillna("").to_dict()

            readout = self.result_combo_match['readout']
            self.readout = readout_utils._get_readout_com(readout, self.BR_readout_dict, self.result_combo_match,
                                                          self.data_type)

            self.detail_view = self.BR_readout_dict.get(self.result_combo_match['detail_view'].split('_')[-1], {}).get(
                self.result_combo_match['detail_view'].split('_')[0], pd.DataFrame())
            self.benchmark_data = pd.DataFrame()
            self.planner_data = pd.DataFrame()
            self.agg_view = self.BR_readout_dict.get(self.result_combo_match['agg_view'].split('_')[-1], {}).get(
                self.result_combo_match['agg_view'].split('_')[0], pd.DataFrame())
            self.benchmark_str = ""
            self.pretext = self.BR_readout_dict.get(self.result_combo_match['pretext'].split('_')[-1], {}).get(
                'pretext', "")
            self.pivot_sort_col = self.BR_readout_dict.get(self.result_combo_match['pivot_sort_col'].split('_')[-1],
                                                           {}).get('pivot_sort_col', "")
            self.pretext_table = self.BR_readout_dict.get(self.result_combo_match['pretext'].split('_')[-1], {}).get(
                'overall_table', pd.DataFrame())
        self.readout = readout_utils._add_na_term(self.core_dimension, self.readout)
        self.readout = readout_utils._metrics_aka(self.readout, self.ner_filter_dict)
        self.agg_view_copy = self.agg_view.copy()
        if 'Tactics' in self.detail_view.columns:
            self.detail_view, self.agg_view = readout_utils.dedup_post(self.detail_view, self.agg_view_copy, self.df_dict,
                                                                   self.result_combo_match['detail_view'].split('_')[
                                                                       -1],
                                                                   self.result_combo_match['agg_view'].split('_')[-1],
                                                                   self.client_code, self.main_metric, type='br')
            if not self.agg_view.empty and len(self.agg_view) < len(self.agg_view_copy):
                self.agg_view = self.agg_view_copy
        if 'Core_Dimension_Term' in self.detail_view.columns:
            self.dim_cols = self.dim_cols + ['Core_Dimension_Term']
        self.detail_view = readout_utils._get_metrics_rename(self.pivot_sort_col, self.metric_df, self.detail_view)
        self.agg_view = readout_utils._get_metrics_rename(self.pivot_sort_col, self.metric_df, self.agg_view)

        self.detail_view, self.agg_view = readout_utils._br_table_reformat(self.detail_view, self.agg_view)
        if self.detail_view.equals(self.agg_view):
            self.agg_view = pd.DataFrame()
            detail_level = self.result_combo_match['detail_view'].split('_')[-1] if self.result_combo_match.get('detail_view') else None
            if detail_level:
                self.readout = readout_utils._get_readout_com('readout_'+detail_level, self.BR_readout_dict, self.result_combo_match,
                                                          self.data_type)
        if df_ls[0].empty:
            biz_driver_check = ['base']
        else:
            if 'Business_Driver' in df_ls[0].columns:
                biz_driver_check = df_ls[0]['Business_Driver'].unique().tolist()
            else:
                biz_driver_check = ['base']
        self.detail_view = readout_utils.apply_level_mapping_display(self.client_code, self.model_group_id,
                                                                     self.detail_view)
        self.agg_view = readout_utils.apply_level_mapping_display(self.client_code, self.model_group_id, self.agg_view)
        agg_table_str, detail_table_str, self.agg_view, self.detail_view = readout_utils.table_to_text(
            self.agg_view, self.detail_view, take_head=False)
        # agg_table_str_cp, detail_table_str_cp = readout_utils.readout_adjust(client_code, model_group_id,
        #                                                                      self.detail_view, self.agg_view,
        #                                                                      ner_res_dict)
        # agg_table_str = agg_table_str_cp if agg_table_str_cp else agg_table_str
        # detail_table_str = detail_table_str_cp if detail_table_str_cp else detail_table_str
        readout_adj = 'LINKEDIN no table readout' in self.readout
        if ('base' not in biz_driver_check and 'other' not in biz_driver_check
                and not self.core_dimension and self.rank_check=='na') and not readout_adj:
            if self.intention in ['source of change', 'sales']:
                self.readout = readout_utils._table_readout_prompt(self.readout, agg_table_str, detail_table_str)
            if self.intention == 'contribution':
                overall_trend_str, self.pretext_table_trend = readout_utils._get_pretext_overall_trend(client_code, model_group_id,
                                                                             ner_filter_dict, ner_res_dict,
                                                                             self.driver_tags['ag'], metric_df, self.detail_view, driver_col='Business Driver')

                self.pretext += "\n" + overall_trend_str
            instruction = readout_utils.load_prompt_instruction(self.client_code, self.model_group_id)
            if not instruction.empty:
                extra_instruction = instruction[instruction['intentionName'] == self.intention]['instruction'].values[0]
                custom_inst, outlier = readout_utils._br_custom_instruction(self.intention, self.time, self.detail_view, self.ner_filter_dict)
                if outlier:
                    self.pretext += "\n\n" + outlier
                parts = []
                if isinstance(extra_instruction, str) and extra_instruction.strip():
                    parts.append(extra_instruction)
                if custom_inst.strip():
                    parts.append(custom_inst)
                # else:
                parts.append(self.readout)
                self.readout = '\n'.join(parts)
        elif not readout_adj:
            self.readout = readout_utils._table_readout_prompt(self.readout, agg_table_str, detail_table_str)
        else:
            self.readout =self.readout.replace('LINKEDIN no table readout. ', '')
        check_driver = any(not df.empty and 'business_driver' in df.columns
                           and df['business_driver'].isin(['base', 'other']).any()
                           for df in [df_dict.get('ag_df'), df_dict.get('mg_df'), df_dict.get('m_df')]
                           if df is not None)
        if check_driver and self.intention == 'contribution':
            self.readout = ("In our business context, 'Base' refers to Base Drivers—factors such as brand loyalty, pricing, "
                    "distribution, and seasonality. These represent the underlying baseline performance of the business, "
                    "against which we measure the true incremental impact of marketing campaigns.\n"
                    + self.readout)
        self.detail_view.columns = [readout_utils._rename_change_columns(col) for col in self.detail_view.columns]
        self.agg_view.columns = [readout_utils._rename_change_columns(col) for col in self.agg_view.columns]
        self.detail_view, self.agg_view = readout_utils._table_month_transform(self.detail_view), readout_utils._table_month_transform(self.agg_view)
        self.pretext_table_trend = readout_utils._table_month_transform(self.pretext_table_trend)
        self.result_combo_match['is_specific'] = bool(self.core_dimension and self.intention != 'sales' and self.business_driver != "irrelevant")
        return self.readout, self.detail_view, self.benchmark_data, self.benchmark_str, self.planner_data, self.spend_share_principle, self.pretext, self.agg_view, self.pretext_table, self.pretext_table_trend, self.result_combo_match, '', pd.DataFrame(), readout_adj


def set_environment_variables(path='config/function_mapping.yaml') -> None:
    logger.info('Setting up function mapping env variables')
    config = load_yaml_file(path)
    for key, value in config.items():
        os.environ[key] = value
    return None


def resolve_function(func_name):
    try:
        if '.' in func_name:
            module_or_class_name, function_name = func_name.rsplit('.', 1)
            try:
                module = importlib.import_module(module_or_class_name)
                return getattr(module, function_name)

            except ModuleNotFoundError:
                module_or_class = globals().get(module_or_class_name)
                if module_or_class and hasattr(module_or_class, function_name):
                    return getattr(module_or_class, function_name)

        return globals().get(func_name)

    except Exception as e:
        logging.warning(f"Failed to resolve function '{func_name}': {e}")
        return None

# Set environment variables
set_environment_variables()

# Load and resolve function mappings
client_func_map_bi = {
    key: resolve_function(func_name)
    for key, func_name in json.loads(os.getenv("client_func_map_bi")).items()
}

client_func_map_bi_detail = {
    key: resolve_function(func_name)
    for key, func_name in json.loads(os.getenv("client_func_map_bi_detail")).items()
}

client_func_map_br = {
    key: resolve_function(func_name)
    for key, func_name in json.loads(os.getenv("client_func_map_br")).items()
}

client_func_map_br_detail = {
    key: resolve_function(func_name)
    for key, func_name in json.loads(os.getenv("client_func_map_br_detail")).items()
}

client_func_map_driver = {
    key: resolve_function(func_name)
    for key, func_name in json.loads(os.getenv("client_func_map_driver")).items()
}


def _get_function_name_bi(client_code: str, model_group_id: int):
    key = f"{client_code}_{model_group_id}"
    return client_func_map_bi.get(key, Postprocess._process_bi_share)


def _get_function_name_bi_detail(client_code: str, model_group_id: int):
    key = f"{client_code}_{model_group_id}"
    return client_func_map_bi_detail.get(key, _process_bi_detail_share)


def _get_function_name_br(client_code: str, model_group_id: int):
    key = f"{client_code}_{model_group_id}"
    return client_func_map_br.get(key, Postprocess._process_br_share)


def _get_function_name_br_detail(client_code: str, model_group_id: int):
    key = f"{client_code}_{model_group_id}"
    return client_func_map_br_detail.get(key, _process_br_detail_share)


def _process_bi(client_code: str, model_group_id: int, df_dict: dict[str, pd.DataFrame], driver_tags: tuple,
                ner_filter: dict, ner_res_dict: dict, map_dict: dict, metric_df: pd.DataFrame):
    function_bi = _get_function_name_bi(client_code, model_group_id)
    if callable(function_bi):
        if hasattr(Postprocess, function_bi.__name__):  # Check if it's a method of Postprocess
            postprocessor = Postprocess(
                client_code, model_group_id, df_dict, driver_tags,
                ner_filter, ner_res_dict, map_dict, metric_df
            )
            function_bi = function_bi.__get__(postprocessor, Postprocess)
            return function_bi(
                client_code, model_group_id, df_dict, driver_tags,
                ner_filter, ner_res_dict, map_dict, metric_df
            )
        return function_bi(
            client_code, model_group_id, df_dict, driver_tags,
            ner_filter, ner_res_dict, map_dict, metric_df
        )
    else:
        logger.warning(f"No BI function found for client_code={client_code}, model_group_id={model_group_id}")
        return None


def _process_br(client_code: str, model_group_id: int, df_dict: dict[str, pd.DataFrame], driver_tags: tuple,
                ner_filter: dict, ner_res_dict: dict, map_dict: dict, metric_df: pd.DataFrame):
    function_br = _get_function_name_br(client_code, model_group_id)
    if callable(function_br):
        if hasattr(Postprocess, function_br.__name__):  # Check if it's a method of Postprocess
            postprocessor = Postprocess(
                client_code, model_group_id, df_dict, driver_tags,
                ner_filter, ner_res_dict, map_dict, metric_df
            )
            function_br = function_br.__get__(postprocessor, Postprocess)
            return function_br(
                client_code, model_group_id, df_dict, driver_tags,
                ner_filter, ner_res_dict, map_dict, metric_df
            )
        return function_br(
            client_code, model_group_id, df_dict, driver_tags,
            ner_filter, ner_res_dict, map_dict, metric_df
        )
    else:
        logger.warning(f"No BR function found for client_code={client_code}, model_group_id={model_group_id}")
        return None


def _get_function_name_driver(client_code: str, model_group_id: int):
    key = f"{client_code}_{model_group_id}"
    return client_func_map_driver.get(key, _get_driver_tag_default)


def process_data(client_code: str, model_group_id: int, readout_data: ReadoutData, process_indicator: ProcessIndicator=None):
    metric_df = readout_utils.load_metric_postprocess(client_code, model_group_id)
    ner_filter_dict = readout_data.ner_filters
    ner_res_dict = readout_data.ner_results
    df_dict = {
        'ag_df': readout_data.df_activity_group,
        'mg_df': readout_data.df_measure_group,
        'm_df': readout_data.df_measure,
    }
    driver_func = _get_function_name_driver(client_code, model_group_id)
    driver_tags = driver_func(ner_filter_dict, ner_res_dict)

    data_type = driver_tags[1]
    map_dict = readout_utils.load_map_dict(client_code, model_group_id)
    for m in map_dict.keys():
        map_dict[m]['sortMetric'] = map_dict[m]['sortMetric'].split(',')

    if data_type == 'bi':
        return _process_bi(client_code, model_group_id, df_dict, driver_tags, ner_filter_dict, ner_res_dict,
                           map_dict, metric_df)
    else:
        return _process_br(client_code, model_group_id, df_dict, driver_tags, ner_filter_dict, ner_res_dict,
                           map_dict, metric_df)

def response_generate(query: str, readout: str) -> str:
    prompt_template = PromptTemplates()
    readout = readout_utils.convert_display_format(readout)
    prompt = prompt_template.build_readout_prompt(query, readout)
    text = generate_text(prompt)
    return text

def response_generate_chart(query: str, readout: str, llm_vendor:str) -> str:
    prompt_template = PromptTemplates()
    readout = readout_utils.convert_display_format(readout)
    prompt = prompt_template.build_readout_prompt(query, readout)
    # text = generate_text(prompt)
    # text = generate_analysis_response(input_question=query,
    #                                   input_prompt=prompt,
    #                                   field_type="response")
    if llm_vendor == "ds-in-house":
        text = generate_text(prompt)
    else:
        text = generate_analysis_response(input_prompt=prompt,
                                          llm_vendor=llm_vendor,
                                          field_type="response")
    return text


def _parse_denial_response(text):
    """Parse the model's JSON output into the denial dict. Falls back to treating raw text as the answer."""
    if isinstance(text, dict):
        data = text
    else:
        try:
            cleaned = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            data = json.loads(cleaned)
        except (json.JSONDecodeError, AttributeError):
            return {"is_denial": False, "denial_reason": None, "answer": text}
    return {
        "is_denial": bool(data.get("is_denial", False)),
        "denial_reason": data.get("denial_reason"),
        "answer": data.get("answer", ""),
    }


def response_generate_with_denial(query: str, readout: str, llm_vendor: str) -> dict:
    prompt_template = PromptTemplates()
    readout = readout_utils.convert_display_format(readout)
    prompt = prompt_template.build_readout_prompt(query, readout)
    if llm_vendor == "ds-in-house":
        text = generate_text(prompt)
    else:
        text = generate_analysis_response(input_prompt=prompt,
                                          llm_vendor=llm_vendor,
                                          field_type="response")
    if isinstance(text, dict) and any(k.startswith("error_") or k == "error" for k in text):
        return text
    return _parse_denial_response(text)


def stream_response(query: str, readout: str) -> Iterator[str]:
    prompt_template = PromptTemplates()
    readout = readout_utils.convert_display_format(readout)
    prompt = prompt_template.build_readout_prompt(query, readout)
    it = stream_text(prompt)
    return it


def check_truncation(query:str, readout: str) -> bool:
    prompt_template = PromptTemplates()
    readout = readout_utils.convert_display_format(readout)
    truncated_readout = truncate_by_tokens(
        text=readout,
        buffer=500  # Rendered prompt excluding readout generally encode into <200 tokens
    )
    if len(truncated_readout) < len(readout):
        return True
    return False