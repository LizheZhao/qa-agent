import json
import logging
import os
import pathlib
import re
import sys
from typing import Optional, Dict, Iterator
import numpy as np
import pandas as pd

from src.data.data_interface import ReadoutData
from src.data.local import get_data_path
from src.integrations.llm import generate_text, stream_text
from src.model.common import prompt_template
import src.model.readout_utils as readout_utils
logger = logging.getLogger(__name__)


# def load_total_data(client_code: str, model_group_id: int) -> pd.DataFrame:
#     return pd.read_csv(get_data_path("total"))
#
# def load_map_dict(client_code: str, model_group_id: int) -> Dict:
#     df_map_dict = pd.read_csv(get_data_path("map_dict"))
#     map_dict = df_map_dict.set_index('intentionName').to_dict(orient='index')
#     return map_dict
# def _check_key_or_value(core_dim, search_terms):
#     found = {term: False for term in search_terms}
#     def recursive_search(core_dim):
#         if isinstance(core_dim, dict):
#             for key, value in core_dim.items():
#                 if key in search_terms:
#                     found[key] = True
#                 if isinstance(value, (dict, list)):
#                     recursive_search(value)
#         elif isinstance(core_dim, list):
#             for item in core_dim:
#                 recursive_search(item)
#     recursive_search(core_dim)
#     return found
#
# def _get_driver_tag(ner_filter_dict, ner_res_dict):
#     '''
#     :param ner_filter_dict:
#     :param ner_res_dict:
#     :return: filtered data, bi/br data, driver_high, driver_detail
#     '''
#     data_type = ner_filter_dict['data']
#     business_driver = ner_res_dict.get('business_driver', 'irrelevant')
#     media_channel = ner_res_dict.get('media_channel', 'irrelevant')
#     media_channel_ls = ner_filter_dict.get('media_channel', [])
#     custom_agg = ner_filter_dict.get('product_focused_media', ['no'])
#     core_dimension = ner_filter_dict.get('core_dimension', {})
#     search_terms = ['activity_group', 'measure_group', 'measure']
#     search_res = _check_key_or_value(core_dimension, search_terms)
#
#     if data_type == 'bi':
#         if custom_agg != ['no']:
#             driver_high = 'detail_tactics'
#         elif media_channel in ['all', 'irrelevant'] and not core_dimension:
#             driver_high = 'tactics'
#         elif isinstance(media_channel_ls, list) and len(media_channel_ls) >= 1:
#             if search_res.get('measure', False):
#                 driver_high = 'detail_tactics'
#             elif search_res.get('activity_group', False):
#                 driver_high = 'tactics'
#             else:
#                 driver_high = 'media_channel'
#         else:
#             driver_high = 'media_channel'
#         file_type = 'ag' if driver_high =='tactics' else 'mg'
#         driver_detail_map = {'ag': 'tactics', 'mg': 'detail_tactics', 'm': 'tactics_detail'}
#         driver_detail = driver_detail_map[file_type]
#     else:
#         driver_high = 'business_driver'
#         if custom_agg != ['no']:
#             file_type = 'm'
#             driver_detail = 'detail_tactics'
#         elif business_driver == 'all':
#             file_type = 'ag'
#             driver_detail = 'business_driver_detail'
#         elif core_dimension:
#             if search_res.get('measure', False):
#                 file_type = 'm'
#                 driver_detail = 'detail_tactics'
#             else:
#                 file_type = 'mg'
#                 driver_detail = 'tactics'
#         else:
#             file_type = 'mg'
#             driver_detail = 'tactics'
#     return file_type, data_type, driver_high, driver_detail
#
# def _drop_col_dict(filt_data, ner_res_dict):
#     '''
#     :param filt_data: ner filtered data
#     :param ner_res_dict:
#     :return: data without redundant columns
#     '''
#     drop_cols = [key for key, value in ner_res_dict.items() if isinstance(value, float) and np.isnan(value)]
#     drop_cols = [c for c in drop_cols if c != 'activity_group']
#     empty_cols = filt_data.columns[filt_data.isna().all(axis=0)].tolist()
#     empty_cols = [c for c in empty_cols if c not in ['business_driver', 'business_driver_detail']]
#     drop = drop_cols + empty_cols + ['clientmetric', 'record_level']
#     data = filt_data.drop(columns=drop, errors='ignore')
#     return data
#
# def _rename_data(data):
#     data = data.rename(
#         columns={'activity_group': 'tactics', 'measure_group': 'tactics_detail', 'measure': 'detail_tactics',
#                  'activity_group_origin': 'tactics_origin',
#                  'measure_group_origin': 'tactics_detail_origin',
#                  'measure_origin': 'detail_tactics_origin'}, errors='ignore')
#     return data
#
# def _convert_time(data, time_col):
#     # fiscal year 2022 --> 2022
#     data[time_col] = pd.to_numeric(data[time_col].str.extract(r'(\d{4})')[0])
#     return data
#
# def _convert_quarter(data, time_col):
#     # fiscal quarter 1 2022 --> Q1 2022
#     data[time_col] = data[time_col].apply(lambda x: re.sub(r'fiscal quarter (\d) (\d{4})', r'Q\1 \2', x))
#     return data

# def _read_and_process_csv(data, ner_filter):
#     '''
#     :param data: ner filtered dataframe
#     :param ner_filter:
#     :return: cleaned dataframe
#     '''
#     data = _drop_col_dict(data, ner_filter)
#     data = _rename_data(data)
#     period_type = ner_filter['period_type'][0]
#     custom_agg = ner_filter.get('product_focused_media', ['no'])
#     if len(data) != 0:
#         if 'year' in period_type:
#             data = _convert_time(data, 'time')
#         elif 'quarter' in period_type:
#             data = _convert_quarter(data, 'time')
#     if data.columns[0] not in ['product', 'category']:
#         data['product'] = 'overall'
#     else:
#         if custom_agg != ['no']:
#             prod_ls = data['product'].unique().tolist()
#             if 'product_focused_media' in data.columns:
#                 if not data['product_focused_media'].isnull().any():
#                     if len(prod_ls)>1 and 'overall' in prod_ls:     #overall as default
#                         data = data[data['product'] == 'overall']
#                         data['product'] = data['product_focused_media']
#                     elif len(prod_ls)>1 and 'overall' not in prod_ls:
#                         data['product'] = data['product'] + ' - ' + data['product_focused_media']
#                     else:
#                         data['product'] = data['product_focused_media']
#         else:
#             data=data.rename(columns={'product': 'channel'})
#
#     if 'product' in data.columns:
#         cols = ['product'] + [col for col in data.columns if col != 'product']
#         data = data[cols]
#     return data


# def _concatenate_ag_mg_bi(mg_df):
#     '''
#     :param ag_df:
#     :param mg_df:
#     :return: concatenate AG and MG values
#     '''
#     data = mg_df
#     data['tactics_concat'] = data['tactics'] + " (" + data['detail_tactics'] + ")"
#     data['tactics_concat_origin'] = data['tactics_origin'] + " (" + data['detail_tactics_origin'] + ")"
#     return data

# def _get_ner_key_value(data, driver_high, file_type, data_type, driver_detail):
#     '''
#     :param data:
#     :param driver_high:
#     :param file_type:
#     :param data_type:
#     :param driver_detail:
#     :return: dict, answer level mapping, answering at which driver level
#     '''
#     ner_dict = {'dimension': data.columns[0], 'time': 'time', 'metric': 'metric', 'driver_high': driver_high,
#                 'driver_detail': driver_detail}
#     if data_type == 'bi':
#         file_driver_map = {'ag': 'tactics', 'mg': 'tactics_concat', 'm': 'tactics_detail'}
#         if driver_high == 'detail_tactics' and driver_detail == 'detail_tactics':
#             file_driver_map['mg'] = 'detail_tactics'
#         ner_dict['driver'] = file_driver_map[file_type]
#     else:
#         file_driver_map_br = {'ag': 'business_driver_detail', 'mg': 'tactics', 'm': 'detail_tactics'}
#         ner_dict['driver'] = file_driver_map_br[file_type]
#     ner_dict['value'] = 'value'
#     ner_dict['period'] = 'period_type'
#     return ner_dict

# def _calc_yoy(data, ner_dict):
#     '''
#     :param data: cleaned dataframe
#     :param ner_dict:
#     :return: dataframe with YOY percentage change
#     '''
#     if not data.empty:
#         dimension, time, driver, value, metric = ner_dict['dimension'], ner_dict['time'], ner_dict['driver'], ner_dict[
#             'value'], ner_dict['metric']
#         data[value] = pd.to_numeric(data[value], errors='coerce')
#         data.sort_values(by=[dimension, metric, driver, time], inplace=True)
#         data['value_prev'] = data.groupby([dimension, metric, driver])[value].shift(1)
#         def calc_yoy_percent_change(row):
#             if pd.isna(row['value_prev']):
#                 return np.nan
#             try:
#                 yoy_change = (row[value] - row['value_prev']) / abs(row['value_prev']) * 100
#                 return yoy_change
#             except ZeroDivisionError:
#                 return np.nan
#
#         data['yoy_percent_change'] = data.apply(calc_yoy_percent_change, axis=1)
#         data = data.round(2)
#         data = data.drop(columns=['value_prev'])
#     return data
#
# def _calc_yoy_total(data, ner_dict):
#     if not data.empty:
#         dimension, time, value, metric = ner_dict['dimension'], ner_dict['time'], ner_dict['value'], ner_dict['metric']
#         data[value] = pd.to_numeric(data[value], errors='coerce')
#         data.sort_values(by=[dimension, metric, time], inplace=True)
#         data['value_prev'] = data.groupby([dimension, metric])[value].shift(1)
#         def calc_yoy_percent_change(row):
#             if pd.isna(row['value_prev']):
#                 return np.nan
#             try:
#                 yoy_change = (row[value] - row['value_prev']) / abs(row['value_prev']) * 100
#                 return yoy_change
#             except ZeroDivisionError:
#                 return np.nan
#
#         data['yoy_percent_change'] = data.apply(calc_yoy_percent_change, axis=1)
#         data = data.round(2)
#         data = data.drop(columns=['value_prev'])
#     return data
#
# def _calc_qoq(data, ner_dict):
#     '''
#     :param data: cleaned dataframe
#     :param ner_dict:
#     :return: dataframe with QOQ percentage change
#     '''
#     if not data.empty:
#         dimension, time, driver, value, metric = ner_dict['dimension'], ner_dict['time'], ner_dict['driver'], ner_dict[
#             'value'], ner_dict['metric']
#         data[value] = pd.to_numeric(data[value], errors='coerce')
#         data.sort_values(by=[dimension, metric, driver, time], inplace=True)
#         data[['quarter', 'year']] = data[time].str.extract(r'Q(\d) (\d{4})')
#         data['year'] = data['year'].astype(int)
#         data['quarter'] = data['quarter'].astype(int)
#         data['value_prev'] = data.groupby([dimension, metric, driver, 'quarter'])[value].shift(1)
#         data['year_prev'] = data.groupby([dimension, metric, driver, 'quarter'])['year'].shift(1)
#         def calc_qoq_percent_change(row):
#             if pd.isna(row['value_prev']) or row['year'] != row['year_prev'] + 1:
#                 return np.nan
#             try:
#                 qoq_change = (row['value'] - row['value_prev']) / abs(row['value_prev']) * 100
#                 return qoq_change
#             except ZeroDivisionError:
#                 return np.nan
#         data['qoq_percent_change'] = data.apply(calc_qoq_percent_change, axis=1)
#         data = data.round(2)
#         data = data.drop(columns=['value_prev', 'year_prev'])
#     return data
#
# def _calc_qoq_total(data, ner_dict):
#     if not data.empty:
#         dimension, time, value, metric = ner_dict['dimension'], ner_dict['time'], ner_dict['value'], ner_dict['metric']
#         data[value] = pd.to_numeric(data[value], errors='coerce')
#         data = data.sort_values(by=[dimension, metric, time])
#         data[['quarter', 'year']] = data[time].str.extract(r'Q(\d) (\d{4})')
#         data['year'] = data['year'].astype(int)
#         data['quarter'] = data['quarter'].astype(int)
#         data['value_prev'] = data.groupby([dimension, metric, 'quarter'])[value].shift(1)
#         data['year_prev'] = data.groupby([dimension, metric, 'quarter'])['year'].shift(1)
#         def calc_qoq_percent_change(row):
#             if pd.isna(row['value_prev']) or row['year'] != row['year_prev'] + 1:
#                 return np.nan
#             try:
#                 qoq_change = (row['value'] - row['value_prev']) / abs(row['value_prev']) * 100
#                 return qoq_change
#             except ZeroDivisionError:
#                 return np.nan
#
#         data['qoq_percent_change'] = data.apply(calc_qoq_percent_change, axis=1)
#         data = data.round(2)
#         data = data.drop(columns=['value_prev', 'year_prev'])
#     return data

# def _sort_filt_data(data, ner_dict, sort_metric, sort_by_col, rank_check):
#     '''
#     Sort tactics/drivers by metric values
#     :param data:
#     :param ner_dict:
#     :param sort_metric:
#     :return: sorted dataframe with rank
#     '''
#     driver, time, value, dim_col, metric_col = ner_dict['driver'], ner_dict['time'], ner_dict['value'], ner_dict[
#         'dimension'], ner_dict['metric']
#
#     ascend = True if rank_check == 'bottom' else False
#
#     def process_group(group):
#         sort_metric_data = group[group[metric_col] == sort_metric]
#         most_recent_year = sort_metric_data[time].max()
#         most_recent_data = sort_metric_data[sort_metric_data[time] == most_recent_year]
#         if sort_metric in ['sourceofchange', 'contribution']:
#             most_recent_data['abs_value'] = most_recent_data[value].abs()
#             ordered_drivers_ranked = most_recent_data.sort_values(by='abs_value', ascending=ascend)
#             ordered_drivers_ranked['rank'] = ordered_drivers_ranked['abs_value'].rank(method='dense', ascending=ascend)
#             ordered_drivers_ranked = ordered_drivers_ranked.drop('abs_value', axis=1)
#         else:
#             if most_recent_data[sort_by_col].isna().all():
#                 ordered_drivers_ranked = most_recent_data.sort_values(by='value', ascending=ascend)
#                 ordered_drivers_ranked['rank'] = ordered_drivers_ranked['value'].rank(method='dense', ascending=ascend)
#             else:
#                 ordered_drivers_ranked = most_recent_data.sort_values(by=sort_by_col, ascending=ascend)
#                 ordered_drivers_ranked['rank'] = ordered_drivers_ranked[sort_by_col].rank(method='dense', ascending=ascend)
#         return ordered_drivers_ranked
#
#     # Apply the function to each dimension group, then merge back to the original dataframe
#     if ner_dict['driver_high'] == 'media_channel':
#         ranked_groups = data.groupby([dim_col, 'media_channel']).apply(process_group).reset_index(level=0, drop=True)
#     else:
#         ranked_groups = data.groupby(dim_col).apply(process_group).reset_index(level=0, drop=True)
#     # Merge the ranked data with original data on the specific columns
#     merged_data = pd.merge(data, ranked_groups[[dim_col, driver, 'rank']], on=[dim_col, driver], how='left')
#     # Sort the merged data by dimension, rank, and metric column
#     merged_data.sort_values(by=[dim_col, 'rank', metric_col], inplace=True)
#
#     return merged_data

# def _get_granular_table(data, ner_dict, metrics_order, time_order, ner_filter, rank_check, sort_by_col, client_code):
#     time_col, dim_col, driver_col_detail, metric_col, value_col = ner_dict['time'], ner_dict['dimension'], ner_dict[
#         'driver'], ner_dict['metric'], ner_dict['value']
#     if driver_col_detail in ['tactics', 'tactics_detail', 'tactics_concat', 'detail_tactics', 'media_tactics_concat']:
#         driver_col_detail_upp = driver_col_detail + '_origin'
#     else:
#         driver_col_detail_upp = driver_col_detail
#     pivot_data, sort_col = _table_display(data, dim_col, driver_col_detail_upp, time_col, metric_col, value_col, metrics_order,
#                                 time_order, ner_filter, rank_check, sort_by_col, client_code)
#     return pivot_data, sort_col
# def _reorder_columns(dynamic_cols, metric_order, quarter_check, month_check):
#     if quarter_check:
#         idx = -3
#     elif month_check:
#         idx = -4
#     else:
#         idx = -2
#     def parse_column(col):
#         parts = col.split()
#         if 'QOQ' in parts or 'YOY' in parts:
#             metric = ' '.join(parts[:idx])
#             time = ' '.join(parts[idx: -1])
#             growth = 2
#         else:
#             metric = ' '.join(parts[:idx+1])
#             time = ' '.join(parts[idx+1:])
#             growth = 1
#         return metric, time, growth
#     metric_mapping = {metric: i for i, metric in enumerate(metric_order)}
#     columns_df = pd.DataFrame([parse_column(col) for col in dynamic_cols], columns=['metric', 'time', 'growth'])
#     columns_df['original_col'] = dynamic_cols
#     columns_df['metric_order'] = columns_df['metric'].map(metric_mapping)
#     if quarter_check:
#         columns_df[['num', 'year']] = columns_df['time'].str.extract(r'Q(\d) (\d{4})')
#     elif month_check:
#         extract_col = columns_df['time'].str.extract(r'month (\d{1,2}) (\d{4})')
#         columns_df['num'] = extract_col[0].astype(int)
#         columns_df['year'] = extract_col[1].astype(int)
#     else:
#         columns_df['year'] = columns_df['time'].str.extract(r'(\d{4})')
#         columns_df['num'] = 1
#     columns_df.sort_values(by=['metric_order', 'growth', 'year', 'num'], ascending=[True, True, False, False],
#                            inplace=True)
#     return columns_df['original_col'].tolist()

# def _table_display(data, dim_col, driver_col, time_col, metric_col, value_col, metrics_order, time_order, ner_filter, rank_check, sort_by_col, client_code):
#     data = data.round(2)
#     data = data[data[metric_col] != 'margin roi index']
#     growth_col = next((col for col in ['yoy_percent_change', 'qoq_percent_change'] if col in data.columns), None)
#     quarter_check = 'quarter' in data.columns
#     month_check = 'month' in data.columns
#     main_metric = ner_filter['main_metric'] if ner_filter['main_metric'] != 'sourceofchange' else 'source of change'
#     most_recent_time = str(data['time'].max())
#     ascend = True if rank_check == 'bottom' else False
#     abs_sort = True if sort_by_col == 'value' and main_metric in ['source of change', 'contribution'] else False
#     value_cols = [value_col]
#     if growth_col:
#         value_cols.append(growth_col)
#
#     kpi_name_mapping = {'sourceofchange': 'Source of Change'}
#     pivot_data = data.pivot_table(index=[dim_col, driver_col], columns=[metric_col, time_col],
#                                  values=value_cols, aggfunc='first')
#
#     pivot_data.columns = ['{} {} {}'.format(kpi_name_mapping.get(col[1], col[1]), col[2], col[0])
#                           for col in pivot_data.columns]
#     pivot_data = pivot_data.reset_index()
#     dim_col_order, driver_col_order = data[dim_col].unique().tolist(), data[driver_col].unique().tolist()
#     pivot_data[dim_col] = pd.Categorical(pivot_data[dim_col], categories=dim_col_order, ordered=True)
#     pivot_data[driver_col] = pd.Categorical(pivot_data[driver_col], categories=driver_col_order, ordered=True)
#     pivot_data = pivot_data.sort_values(by=[dim_col, driver_col])
#     pivot_data.columns = pivot_data.columns.str.replace(' value', '')
#     pivot_data.columns = pivot_data.columns.str.replace(' yoy_percent_change', ' YOY')
#     pivot_data.columns = pivot_data.columns.str.replace(' qoq_percent_change', ' QOQ')
#
#     if 'tactics' in driver_col:
#         pivot_data.columns = pivot_data.columns.str.replace(driver_col, 'tactics')
#     dim_map = _dim_value_map()
#     pivot_data[dim_col] = pivot_data[dim_col].map(dim_map).fillna(pivot_data[dim_col])
#
#     static_columns = pivot_data.columns[:2].tolist()
#
#     dynamic_columns = [col for col in pivot_data.columns if col not in static_columns]
#     pivot_data[dynamic_columns] = pivot_data[dynamic_columns].replace([np.inf, -np.inf], np.nan)
#     pivot_data = pivot_data.dropna(axis=0, how='all', subset=dynamic_columns)
#     sorted_dynamic_columns = _reorder_columns(dynamic_columns, metrics_order, quarter_check, month_check)
#     # Sort columns using the sorting key
#     sorted_columns = static_columns + sorted_dynamic_columns
#     pivot_data = pivot_data[sorted_columns]
#
#     def _sort_condition(col):
#         if sort_by_col != 'value':
#             return (most_recent_time in col) and (main_metric in col.lower()) and (
#                         'yoy' in col.lower() or 'qoq' in col.lower())
#         else:
#             return (most_recent_time in col) and (main_metric in col.lower()) and not (
#                         'yoy' in col.lower() or 'qoq' in col.lower())
#
#     pivot_sort_cols = [col for col in pivot_data.columns if
#                        _sort_condition(col) and 'share of spend' not in col.lower()]
#     pivot_sort_col = None
#     if pivot_sort_cols and not month_check:
#         pivot_sort_col = pivot_sort_cols[0]
#         if abs_sort:
#             pivot_data = pivot_data.reindex(pivot_data[pivot_sort_col].abs().sort_values(ascending=ascend).index)
#         else:
#             pivot_data = pivot_data.sort_values(by=pivot_sort_col, ascending=ascend)
#         if len(dim_col_order) > 1:
#             pivot_data['sort_helper'] = pivot_data.groupby(dim_col).cumcount()
#             pivot_data = pivot_data.sort_values(by=[dim_col, 'sort_helper'])
#             pivot_data = pivot_data.groupby(dim_col).head(5)
#             pivot_data = pivot_data.drop(columns=['sort_helper'])
#
#     # spend, response, activity with MM
#     def transform_to_mm(df):
#         for col in df.columns:
#             if (kpi_name_mapping.get('share of spend', 'share of spend') not in col and 'YOY' not in col
#                     and kpi_name_mapping.get('cost per', 'cost per') not in col and 'QOQ' not in col):
#                 # if 'spend' in col or 'response' in col or 'activity' in col:
#                 if 'spend' in col or 'activity' in col:
#                     df[col] = df[col] / 1000000  # Convert to millions
#                     df[col] = df[col].apply(lambda x: f"{x:.2f}" if not pd.isna(x) else x)
#                     df.rename(columns={col: col + ' (MM)'}, inplace=True)
#         return df
#
#     pivot_data = transform_to_mm(pivot_data)
#
#     # YOY, share of spend add %
#     # soc, contribution add %
#     def append_percentage(df):
#         # Iterate over columns and apply transformation if conditions are met
#         for col in df.columns:
#             if 'YOY' in col or 'share of spend' in col or 'contribution' in col or 'sourceofchange' in col or 'source of change' in col or 'QOQ' in col:
#                 df[col] = pd.to_numeric(df[col], errors='coerce')
#                 df[col] = df[col].apply(lambda x: f"{x:.1f}%" if not pd.isna(x) else x)
#         return df
#
#     pivot_data = append_percentage(pivot_data)
#
#     # sales add $, 0,000
#
#     def format_sales_values(df):
#         for col in df.columns:
#             if (kpi_name_mapping.get('sales', 'sales') in col
#                     and kpi_name_mapping.get('response', 'response') not in col and 'YOY' not in col
#                     and 'QOQ' not in col):
#                 df[col] = df[col].apply(lambda x: f"${x:,.2f}" if not pd.isna(x) else x)
#         return df
#
#     pivot_data = format_sales_values(pivot_data)
#
#     def thousand_comma(df):
#         for col in df.columns:
#             if kpi_name_mapping.get('response', 'response') in col and 'YOY' not in col and 'QOQ' not in col:
#                 df[col] = df[col].apply(lambda x: "{:,}".format(x) if pd.notna(x) else "")
#         return df
#
#     pivot_data = thousand_comma(pivot_data)
#
#     # column name uppercase
#     def custom_title_case(column_name):
#         words = column_name.split()
#         final_words = [word.upper() if word in ['YOY', '(MM)', 'roi', 'QOQ'] else word.title() for word in words]
#         return ' '.join(final_words)
#
#     pivot_data.columns = [custom_title_case(col) for col in pivot_data.columns]
#     if pivot_sort_col:
#         pivot_sort_col = custom_title_case(pivot_sort_col)
#     return pivot_data, pivot_sort_col

# def _top_data(data, ner_dict, top_num='na'):
#     '''
#
#     :param data:
#     :param ner_dict:
#     :param top_num:
#     :return: top_num tactics/drivers dataframe
#     '''
#     rank_num = data['rank'].max()
#
#     if rank_num >= 3:
#         top_num = 3 if top_num == 'na' else min(int(top_num), rank_num)
#     else:
#         top_num = rank_num if top_num == 'na' else min(int(top_num), rank_num)
#     top_data = data[data['rank'] <= top_num]
#     if ner_dict['driver_high'] == 'media_channel':
#         top_data = data[data['rank'] <= top_num]
#         def dedup(group):
#             if group['tactics_concat'].nunique() > 1:
#                 first_value = group['tactics_concat'].iloc[0]
#                 group = group[group['tactics_concat'] == first_value]
#             return group
#
#         top_data = top_data.groupby([ner_dict['dimension'], 'tactics_concat']).apply(dedup).reset_index(drop=True)
#     return top_data

# def _top_increase_data(data, ner_dict, growth_col, main_metric, top_num=1):
#     '''
#     :return: top 1 growing tactic/driver data
#     '''
#     driver, time, value, dim_col, metric_col = ner_dict['driver'], ner_dict['time'], growth_col, ner_dict[
#         'dimension'], ner_dict['metric']
#     most_recent_time = data[time].sort_values(ascending=False).values[0]
#     if 'qoq' in growth_col:
#         most_recent_year = data['year'].max()
#         t = 'quarter'
#         most_recent_quarter = data[data['year'] == most_recent_year][t].max()
#         most_recent_time = data[(data[t] == most_recent_quarter) & (data['year'] == most_recent_year)][time].values[0]
#     growth_data = data[(data[time] == most_recent_time) & (data[value].notnull()) & (data[metric_col] == main_metric)]
#
#     top_inc_data = growth_data.groupby([dim_col, metric_col], as_index=False).apply(
#         lambda x: x.nlargest(top_num, value)).reset_index(drop=True)
#
#     return top_inc_data
#
# def _top_decrease_data(data, ner_dict, growth_col, main_metric, top_num=1):
#     '''
#     :return: top 1 decreasing tactic/driver data
#     '''
#     driver, time, value, dim_col, metric_col = ner_dict['driver'], ner_dict['time'], growth_col, ner_dict[
#         'dimension'], ner_dict['metric']
#     most_recent_time = data[time].sort_values(ascending=False).values[0]
#     if 'qoq' in growth_col:
#         most_recent_year = data['year'].max()
#         t = 'quarter'
#         most_recent_quarter = data[data['year'] == most_recent_year][t].max()
#         most_recent_time = data[(data[t] == most_recent_quarter) & (data['year'] == most_recent_year)][time].values[0]
#     growth_data = data[(data[time] == most_recent_time) & (data[value].notnull()) & (data[metric_col] == main_metric)]
#
#     top_dec_data = growth_data.groupby([dim_col, metric_col], as_index=False).apply(
#         lambda x: x.nsmallest(top_num, value)).reset_index(drop=True)
#     return top_dec_data
#
#
# def _get_granular_level_pivot(data, ner_dict, ner_filter, trend_check):
#     time_col, dim_col, driver_col_detail, metric_col, value_col = ner_dict['time'], ner_dict['dimension'], ner_dict[
#         'driver'], ner_dict['metric'], ner_dict['value']
#     main_metric = ner_filter['main_metric']
#     output_text = _data_to_text(data, dim_col, driver_col_detail, time_col, metric_col, value_col,
#                                 main_metric, trend_check)
#     return output_text
#
# def _data_to_text(data, dim_col, driver_col, time_col, metric_col, value_col, main_metric, trend_check):
#     data = data.round(2)
#     dimension = data[dim_col].unique().tolist()  # > 1
#     growth_col = next((col for col in ['yoy_percent_change', 'qoq_percent_change'] if col in data.columns), None)
#     output_text = ""
#     for dim in dimension:
#         if dim != 'overall':
#             output_text += f"For '{dim}': "
#         dim_group = data[data[dim_col] == dim]
#         # Get the most recent year
#         if 'year' in dim_group.columns:
#             most_recent_year = dim_group['year'].max()
#             recent_year_values = dim_group[
#                 (dim_group['year'] == most_recent_year) & (dim_group[metric_col] == main_metric)]
#         else:
#             most_recent_year = dim_group[time_col].sort_values(ascending=False).values[0]
#             recent_year_values = dim_group[
#                 (dim_group[time_col] == most_recent_year) & (dim_group[metric_col] == main_metric)]
#
#         if 'quarter' in recent_year_values.columns:
#             quarter = recent_year_values['quarter'].max()
#             filtered = dim_group[dim_group['quarter'] == quarter]
#             recent_periods = filtered[time_col].sort_values(ascending=False).unique()[:2].tolist()
#         else:
#             recent_periods = None
#
#         sorted_drivers = recent_year_values.groupby(driver_col)[value_col].mean().sort_values(
#             ascending=False).index.tolist()
#         for driver in sorted_drivers:
#             driver_group = dim_group[dim_group[driver_col] == driver]
#             if recent_periods:
#                 driver_group = driver_group[driver_group[time_col].isin(recent_periods)]
#             sorted_group = driver_group.sort_values(by=time_col, ascending=False)
#             values_by_time = sorted_group[sorted_group[metric_col] == main_metric][value_col].tolist()
#             period = sorted_group[time_col].unique().tolist()
#             if main_metric not in ['sourceofchange', 'contribution']:
#                 output_text += f"'{driver}' '{main_metric}' is {', '.join([str(v) for v in values_by_time])} in {', '.join(map(str, period))} "
#             elif main_metric == 'sourceofchange':
#                 output_text += f"'{driver}' drives {', '.join([str(v) + '%' for v in values_by_time])} sales change in {', '.join(map(str, period))}"
#             elif main_metric == 'contribution':
#                 output_text += f"'{driver}' '{main_metric}' is {', '.join([str(v) + '%' for v in values_by_time])} in {', '.join(map(str, period))}"
#
#             if len(period) > 1:
#                 output_text += " respectively"
#             if growth_col and trend_check != 'no':
#                 change_type_mapping = {'qoq': 'QOQ', 'yoy': 'YOY'}
#                 change_type = next((change_type_mapping[key] for key in change_type_mapping if key in growth_col), "")
#                 growth_by_time = \
#                     sorted_group[(sorted_group[metric_col] == main_metric) & (sorted_group[growth_col].notnull())][
#                         [growth_col, time_col]]
#                 growths = growth_by_time[growth_col].tolist()
#                 periods = growth_by_time[time_col].tolist()
#                 if len(growths) != 0:
#                     output_text += f", resulting in {', '.join([f'{g:.2f}%' for g in growths])} {change_type} change in {', '.join([str(t) for t in periods])}"
#                 if len(periods) > 1:
#                     output_text += " respectively"
#
#             output_text += ".\n"
#
#     return output_text
#
# def _dim_value_map():
#     # Titled dimension values
#     dim_map = {'overall': 'Overall', 'total tp': 'Total TP', 'all other tp': 'All Other TP',
#                'optic white tp': 'Optic White TP', 'max tp': 'Max TP', 'masterbrand': 'Masterbrand', 'ahw': 'AHW', 'ahw led': 'AHW LED', 'ahw pen': 'AHW PEN',
#                'optic white': 'Optic White', 'tp': 'TP', 'total': 'Total'}
#     return dim_map
#
# def _get_total_level_pivot(data, ner_dict, ner_filter):
#     time_col, dim_col, metric_col, value_col = ner_dict['time'], ner_dict['dimension'], ner_dict['metric'], ner_dict[
#         'value']
#     main_metric = ner_filter['main_metric']
#     output_text = _total_to_text(data, dim_col, time_col, metric_col, value_col, main_metric)
#     return output_text
#
# def _total_to_text(data, dim_col, time_col, metric_col, value_col, main_metric):
#     data = data.round(2)
#     dimension = data[dim_col].unique().tolist()  # > 1
#     growth_col = next((col for col in ['yoy_percent_change', 'qoq_percent_change'] if col in data.columns), None)
#
#     output_text = ""
#     # if dimension:
#     for dim in dimension:  # , dim_group in pivot_data.groupby(dim_col):
#         if dim != 'overall':
#             output_text += f"For '{dim}': "
#         dim_group = data[(data[dim_col] == dim) & (data[metric_col] == main_metric)]
#         if 'quarter' in dim_group.columns:
#             quarter = dim_group['quarter'].max()
#             filtered = dim_group[dim_group['quarter'] == quarter]
#             recent_periods = filtered[time_col].sort_values(ascending=False).unique()[:2].tolist()
#             dim_group = dim_group[dim_group[time_col].isin(recent_periods)]
#         sorted_group = dim_group.sort_values(by=time_col, ascending=False)
#         values_by_time = sorted_group[value_col].tolist()
#         period = sorted_group[time_col].unique().tolist()
#         output_text += f"Overall '{main_metric}' is {', '.join([str(v) + ' in ' + str(p) for v, p in zip(values_by_time, period)])}"
#         if len(period) > 1:
#             output_text += " respectively"
#         if growth_col:
#             change_type_mapping = {'qoq': 'QOQ', 'yoy': 'YOY'}
#             change_type = next((change_type_mapping[key] for key in change_type_mapping if key in growth_col), "")
#             growth_by_time = \
#                 sorted_group[sorted_group[growth_col].notnull()][[growth_col, time_col]]
#             growths = growth_by_time[growth_col].tolist()
#             periods = growth_by_time[time_col].tolist()
#             if len(growths)!=0:
#                 output_text += f", resulting in {', '.join([f'{g:.2f}%' for g in growths])} {change_type} change in {', '.join([str(t) for t in periods])}"
#             if len(periods) > 1:
#                 output_text += " respectively"
#         output_text += ".\n"
#     return output_text
#
#
# def _get_total_table(data, ner_dict, ner_filter, rank_check, sort_by_col):
#     time_col, dim_col, metric_col, value_col = ner_dict['time'], ner_dict['dimension'], ner_dict['metric'], ner_dict[
#         'value']
#     pivot_data = _total_table_display(data, dim_col, time_col, metric_col, value_col, ner_filter, rank_check, sort_by_col)
#     return pivot_data
#
# def _total_table_display(data, dim_col, time_col, metric_col, value_col, ner_filter, rank_check, sort_by_col):
#     data = data.round(2)
#     growth_col = next((col for col in ['yoy_percent_change', 'qoq_percent_change'] if col in data.columns), None)
#     quarter_check = 'quarter' in data.columns
#     month_check = 'month' in data.columns
#     ascend = True if rank_check == 'bottom' else False
#     main_metric = ner_filter['main_metric']
#     most_recent_time = str(data['time'].max())
#     pivot_data = data.pivot_table(index=[dim_col], columns=[metric_col, time_col],
#                                   values=[value_col, growth_col], aggfunc='first')
#     pivot_data.columns = ['{} {} {}'.format(col[1], col[2], col[0])
#                           for col in pivot_data.columns]
#     pivot_data = pivot_data.reset_index()
#     dim_col_order = data[dim_col].unique().tolist()
#     pivot_data[dim_col] = pd.Categorical(pivot_data[dim_col], categories=dim_col_order, ordered=True)
#     pivot_data = pivot_data.sort_values(by=[dim_col], ascending = ascend)
#     pivot_data.columns = pivot_data.columns.str.replace(' value', '')
#     pivot_data.columns = pivot_data.columns.str.replace(' yoy_percent_change', ' YOY')
#     pivot_data.columns = pivot_data.columns.str.replace(' qoq_percent_change', ' QOQ')
#
#     dim_map = _dim_value_map()
#     pivot_data[dim_col] = pivot_data[dim_col].map(dim_map).fillna(pivot_data[dim_col])
#     time_order = sorted(data[time_col].unique().tolist(), reverse=True)
#
#     static_columns = pivot_data.columns[:1].tolist()
#     dynamic_columns = [col for col in pivot_data.columns if col not in static_columns]
#     pivot_data[dynamic_columns] = pivot_data[dynamic_columns].replace([np.inf, -np.inf], np.nan)
#     pivot_data = pivot_data.dropna(axis=0, how='all', subset=dynamic_columns)
#
#     sorted_dynamic_columns = _reorder_columns(dynamic_columns, list(main_metric), quarter_check, month_check)
#     sorted_columns = static_columns + sorted_dynamic_columns
#     pivot_data = pivot_data[sorted_columns]
#
#     def _sort_condition(col):
#         if sort_by_col != 'value':
#             return (most_recent_time in col) and (main_metric in col.lower()) and (
#                         'yoy' in col.lower() or 'qoq' in col.lower())
#         else:
#             return (most_recent_time in col) and (main_metric in col.lower()) and not (
#                         'yoy' in col.lower() or 'qoq' in col.lower())
#
#     pivot_sort_cols = [col for col in pivot_data.columns if _sort_condition(col) and 'share of spend' not in col.lower()]
#     if pivot_sort_cols and not month_check:
#         pivot_sort_col = pivot_sort_cols[0]
#         pivot_data = pivot_data.sort_values(by=pivot_sort_col, ascending=ascend)
#         if len(dim_col_order) > 1:
#             pivot_data['sort_helper'] = pivot_data.groupby(dim_col).cumcount()
#             pivot_data = pivot_data.sort_values(by = [dim_col, 'sort_helper'])
#             pivot_data = pivot_data.groupby(dim_col).head(5)
#             pivot_data = pivot_data.drop(columns=['sort_helper'])
#
#     def transform_to_mm(df):
#         for col in df.columns:
#             if 'share of spend' not in col and 'YOY' not in col and 'cost per' not in col and 'QOQ' not in col:
#                 # if 'spend' in col or 'response' in col or 'activity' in col:
#                 if 'spend' in col or 'activity' in col:
#                     df[col] = df[col] / 1000000  # Convert to millions
#                     df[col] = df[col].apply(lambda x: f"{x:.2f}" if not pd.isna(x) else x)
#                     df.rename(columns={col: col + ' (MM)'}, inplace=True)
#         return df
#
#     pivot_data = transform_to_mm(pivot_data)
#
#     # yoy add %
#     def append_percentage(df):
#         # Iterate over columns and apply transformation if conditions are met
#         for col in df.columns:
#             if 'YOY' in col or 'share of spend' in col or 'contribution' in col or 'sourceofchange' in col or 'QOQ' in col:
#                 df[col] = pd.to_numeric(df[col], errors='coerce')
#                 df[col] = df[col].apply(lambda x: f"{x:.1f}%" if not pd.isna(x) else x)
#         return df
#
#     pivot_data = append_percentage(pivot_data)
#
#     # sales add $, add ,
#     def format_sales_values(df):
#         for col in df.columns:
#             if 'sales' in col and 'response' not in col and 'YOY' not in col and 'QOQ' not in col:
#                 df[col] = df[col].apply(lambda x: f"${x:,.2f}" if not pd.isna(x) else x)
#         return df
#
#     pivot_data = format_sales_values(pivot_data)
#
#     # column name uppercase
#     def custom_title_case(column_name):
#         words = column_name.split()
#         final_words = [word.upper() if word in ['YOY', '(MM)', 'roi', 'QOQ'] else word.title() for word in words]
#         return ' '.join(final_words)
#
#     pivot_data.columns = [custom_title_case(col) for col in pivot_data.columns]
#     return pivot_data
#
#
# def convert_display_format(text):
#     # Pattern to find numbers with more than 7 digits
#     pattern = r'\b\d{7,}\.?\d*\b'
#
#     # Function to replace each match
#     def replace_with_millions(match):
#         number = float(match.group())
#         # Convert to millions and format to two decimal places with "MM"
#         millions = round(number / 1_000_000, 2)
#         return f"{millions} MM"
#
#     # Replace all found numbers with their 'million' equivalents
#     res = re.sub(pattern, replace_with_millions, text)
#
#     res = re.sub(r' {2,}', ' ', res)
#     res = re.sub(r'\$', '\\$', res)
#     res = re.sub(r'#', '', res)
#
#     return res


def _process_bi_detail(client_code: str, model_group_id: int, df_dict: dict[str, pd.DataFrame], driver_tags: tuple,
                ner_filter: dict, ner_res_dict: dict, map_dict: dict):
    file_mapping_bi = {'tactics': 'ag_df', 'detail_tactics': 'mg_df', 'tactics_detail': 'm_df'}

    file_type, data_type, driver_high, driver_detail = driver_tags

    intention = ner_filter['intention'][0]
    media_channel_check = ner_res_dict.get('media_channel', 'irrelevant')
    media_channel_ls = ner_filter.get('media_channel', [])
    trend_check, rank_check = ner_res_dict.get('trend', 'no'), ner_res_dict.get('rank', 'na')
    how_many = ner_res_dict.get('how_many', 'na')
    product_media_check = 'product_focused_media' in ner_filter.keys()  # True: custom_agg data
    retailer_check, prog_check = ner_filter.get('retailer', 'no'), ner_res_dict.get('programmatic', 'irrelevant')
    condition_extra = retailer_check != 'no' or prog_check != 'irrelevant'
    rank_metric, sort_metric = map_dict[intention]['rankMetric'], map_dict[intention]['sortMetric']
    metrics_order = ner_filter['metric']
    period_type = ner_filter['period_type'][0]
    fixtext = ""
    pivot_sort_col = ""
    core_dimension = ner_filter.get('core_dimension', {})

    # for questions asking about overall spending
    if intention == 'spending' and media_channel_check == 'irrelevant' and not (isinstance(media_channel_ls, list) and len(media_channel_ls)!=0) and not condition_extra and not product_media_check and not core_dimension:
        str_q, pivot_table = readout_utils._overall_metric(client_code, model_group_id, ner_filter, ner_res_dict)
    elif intention == 'planner':
        str_q, fixtext, pivot_table, _, _, _ = readout_utils._planner_intention_res(ner_filter, pd.DataFrame(), 'irrelevant', media_channel_check, media_channel_ls)
    else:
        # read in and clean driver_detail dataframe
        curr_df_name = file_mapping_bi[driver_detail]
        curr_df = df_dict[curr_df_name]
        curr_df = readout_utils._filter_paid_search(intention, curr_df)

        if curr_df.empty:
            raise ValueError('No valid data.')

        data1 = readout_utils._read_and_process_csv_other_category(curr_df, ner_filter)
        overall_spend_condition = intention in ['spending'] and media_channel_check in ['all', 'irrelevant'] and not core_dimension and not condition_extra and not product_media_check
        if overall_spend_condition:
            fixtext, _ = readout_utils._overall_metric(client_code, model_group_id, ner_filter, ner_res_dict)

        if driver_high != 'media_channel':
            data = data1
        else:
            data = readout_utils._concatenate_ag_mg_bi(data1, 'detail_tactics')

        ner_dict = readout_utils._get_ner_key_value_other_category(data, driver_high, file_type, data_type, driver_detail)
        # compute percentage change
        data = readout_utils._calc_growth(period_type, ner_dict, data)
        if data.empty:
            raise ValueError('No valid data.')
        growth_col, sort_by_col = readout_utils._get_growth_sort_col(trend_check, data)
        data = readout_utils._sort_filt_data(data, ner_dict, sort_metric, sort_by_col, rank_check)
        data = readout_utils._ignore_growth_outlier(growth_col, data)
        # generate table display
        pivot_table, pivot_sort_col = readout_utils._get_granular_table(data, ner_dict, metrics_order, ner_filter, rank_check, sort_by_col, pd.DataFrame(), client_code, model_group_id)
        # get top performing tactics data, support user given top number
        top_data = readout_utils._top_data(data, ner_dict, how_many)
        top_data = readout_utils._filter_top_media_channel(driver_high, how_many, ner_filter, ner_dict, sort_by_col, top_data)

        # get planner and benchmark info
        planner_data = readout_utils._merge_planner(intention, media_channel_check, core_dimension,{}, data, pd.DataFrame())
        benchmark_str, benchmark_data = readout_utils._merge_benchmark(intention, media_channel_check, core_dimension,{}, ner_dict, data, top_data, pd.DataFrame())

        str_q = readout_utils._get_readout(intention, growth_col, trend_check, rank_check, sort_metric, how_many,
                                           ner_dict, ner_filter, data, top_data)

    return str_q, pivot_table, benchmark_data, benchmark_str, planner_data, fixtext, pd.DataFrame(), pivot_sort_col

def _process_bi(client_code: str, model_group_id: int, df_dict: dict[str, pd.DataFrame], driver_tags: tuple,
                ner_filter: dict, ner_res_dict: dict, map_dict: dict):
    core_dimension = ner_filter.get('core_dimension', {})
    try:
        str_q, pivot_table, benchmark_data, benchmark_str, planner_data, fixtext, pivot_biz_table, pivot_sort_col = _process_bi_detail(client_code, model_group_id, df_dict, driver_tags, ner_filter, ner_res_dict, map_dict)
    except Exception as e:
        logger.warning(f"Failed to process initial BI details: {e}")
        str_q, pivot_table, benchmark_data, benchmark_str, planner_data, fixtext, pivot_biz_table, pivot_sort_col = "", pd.DataFrame(), pd.DataFrame(), "", pd.DataFrame(), "", pd.DataFrame(), ""
    if core_dimension:
        search_level = readout_utils._check_key_or_value(core_dimension, ['activity_group', 'measure_group', 'measure'])
        if search_level['measure'] and search_level['activity_group']:
            driver_tags_ag = 'ag', 'bi', 'tactics', 'tactics'
            try:
                str_q_ag, pivot_table_ag, _, _, _, fixtext_ag, pivot_biz_table_ag, _ = _process_bi_detail(client_code, model_group_id, df_dict, driver_tags_ag, ner_filter, ner_res_dict, map_dict)
            except Exception as e:
                logger.warning(f"Failed to process activitygroup level: {e}")
                str_q_ag, pivot_table_ag, fixtext_ag, pivot_biz_table_ag = "", pd.DataFrame(), "", pd.DataFrame()
            pivot_biz_table = pivot_table_ag
            str_q = str_q_ag + 'At Tactics level: ' + str_q
        elif not search_level['measure'] and search_level['activity_group']:
            driver_tags_m = 'mg', 'bi', 'media_channel', 'detail_tactics'
            try:
                str_q_m, pivot_table_m, _, _, _, fixtext_m, pivot_biz_table_m, _ = _process_bi_detail(client_code, model_group_id, df_dict, driver_tags_m, ner_filter, ner_res_dict, map_dict)
            except Exception as e:
                logger.warning(f"Failed to process activitygroup level: {e}")
                str_q_m, pivot_table_m, fixtext_m, pivot_biz_table_m = "", pd.DataFrame(), "", pd.DataFrame()
            if not pivot_table_m.empty:
                pivot_biz_table = pivot_table
                pivot_table = pivot_table_m
            str_q = str_q + 'At Tactics level: ' + str_q_m

    pivot_table = readout_utils._dynamic_top(pivot_sort_col, pivot_table)
    dim_col = pivot_table.columns[0] if not pivot_table.empty else ""
    dim_col_biz = pivot_biz_table.columns[0] if not pivot_biz_table.empty else ""
    pivot_table = readout_utils._drop_single_dim(pivot_table)
    pivot_table = readout_utils._rename_product_media_other_category(ner_filter, dim_col, pivot_table)
    pivot_biz_table = readout_utils._dynamic_top(pivot_sort_col, pivot_biz_table)
    pivot_biz_table = readout_utils._drop_single_dim(pivot_biz_table)
    pivot_biz_table = readout_utils._rename_product_media_other_category(ner_filter, dim_col_biz, pivot_biz_table)

    return str_q, pivot_table, benchmark_data, benchmark_str, planner_data, fixtext, pivot_biz_table

# def _overall_metric(client_code, model_group_id, ner_filter, ner_res_dict):
#     driver_tags = _get_driver_tag(ner_filter, ner_res_dict)
#     file_type, data_type, driver_high, driver_detail = driver_tags
#     trend_check, rank_check = ner_res_dict.get('trend', 'no'), ner_res_dict.get('rank', 'na')
#     product_media_check = 'product_focused_media' in ner_filter.keys()
#     metric = ner_filter['main_metric']
#     data = load_total_data(client_code, model_group_id)
#     if data.columns[0] not in ['product', 'category']:
#         data['product'] = 'overall'
#     else:
#         data=data.rename(columns={'product':'channel'})
#     if 'product' in data.columns:
#         cols = ['product'] + [col for col in data.columns if col != 'product']
#         data = data[cols]
#     ner_dict = _get_ner_key_value(data, driver_high, file_type, data_type, driver_detail)
#     if product_media_check:
#         ner_dict['dimension'] = 'channel'
#     metric_col, dim_col, time_col, value_col, period_col = ner_dict['metric'], ner_dict['dimension'], ner_dict[
#         'time'], ner_dict['value'], ner_dict['period']
#     period_type = ner_filter.get('period_type', ['fiscal year'])[0]
#     if 'product' in ner_filter:
#         product_check = ner_filter['product']
#     elif 'category' in ner_filter:
#         product_check = ner_filter['category']
#     else:
#         product_check = ['overall']
#     flat = False
#
#     data = data[(data[metric_col].str.contains(metric, case=False))&(data[dim_col].isin(product_check))]
#     data = data[data[time_col].isin(ner_filter['time'])]
#
#     data = data.sort_values(by=[dim_col, metric_col, period_col, time_col])
#     if 'quarter' in period_type:
#         data = _convert_quarter(data, time_col)
#         data = _calc_qoq_total(data, ner_dict)
#     elif 'year' in period_type:
#         data = _convert_time(data, time_col)
#         data = _calc_yoy_total(data, ner_dict)
#
#     growth_col = next((col for col in ['yoy_percent_change', 'qoq_percent_change'] if col in data.columns), None)
#     sort_by_col = growth_col if (growth_col and trend_check == 'yes') else 'value'
#     if growth_col in data.columns:
#         growth_values = data[growth_col].dropna()
#         if not growth_values.empty and (abs(growth_values)<0.5).all():
#             data = data.drop(growth_col, axis=1)
#             flat = True
#     str_q = _get_total_level_pivot(data, ner_dict, ner_filter)
#     if flat:
#         str_q = str_q.rstrip() + "The value changes over time remain flat.\n"
#     pivot_table = _get_total_table(data, ner_dict, ner_filter, rank_check, sort_by_col)
#     if pivot_table.empty == False and pivot_table[ner_dict['dimension'].title()].unique().tolist() == ['Overall']:
#         pivot_table = pivot_table.drop(ner_dict['dimension'].title(), axis=1)
#     return str_q, pivot_table


def _process_br(client_code: str, model_group_id: int, df_dict, driver_tags, ner_filter, ner_res_dict, map_dict):
    fixtext = ""
    file_mapping_br = {'business_driver_detail': 'ag_df', 'tactics': 'mg_df', 'detail_tactics': 'm_df'}
    intention, biz_driver_check = ner_res_dict['intention'][0], ner_res_dict.get('business_driver', 'irrelevant')
    biz_driver_ls = ner_filter.get('business_driver', [])
    trend_check, rank_check = ner_res_dict.get('trend', 'no'), ner_res_dict.get('rank', 'na')
    how_many = ner_res_dict.get('how_many', 'na')
    product_media_check = 'product_focused_media' in ner_filter.keys()
    retailer_check, prog_check = ner_filter.get('retailer', 'no'), ner_res_dict.get('programmatic', 'irrelevant')
    condition_extra = retailer_check != 'no' or prog_check != 'irrelevant'
    period_type = ner_filter['period_type'][0]
    if intention != 'sales' and biz_driver_check == 'all':
        file_type, data_type, driver_high, driver_detail = 'mg', 'br', 'business_driver', 'tactics'
    else:
        file_type, data_type, driver_high, driver_detail = driver_tags
    rank_metric, sort_metric = map_dict[intention]['rankMetric'], map_dict[intention]['sortMetric']
    media_channel_check = ner_res_dict.get('media_channel', 'irrelevant')
    metrics_order = ner_filter['metric']
    pivot_table_biz = pd.DataFrame()
    core_dimension = ner_filter.get('core_dimension', {})

    if intention == 'sales' and biz_driver_check == 'irrelevant' and not (isinstance(biz_driver_ls, list) and len(biz_driver_ls)!=0) and not condition_extra and not core_dimension and not product_media_check:
        str_q, pivot_table = readout_utils._overall_metric(client_code, model_group_id, ner_filter, ner_res_dict)
    else:
        curr_df_name_biz = file_mapping_br['business_driver_detail']
        curr_df_biz = df_dict[curr_df_name_biz]
        data1 = readout_utils._read_and_process_csv_other_category(curr_df_biz, ner_filter)

        curr_df_name = file_mapping_br[driver_detail]
        curr_df = df_dict[curr_df_name]
        curr_df = readout_utils._empty_mg_replace_other_category(df_dict, file_mapping_br, curr_df_name, curr_df)
        if curr_df.empty:
            raise ValueError('No valid data.')
        overall_sales_condition = intention in ['sales'] and biz_driver_check in ['all', 'irrelevant'] and not condition_extra
        if overall_sales_condition:
            fixtext, _ = readout_utils._overall_metric(client_code, model_group_id, ner_filter, ner_res_dict)

        data2 = readout_utils._read_and_process_csv_other_category(curr_df, ner_filter)
        ner_dict = readout_utils._get_ner_key_value_other_category(data2, driver_high, file_type, data_type, driver_detail)
        ner_dict1 = readout_utils._get_ner_key_value_other_category(data1, driver_high, 'ag', 'br', 'business_driver_detail')

        if intention == 'contribution' and 'other' in curr_df['business_driver'].unique().tolist() and biz_driver_check != 'all':
            fixtext += 'We only provide contribution by marketing drivers. '
        data1 = readout_utils._filter_media_promotion(intention, biz_driver_check, media_channel_check, data1)
        data2 = readout_utils._filter_media_promotion(intention, biz_driver_check, media_channel_check, data2)
        if data2.empty:
            raise ValueError('No valid data.')

        data1 = readout_utils._calc_growth_br(intention, ner_res_dict, period_type, ner_dict1, data1)
        data2 = readout_utils._calc_growth_br(intention, ner_res_dict, period_type, ner_dict, data2)
        data1 = readout_utils._exclude_system(intention, data1)
        data2 = readout_utils._exclude_system(intention, data2)
        growth_col, sort_by_col = readout_utils._get_growth_sort_col(trend_check, data2)
        data1 = readout_utils._ignore_growth_outlier(growth_col, data1)
        data2 = readout_utils._ignore_growth_outlier(growth_col, data2)
        data2 = readout_utils._sort_filt_data(data2, ner_dict, sort_metric, sort_by_col, rank_check)

        top_data = readout_utils._top_data(data2, ner_dict, how_many)

        fixtext = readout_utils._get_fixtext_br(intention, biz_driver_check, media_channel_check, ner_dict1, ner_filter, trend_check, driver_detail, curr_df, data1)
        str_q = readout_utils._get_readout(intention, growth_col, trend_check, rank_check, sort_metric, how_many, ner_dict, ner_filter, data2, top_data)
        str_q = readout_utils._get_readout_br(intention, driver_detail, str_q)

        pivot_table, pivot_sort_col = readout_utils._get_granular_table(data2, ner_dict, metrics_order, ner_filter, rank_check, sort_by_col, pd.DataFrame(), client_code, model_group_id)
        pivot_table_biz = readout_utils._get_pivot_biz_table(intention, biz_driver_check, media_channel_check, ner_dict1, ner_filter, rank_check, sort_by_col, metrics_order, client_code, data1)

        str_q, fixtext = readout_utils._colgate_specific_soc(intention, ner_dict1, ner_filter, trend_check, data1, pivot_table_biz, str_q, fixtext)
        if intention == 'source of change':
            ner_filter_copy = ner_filter.copy()
            ner_filter_copy['main_metric'] = 'sales'
            str_q_overall_sales, _ = readout_utils._overall_metric(client_code, model_group_id, ner_filter_copy, ner_res_dict)
            str_q += str_q_overall_sales

        pivot_table = readout_utils._drop_single_dim(pivot_table)
        pivot_table_biz = readout_utils._drop_single_dim(pivot_table_biz)
        pivot_table, pivot_table_biz = readout_utils._br_table_reformat(pivot_table, pivot_table_biz)

    return str_q, pivot_table, pd.DataFrame(), "", pd.DataFrame(), fixtext, pivot_table_biz

# def process_data(client_code: str, model_group_id: int, readout_data: ReadoutData):
#     ner_filter_dict = readout_data.ner_filters
#     ner_res_dict = readout_data.ner_results
#     df_dict = {
#         'ag_df': readout_data.df_activity_group,
#         'mg_df': readout_data.df_measure_group,
#         'm_df': readout_data.df_measure,
#     }
#     driver_tags = _get_driver_tag(ner_filter_dict, ner_res_dict)
#     data_type = driver_tags[1]
#     map_dict = load_map_dict(client_code, model_group_id)
#     if data_type == 'bi':
#         return _process_bi(client_code, model_group_id, df_dict, driver_tags, ner_filter_dict, ner_res_dict, map_dict)
#     else:
#         return _process_br(client_code, model_group_id, df_dict, driver_tags, ner_filter_dict, ner_res_dict, map_dict)

# def response_generate(query: str, readout: str) -> str:
#     prompt = prompt_template.build_readout_prompt(query, readout)
#     text = generate_text(prompt)
#     return text
#
# def stream_response(query: str, readout: str) -> Iterator[str]:
#     prompt = prompt_template.build_readout_prompt(query, readout)
#     it = stream_text(prompt)
#     return it
