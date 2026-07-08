import re
from typing import Optional, Dict, Iterator
import numpy as np
import pandas as pd
import json
import logging
import os
import random
from calendar import month_name
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.stats import spearmanr
from src.data.data_interface import ReadoutData
from src.data.local import get_data_path
from src.integrations.llm import generate_text
from src.model.common import PromptTemplates
from src.model.common import tokenize_text
from datetime import datetime
from scipy.stats import mannwhitneyu

logger = logging.getLogger(__name__)

def load_activity_type_data(client_code: str, model_group_id: int) -> pd.DataFrame:
    file_path = get_data_path('activity_type', 'activity_type.csv')
    try:
        return pd.read_csv(file_path, low_memory=False)
    except FileNotFoundError:
        return pd.DataFrame()

def load_total_viz_data(client_code: str, model_group_id: int) -> pd.DataFrame:
    file_path = get_data_path('total_viz', 'total_viz.csv')
    try:
        return pd.read_csv(file_path, low_memory=False)
    except FileNotFoundError:
        return pd.DataFrame()

def load_example_roi_spend_data(client_code: str, model_group_id: int) -> pd.DataFrame:
    file_path = get_data_path('insight_example', 'insight_example.csv')
    try:
        return pd.read_csv(file_path, low_memory=False)
    except FileNotFoundError:
        return pd.DataFrame()

def load_example_roi_spend_trend_data(client_code: str, model_group_id: int) -> pd.DataFrame:
    file_path = get_data_path('insight_example_trend', 'insight_example_trend.csv')
    try:
        return pd.read_csv(file_path, low_memory=False)
    except FileNotFoundError:
        return pd.DataFrame()

def load_example_other_data(client_code: str, model_group_id: int) -> pd.DataFrame:
    file_path = get_data_path('insight_example_other', 'insight_example_other.csv')
    try:
        return pd.read_csv(file_path, low_memory=False)
    except FileNotFoundError:
        return pd.DataFrame()


def load_total_data(client_code: str, model_group_id: int) -> pd.DataFrame:
    return pd.read_csv(get_data_path("total"))


def load_planner_data(client_code: str, model_group_id: int) -> pd.DataFrame:
    try:
        return pd.read_csv(get_data_path("planner"))
    except FileNotFoundError:
        return pd.DataFrame()


def load_diminishing_returns_data(client_code: str, model_group_id: int) -> (pd.DataFrame, pd.DataFrame):
    try:
        return pd.read_csv(get_data_path("dim_returns_curve")), pd.read_csv(get_data_path("scenario_summary"))
    except FileNotFoundError:
        return pd.DataFrame(), pd.DataFrame()


def load_map_dict(client_code: str, model_group_id: int) -> Dict:
    df_map_dict = pd.read_csv(get_data_path("map_dict"))
    map_dict = df_map_dict.set_index('intentionName').to_dict(orient='index')

    return map_dict


def load_metric_postprocess(client_code: str, model_group_id: int) -> pd.DataFrame:
    file_path = get_data_path('metric_postprocess', 'metric_postprocess.csv')
    try:
        return pd.read_csv(file_path, low_memory=False)
    except FileNotFoundError:
        return pd.DataFrame()

def load_result_lookup_bi(client_code: str, model_group_id: int) -> pd.DataFrame:
    file_path = get_data_path('result_lookup_bi', 'result_lookup_bi.csv')
    try:
        return pd.read_csv(file_path, low_memory=False)
    except FileNotFoundError:
        return pd.DataFrame()
def load_result_lookup_br(client_code: str, model_group_id: int) -> pd.DataFrame:
    file_path = get_data_path('result_lookup_br', 'result_lookup_br.csv')
    try:
        return pd.read_csv(file_path, low_memory=False)
    except FileNotFoundError:
        return pd.DataFrame()

def load_agg_view_logic(client_code: str, model_group_id: int) -> pd.DataFrame:
    setting_path = get_data_path('agg_view_setting', 'agg_view_setting.csv')
    mapping_path = get_data_path('agg_view_mapping', 'agg_view_mapping.csv')
    if os.path.exists(setting_path) and os.path.exists(mapping_path):
        agg_view = {
            'setting': pd.read_csv(setting_path),
            'mapping': pd.read_csv(mapping_path)
        }
    else:
        agg_view = None
    return agg_view

def load_tactics_level(client_code: str, model_group_id: int) -> pd.DataFrame:
    file_path = get_data_path('tactics_level', 'tactics_level.csv')
    try:
        return pd.read_csv(file_path, low_memory=False)
    except FileNotFoundError:
        return pd.DataFrame()

def load_benchmark_data(client_code: str, model_group_id: int) -> pd.DataFrame:
    file_path = get_data_path("benchmark")
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        df = df.rename(columns={'channel': 'media_channel', 'benchmarkIndex': 'benchmark roi index'})
        df['media_channel'] = df['media_channel'].str.lower()
        df['benchmark roi index'] = df['benchmark roi index'].astype(int)
        return df
    else:
        return pd.DataFrame()

def load_planner_filter(client_code: str, model_group_id: int) -> pd.DataFrame:
    file_path = get_data_path('planner_filter', 'planner_filter.csv')
    try:
        df = pd.read_csv(file_path, low_memory=False)
        return df
    except FileNotFoundError:
        return pd.DataFrame()


def load_prompt_instruction(client_code: str, model_group_id: int) -> pd.DataFrame:
    file_path = get_data_path('prompt_instruction', 'prompt_instruction.csv')
    if os.path.exists(file_path):
        df = pd.read_csv(file_path, low_memory=False)
        return df
    else:
        return pd.DataFrame()


def load_planner_principle(client_code: str, model_group_id: int) -> pd.DataFrame:
    file_path = get_data_path('planner_principle', 'planner_principle.csv')
    if os.path.exists(file_path):
        df = pd.read_csv(file_path, low_memory=False)
        return df
    else:
        return pd.DataFrame()

def load_genome_principle(client_code: str, model_group_id: int) -> pd.DataFrame:
    file_path = get_data_path('genome_principle', 'genome_principle.csv')
    if os.path.exists(file_path):
        df = pd.read_csv(file_path, low_memory=False)
        return df
    else:
        return pd.DataFrame()

def load_driver_curves_data(client_code: str, model_group_id: int):
    file_path = get_data_path('response_curve_bychannel', 'response_curve_bychannel.json')
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return []


def load_level_rename_display(client_code: str, model_group_id: int) -> pd.DataFrame:
    file_path = get_data_path('level_rename_display', 'level_rename_display.csv')
    if os.path.exists(file_path):
        df = pd.read_csv(file_path, low_memory=False)
        return df
    else:
        return pd.DataFrame()


def load_unit_sign_config() -> pd.DataFrame:
    file_path = get_data_path('unit_sign_config', 'unit_sign_config.csv')
    if os.path.exists(file_path):
        df = pd.read_csv(file_path, low_memory=False)
        return df
    else:
        return pd.DataFrame()


def load_insight_instructions(client_code: str, model_group_id: int, intention: str) -> pd.DataFrame:
    file_path = get_data_path('insight_instruction', 'insight_instruction.csv')
    if os.path.exists(file_path):
        df = pd.read_csv(file_path, low_memory=False)
        try:
            res = df[df["intentionName"] == intention]
        except:
            res = pd.DataFrame()
    else:
        res = pd.DataFrame()
    return res


def _get_sort_order(metric_df: pd.DataFrame, metric: str, rank_check: str) -> bool:
    if metric:
        metric_ascend = metric_df.loc[metric_df['metric'] == metric, 'ascend'].head(1).iloc[0] if not metric_df.loc[
            metric_df['metric'] == metric, 'ascend'].empty else False

        if rank_check == 'bottom' and metric_ascend:
            ascend = False
        elif rank_check == 'bottom' or metric_ascend:
            ascend = True
        else:
            ascend = False
    else:
        ascend = False
    return ascend


def _check_key_or_value(core_dim, search_terms):
    found = {term: False for term in search_terms}

    def recursive_search(core_dim):
        if isinstance(core_dim, dict):
            for key, value in core_dim.items():
                if key in search_terms:
                    found[key] = True
                if isinstance(value, (dict, list)):
                    recursive_search(value)
        elif isinstance(core_dim, list):
            for item in core_dim:
                recursive_search(item)

    recursive_search(core_dim)
    return found


def find_top_level_empty_keys(core_dim):
    return [key for key, value in core_dim.items() if value == []]


def _get_driver_tag_COLGUS_tp(ner_filter_dict, ner_res_dict):
    '''
    :param ner_filter_dict:
    :param ner_res_dict:
    :return: filtered data, bi/br data, driver_high, driver_detail
    '''
    data_type = ner_filter_dict['data']
    business_driver = ner_res_dict.get('business_driver', 'irrelevant')
    media_channel = ner_res_dict.get('media_channel', 'irrelevant')
    media_channel_ls = ner_filter_dict.get('media_channel', [])
    core_dimension = ner_filter_dict.get('core_dimension', {})
    search_terms = ['activity_group', 'measure_group', 'measure']
    search_res = _check_key_or_value(core_dimension, search_terms)
    product_halo = ner_res_dict.get('product_halo', 'irrelevant')

    if data_type == 'bi':
        if media_channel in ['all', 'irrelevant'] and not core_dimension and product_halo == 'irrelevant':
            driver_high = 'tactics'
        elif media_channel != 'irrelevant' and isinstance(media_channel_ls, list) and len(
                media_channel_ls) >= 1 and product_halo == 'irrelevant':
            if search_res.get('measure', False):
                driver_high = 'detail_tactics'
            elif search_res.get('measure_group', False):
                driver_high = 'tactics_detail'
            else:
                driver_high = 'media_channel'
        else:
            driver_high = 'media_channel'
        file_type_mapping = {'media_channel': 'mg', 'tactics': 'ag', 'tactics_detail': 'mg', 'detail_tactics': 'm'}
        file_type = file_type_mapping[driver_high]
        driver_detail_map = {'ag': 'tactics', 'mg': 'tactics_detail', 'm': 'detail_tactics'}
        driver_detail = driver_detail_map[file_type]
    else:
        driver_high = 'business_driver'
        if business_driver == 'all':
            file_type = 'mg'
            driver_detail = 'business_driver_detail'
        else:
            file_type = 'm'
            driver_detail = 'tactics'

    return file_type, data_type, driver_high, driver_detail


def _get_driver_tag_COLGUS_other_category(ner_filter_dict, ner_res_dict):
    '''
    :param ner_filter_dict:
    :param ner_res_dict:
    :return: filtered data, bi/br data, driver_high, driver_detail
    '''
    data_type = ner_filter_dict['data']
    business_driver = ner_res_dict.get('business_driver', 'irrelevant')
    media_channel = ner_res_dict.get('media_channel', 'irrelevant')
    media_channel_ls = ner_filter_dict.get('media_channel', [])
    custom_agg = ner_filter_dict.get('product_focused_media', ['no'])
    core_dimension = ner_filter_dict.get('core_dimension', {})
    search_terms = ['activity_group', 'measure_group', 'measure']
    search_res = _check_key_or_value(core_dimension, search_terms)

    if data_type == 'bi':
        if custom_agg != ['no']:
            driver_high = 'detail_tactics'
        elif media_channel in ['all', 'irrelevant'] and not core_dimension:
            driver_high = 'tactics'
        elif isinstance(media_channel_ls, list) and len(media_channel_ls) >= 1:
            if search_res.get('measure', False):
                driver_high = 'detail_tactics'
            elif search_res.get('activity_group', False):
                driver_high = 'tactics'
            else:
                driver_high = 'media_channel'
        else:
            driver_high = 'media_channel'
        file_type = 'ag' if driver_high == 'tactics' else 'mg'
        driver_detail_map = {'ag': 'tactics', 'mg': 'detail_tactics', 'm': 'tactics_detail'}
        driver_detail = driver_detail_map[file_type]
    else:
        driver_high = 'business_driver'
        if custom_agg != ['no']:
            file_type = 'm'
            driver_detail = 'detail_tactics'
        elif business_driver == 'all':
            file_type = 'ag'
            driver_detail = 'business_driver_detail'
        elif core_dimension:
            if search_res.get('measure', False):
                file_type = 'm'
                driver_detail = 'detail_tactics'
            else:
                file_type = 'mg'
                driver_detail = 'tactics'
        else:
            file_type = 'mg'
            driver_detail = 'tactics'
    return file_type, data_type, driver_high, driver_detail


def _get_driver_tag_FTR(ner_filter_dict, ner_res_dict):
    '''
    :param ner_filter_dict:
    :param ner_res_dict:
    :return: filtered data, bi/br data, driver_high, driver_detail
    '''
    data_type = ner_filter_dict['data']
    # business_driver = ner_res_dict.get('business_driver', 'irrelevant')
    # media_channel = ner_res_dict.get('media_channel', 'irrelevant')
    # media_channel_ls = ner_filter_dict.get('media_channel', [])
    core_dimension = ner_filter_dict.get('core_dimension', {})
    search_terms = ['activity_group', 'measure_group', 'measure']
    search_res = _check_key_or_value(core_dimension, search_terms)

    if data_type == 'bi':
        if not core_dimension:
            driver_high = 'tactics'
        else:
            if search_res.get('measure', False):
                driver_high = 'detail_tactics'
            elif search_res.get('measure_group', False):
                driver_high = 'tactics_detail'
            else:
                driver_high = 'tactics'
        file_type_mapping = {'tactics': 'ag', 'tactics_detail': 'mg', 'detail_tactics': 'm'}
        file_type = file_type_mapping[driver_high]
        # driver_detail_map = {'ag': 'tactics', 'mg': 'tactics_detail', 'm': 'detail_tactics'}
        # driver_detail = driver_detail_map[file_type]
        driver_detail = driver_high
    else:
        if not core_dimension:
            driver_detail, driver_high = 'tactics', 'business_driver_detail'
        else:
            if (search_res.get('measure_group', False) and search_res.get('activity_group', False)) or search_res.get(
                    'activity_group', False):
                driver_detail, driver_high = 'tactics_detail', 'tactics'
            elif search_res.get('measure_group', False):
                driver_detail, driver_high = 'tactics_detail', 'tactics_detail'
            # elif search_res.get('activity_group', False):
            #     driver_detail, driver_high = 'tactics_detail', 'tactics'
            else:
                driver_detail, driver_high = 'tactics', 'tactics'
        file_type_mapping = {'tactics': 'mg', 'tactics_detail': 'm', 'business_driver_detail': 'ag'}
        file_type = file_type_mapping[driver_detail]

    return file_type, data_type, driver_high, driver_detail


def _get_driver_tag_default(ner_filter_dict, ner_res_dict):
    data_type = ner_filter_dict['data']
    if data_type == 'bi':
        return "ag", "bi", "tactics", "tactics"
    else:
        return "m", "br", "tactics", "tactics"


def _get_driver_tag_HILSP(ner_filter_dict, ner_res_dict):
    data_type = ner_filter_dict['data']
    core_dimension = ner_filter_dict.get('core_dimension', {})
    search_terms = ['activity_group', 'measure_group', 'measure']
    search_res = _check_key_or_value(core_dimension, search_terms)
    mg_tagging = ner_res_dict.get('brand_focused_marketing', 'irrelevant')
    mg_m_tagging = ner_res_dict.get('campaign', 'irrelevant')
    response_mod = ner_filter_dict.get('response_modifier', [])

    if data_type == 'bi':
        if 'platform' in response_mod:
            driver_detail, driver_high = 'detail_tactics', 'detail_tactics'
        elif (mg_tagging != 'irrelevant' or mg_m_tagging != 'irrelevant') and not core_dimension:
            driver_detail, driver_high = 'tactics_detail', 'tactics_concat'
        elif not core_dimension:
            driver_detail, driver_high = 'tactics', 'tactics'
        else:
            if search_res.get('measure_group', False):
                driver_detail, driver_high = 'tactics_detail', 'tactics_concat'
            elif search_res.get('activity_group', False):
                driver_detail, driver_high = 'tactics', 'tactics'
            else:
                driver_detail, driver_high = 'detail_tactics', 'detail_tactics'
        file_type_mapping = {'tactics': 'ag', 'tactics_detail': 'mg', 'detail_tactics': 'm'}
        file_type = file_type_mapping[driver_detail]
    else:
        if not core_dimension:
            driver_high, driver_detail = 'business_driver_detail', 'tactics'
        else:
            # if search_res.get('activity_group', False):
            #     driver_detail, driver_high = 'tactics', 'tactics'
            # else:
            driver_detail, driver_high = 'tactics', 'tactics'
        file_type_mapping = {'tactics': 'm', 'business_driver': 'ag', 'business_driver_detail': 'mg'}
        file_type = file_type_mapping[driver_detail]

    return file_type, data_type, driver_high, driver_detail


def _drop_col_dict(filt_data, ner_res_dict):
    '''
    :param filt_data: ner filtered data
    :param ner_res_dict:
    :return: data without redundant columns
    '''
    drop_cols = [key for key, value in ner_res_dict.items() if isinstance(value, float) and np.isnan(value)]
    drop_cols = [c for c in drop_cols if c != 'activity_group']
    empty_cols = filt_data.columns[filt_data.isna().all(axis=0)].tolist()
    empty_cols = [c for c in empty_cols if c not in ['business_driver', 'business_driver_detail']]
    drop = drop_cols + empty_cols + ['clientmetric', 'record_level']
    data = filt_data.drop(columns=drop, errors='ignore')
    return data


def _rename_data(data):
    data = data.rename(
        columns={'activity_group': 'tactics', 'measure_group': 'tactics_detail', 'measure': 'detail_tactics',
                 'activity_group_origin': 'tactics_origin',
                 'measure_group_origin': 'tactics_detail_origin',
                 'measure_origin': 'detail_tactics_origin'}, errors='ignore')
    return data


def _convert_time(data, time_col):
    # fiscal year 2022 --> 2022
    data[time_col] = pd.to_numeric(data[time_col].str.extract(r'(\d{4})')[0])
    return data


def _convert_quarter(data, time_col):
    # fiscal quarter 1 2022 --> Q1 2022
    data[time_col] = data[time_col].apply(lambda x: re.sub(r'(?:fiscal )?quarter (\d) (\d{4})', r'Q\1 \2', x))
    return data


def _convert_month(data, time_col):
    data[time_col] = data[time_col].apply(lambda x: re.sub(r'(?:fiscal )?month (\d{1,2}) (\d{4})', r'month \1 \2', x))
    return data


def _convert_half(data, time_col):
    data[time_col] = data[time_col].apply(lambda x: re.sub(r'(?:fiscal )?half (\d) (\d{4})', r'h\1 \2', x))
    return data


def _read_and_process_csv(data, ner_filter):
    '''
    :param data: ner filtered dataframe
    :param ner_filter:
    :return: cleaned dataframe
    '''
    data = _drop_col_dict(data, ner_filter)
    data = _rename_data(data)
    period_type = ner_filter.get('period_type', ['year'])[0]
    if not data.empty:
        if 'year' in period_type:
            data = _convert_time(data, 'time')
        elif 'quarter' in period_type:
            data = _convert_quarter(data, 'time')
        elif 'month' in period_type:
            data = _convert_month(data, 'time')
        elif 'half' in period_type:
            data = _convert_half(data, 'time')
        data = data.replace([np.inf, -np.inf], np.nan)
        data = data.dropna(how='any', axis=0, subset=['value'])
        if 'marketing_channel' in data.columns and 'media_channel' not in data.columns:
            data = data.rename(columns = {'marketing_channel':'media_channel'})
        if not 'media_channel' in data.columns and 'tactics' in data.columns:
            data['media_channel'] = data['tactics']
    return data


def _read_and_process_csv_other_category(data, ner_filter, dim_cols):
    '''
    :param data: ner filtered dataframe
    :param ner_filter:
    :return: cleaned dataframe
    '''
    data = _drop_col_dict(data, ner_filter)
    data = _rename_data(data)
    period_type = ner_filter.get('period_type', ['year'])[0]
    custom_agg = ner_filter.get('product_focused_media', ['no'])
    if len(data) != 0:
        if 'year' in period_type:
            data = _convert_time(data, 'time')
        elif 'quarter' in period_type:
            data = _convert_quarter(data, 'time')
    else:
        return data, dim_cols
    if data.columns[0] not in ['product', 'category']:
        data['product'] = 'overall'
        dim_cols = ['product']
    else:
        if custom_agg != ['no']:
            prod_ls = data['product'].unique().tolist()
            if 'product_focused_media' in data.columns:
                if not data['product_focused_media'].isnull().any():
                    if len(prod_ls) > 1 and 'overall' in prod_ls:  # overall as default
                        data = data[data['product'] == 'overall']
                        data['product'] = data['product_focused_media']
                    elif len(prod_ls) > 1 and 'overall' not in prod_ls:
                        data['product'] = data['product'] + ' - ' + data['product_focused_media']
                    else:
                        data['product'] = data['product_focused_media']
        # else:
        #     data = data.rename(columns={'product': 'channel'})
        #     dim_cols = ['channel']

    if 'product' in data.columns:
        cols = ['product'] + [col for col in data.columns if col != 'product']
        data = data[cols]
    return data, dim_cols


def _concatenate_ag_mg_bi(mg_df, concat_col='tactics_detail'):
    '''
    :param ag_df:
    :param mg_df:
    :return: concatenate AG and MG values
    '''
    data = mg_df
    concat_col_ori = concat_col + '_origin'
    data['tactics_concat'] = data['tactics'] + " (" + data[concat_col] + ")"
    data['tactics_concat_origin'] = data['tactics_origin'] + " (" + data[concat_col_ori] + ")"
    return data

def _concatenate_br(data, driver_detail):
    driver_col_idx = data.columns.get_loc(driver_detail)
    concat_col_name = data.columns[driver_col_idx - 1]
    concat_dict = data.groupby(driver_detail)[concat_col_name].apply(list).to_dict()
    if concat_dict and any(len(v) > 1 for v in concat_dict.values()) and driver_detail in ['tactics_detail']:
        concat_col = driver_detail
        data = _concatenate_ag_mg_bi(data, concat_col)
        driver_detail = 'tactics_concat'
    return data, driver_detail

def _get_dimension_cols(data, idx_col = 'tactics'):
    dim_cols_idx = data.columns.get_loc(idx_col)
    dim_cols = [col for col in data.columns[:dim_cols_idx] if col not in {'business_driver', 'business_driver_detail'}]
    if not dim_cols:
        dim_cols = data.columns[0]
    return dim_cols

def _get_dimension_cols_unique(data, client_code, ner_filter, idx_col = 'tactics'):
    dim_cols_idx = data.columns.get_loc(idx_col)
    dim_cols = [col for col in data.columns[:dim_cols_idx] if col not in {'business_driver', 'business_driver_detail'}]
    if client_code=='LINKEDIN':
        halo_level = ner_filter.get('level', {}).get('halo_tagging', [])
        if halo_level and halo_level[0] in data.columns:
            same_values = data[halo_level[0]].fillna("__NA__").equals(data['business_unit_for_kpi'].fillna("__NA__"))
            if not same_values:
                dim_cols.append(halo_level[0])
        dim_cols = [col for col in dim_cols if not data[col].dropna().eq('overall').all()]

    if not dim_cols:
        dim_cols = [data.columns[0]]

    return dim_cols


def _get_ner_key_value(data, driver_high, file_type, data_type, driver_detail, dim_cols):
    '''
    :param data:
    :param driver_high:
    :param file_type:
    :param data_type:
    :param driver_detail:
    :return: dict, answer level mapping, answering at which driver level
    '''
    # dim_cols = _get_dimension_cols(data)
    ner_dict = {'dimension': dim_cols, 'time': 'time', 'metric': 'metric', 'driver_high': driver_high,
                'driver_detail': driver_detail}
    if data_type == 'bi':
        ner_dict['driver'] = driver_detail if driver_high == driver_detail else 'tactics_concat'
    else:
        ner_dict['driver'] = driver_detail
    ner_dict['value'] = 'value'
    ner_dict['period'] = 'period_type'
    return ner_dict


def _get_ner_key_value_other_category(data, driver_high, file_type, data_type, driver_detail):
    '''
    :param data:
    :param driver_high:
    :param file_type:
    :param data_type:
    :param driver_detail:
    :return: dict, answer level mapping, answering at which driver level
    '''
    dim_col = [data.columns[0]]
    ner_dict = {'dimension': dim_col, 'time': 'time', 'metric': 'metric', 'driver_high': driver_high,
                'driver_detail': driver_detail}
    if data_type == 'bi':
        file_driver_map = {'ag': 'tactics', 'mg': 'tactics_concat', 'm': 'tactics_detail'}
        if driver_high == 'detail_tactics' and driver_detail == 'detail_tactics':
            file_driver_map['mg'] = 'detail_tactics'
        ner_dict['driver'] = file_driver_map[file_type]
    else:
        file_driver_map_br = {'ag': 'business_driver_detail', 'mg': 'tactics', 'm': 'detail_tactics'}
        ner_dict['driver'] = file_driver_map_br[file_type]
    ner_dict['value'] = 'value'
    ner_dict['period'] = 'period_type'
    return ner_dict


def _calc_yoy(data, ner_dict):
    '''
    :param data: cleaned dataframe
    :param ner_dict:
    :return: dataframe with YOY percentage change
    '''
    if not data.empty:
        dimension, time, driver, value, metric = ner_dict['dimension'], ner_dict['time'], ner_dict['driver'], ner_dict[
            'value'], ner_dict['metric']
        data[value] = pd.to_numeric(data[value], errors='coerce')
        if data[time].nunique()<2:
            return data
        sort_by_cols = dimension + [metric, driver, time]
        group_by_cols = dimension + [metric, driver]
        data.sort_values(by=sort_by_cols, inplace=True)
        data['value_prev'] = data.groupby(group_by_cols)[value].shift(1)

        def calc_yoy_percent_change(row):
            if pd.isna(row['value_prev']):
                return np.nan
            try:
                yoy_change = (row[value] - row['value_prev']) / abs(row['value_prev']) * 100
                return yoy_change
            except ZeroDivisionError:
                return np.nan

        data['yoy_percent_change'] = data.apply(calc_yoy_percent_change, axis=1)
        data = data.round(2)
        data = data.drop(columns=['value_prev'])
    return data


def _calc_yoy_total(data, ner_dict):
    if not data.empty:
        dimension, time, value, metric = ner_dict['dimension'], ner_dict['time'], ner_dict['value'], ner_dict['metric']
        data[value] = pd.to_numeric(data[value], errors='coerce')
        if data[time].nunique()<2:
            return data
        sort_by_cols = dimension + [metric, time]
        group_by_cols = dimension + [metric]
        data.sort_values(by=sort_by_cols, inplace=True)
        data['value_prev'] = data.groupby(group_by_cols)[value].shift(1)

        def calc_yoy_percent_change(row):
            if pd.isna(row['value_prev']):
                return np.nan
            try:
                yoy_change = (row[value] - row['value_prev']) / abs(row['value_prev']) * 100
                return yoy_change
            except ZeroDivisionError:
                return np.nan

        data['yoy_percent_change'] = data.apply(calc_yoy_percent_change, axis=1)
        data = data.round(2)
        data = data.drop(columns=['value_prev'])
    return data


def _calc_qoq(data, ner_dict):
    '''
    :param data: cleaned dataframe
    :param ner_dict:
    :return: dataframe with QOQ percentage change
    '''
    if not data.empty:
        dimension, time, driver, value, metric = ner_dict['dimension'], ner_dict['time'], ner_dict['driver'], ner_dict[
            'value'], ner_dict['metric']
        data[value] = pd.to_numeric(data[value], errors='coerce')
        sort_by_cols = dimension + [metric, driver, time]
        group_by_cols = dimension + [metric, driver, 'quarter']
        data.sort_values(by=sort_by_cols, inplace=True)
        data[['quarter', 'year']] = data[time].str.extract(r'Q(\d) (\d{4})')
        data['year'] = data['year'].astype(int)
        data['quarter'] = data['quarter'].astype(int)
        data['value_prev'] = data.groupby(group_by_cols)[value].shift(1)
        data['year_prev'] = data.groupby(group_by_cols)['year'].shift(1)

        def calc_qoq_percent_change(row):
            if pd.isna(row['value_prev']) or row['year'] != row['year_prev'] + 1:
                return np.nan
            try:
                qoq_change = (row['value'] - row['value_prev']) / abs(row['value_prev']) * 100
                return qoq_change
            except ZeroDivisionError:
                return np.nan

        data['qoq_percent_change'] = data.apply(calc_qoq_percent_change, axis=1)
        # -------------------- FALLBACK (only add this block) --------------------
        # If ALL value_prev are NaN under the existing logic, compute consecutive-quarter QoQ instead.
        if data['value_prev'].isna().all():
            # ensure chronological order for consecutive quarters
            data.sort_values(by=dimension + [metric, driver, 'year', 'quarter'], inplace=True)

            fallback_grp = dimension + [metric, driver]
            data['_prev_value_qoq'] = data.groupby(fallback_grp)[value].shift(1)
            data['_prev_year_qoq'] = data.groupby(fallback_grp)['year'].shift(1)
            data['_prev_quarter_qoq'] = data.groupby(fallback_grp)['quarter'].shift(1)

            def _is_adjacent_q(row):
                if pd.isna(row['_prev_year_qoq']) or pd.isna(row['_prev_quarter_qoq']):
                    return False
                y, q = int(row['year']), int(row['quarter'])
                yp, qp = int(row['_prev_year_qoq']), int(row['_prev_quarter_qoq'])
                return (y == yp and q == qp + 1) or (y == yp + 1 and q == 1 and qp == 4)

            def _calc_fallback_qoq(row):
                prev = row['_prev_value_qoq']
                if pd.isna(prev) or prev == 0 or not _is_adjacent_q(row):
                    return np.nan
                return (row['value'] - prev) / abs(prev) * 100

            data['qoq_percent_change'] = data.apply(_calc_fallback_qoq, axis=1)
            data = data.drop(columns=['_prev_value_qoq', '_prev_year_qoq', '_prev_quarter_qoq'])
        # ------------------ END FALLBACK BLOCK ------------------

        data = data.round(2)
        data = data.drop(columns=['value_prev', 'year_prev'])
    return data


def _calc_qoq_total(data, ner_dict):
    if not data.empty:
        dimension, time, value, metric = ner_dict['dimension'], ner_dict['time'], ner_dict['value'], ner_dict['metric']
        data[value] = pd.to_numeric(data[value], errors='coerce')
        sort_by_cols = dimension + [metric, time]
        group_by_cols = dimension + [metric, 'quarter']
        data = data.sort_values(by=sort_by_cols)
        data[['quarter', 'year']] = data[time].str.extract(r'Q(\d) (\d{4})')
        data['year'] = data['year'].astype(int)
        data['quarter'] = data['quarter'].astype(int)
        data['value_prev'] = data.groupby(group_by_cols)[value].shift(1)
        data['year_prev'] = data.groupby(group_by_cols)['year'].shift(1)

        def calc_qoq_percent_change(row):
            if pd.isna(row['value_prev']) or row['year'] != row['year_prev'] + 1:
                return np.nan
            try:
                qoq_change = (row['value'] - row['value_prev']) / abs(row['value_prev']) * 100
                return qoq_change
            except ZeroDivisionError:
                return np.nan

        data['qoq_percent_change'] = data.apply(calc_qoq_percent_change, axis=1)
        data = data.round(2)
        data = data.drop(columns=['value_prev', 'year_prev'])
    return data


def _calc_hoh_total(data, ner_dict):
    if not data.empty:
        dimension, time, value, metric = ner_dict['dimension'], ner_dict['time'], ner_dict['value'], ner_dict['metric']
        data[value] = pd.to_numeric(data[value], errors='coerce')
        sort_by_cols = dimension + [metric, time]
        group_by_cols = dimension + [metric, 'half']
        data = data.sort_values(by=sort_by_cols)
        data[['half', 'year']] = data[time].str.extract(r'h(\d) (\d{4})')
        data['year'] = data['year'].astype(int)
        data['half'] = data['half'].astype(int)
        data['value_prev'] = data.groupby(group_by_cols)[value].shift(1)
        data['year_prev'] = data.groupby(group_by_cols)['year'].shift(1)

        def calc_hoh_percent_change(row):
            if pd.isna(row['value_prev']) or row['year'] != row['year_prev'] + 1:
                return np.nan
            try:
                hoh_change = (row['value'] - row['value_prev']) / abs(row['value_prev']) * 100
                return hoh_change
            except ZeroDivisionError:
                return np.nan

        data['hoh_percent_change'] = data.apply(calc_hoh_percent_change, axis=1)
        data = data.round(2)
        data = data.drop(columns=['value_prev', 'year_prev'])
    return data


def _calc_hoh(data, ner_dict):
    if not data.empty:
        dimension, time, driver, value, metric = ner_dict['dimension'], ner_dict['time'], ner_dict['driver'], ner_dict[
            'value'], ner_dict['metric']
        data[value] = pd.to_numeric(data[value], errors='coerce')
        sort_by_cols = dimension + [metric, driver, time]
        group_by_cols = dimension + [metric, driver, 'half']
        data.sort_values(by=sort_by_cols, inplace=True)
        data[['half', 'year']] = data[time].str.extract(r'h(\d) (\d{4})')
        data['year'] = data['year'].astype(int)
        data['half'] = data['half'].astype(int)
        data['value_prev'] = data.groupby(group_by_cols)[value].shift(1)
        data['year_prev'] = data.groupby(group_by_cols)['year'].shift(1)

        def calc_hoh_percent_change(row):
            if pd.isna(row['value_prev']) or row['year'] != row['year_prev'] + 1:
                return np.nan
            try:
                hoh_change = (row['value'] - row['value_prev']) / abs(row['value_prev']) * 100
                return hoh_change
            except ZeroDivisionError:
                return np.nan

        data['hoh_percent_change'] = data.apply(calc_hoh_percent_change, axis=1)
        data = data.round(2)
        data = data.drop(columns=['value_prev', 'year_prev'])
    return data


def _calc_mom(data, ner_dict):
    '''
    :param data: cleaned dataframe
    :param ner_dict:
    :return: dataframe with MOM percentage change
    '''
    if not data.empty:
        dimension, time, driver, value, metric = ner_dict['dimension'], ner_dict['time'], ner_dict['driver'], ner_dict[
            'value'], ner_dict['metric']
        data[value] = pd.to_numeric(data[value], errors='coerce')
        extracted_cols = data[time].str.extract(r'month (\d{1,2}) (\d{4})')
        data['month'] = extracted_cols[0].astype(int)
        data['year'] = extracted_cols[1].astype(int)
        sort_by_cols = dimension + [metric, driver, 'year', 'month']
        group_by_cols = dimension + [metric, driver]
        data['value_prev'] = data.groupby(group_by_cols)[value].shift(1)
        data['year_prev'] = data.groupby(group_by_cols)['year'].shift(1)
        data['month_prev'] = data.groupby(group_by_cols)['month'].shift(1)
        data = data.sort_values(by=sort_by_cols)

        def calc_mom_percent_change(row):
            if pd.isna(row['value_prev']) or row['year'] != row['year_prev'] or row['month'] != row['month_prev'] + 1:
                return np.nan
            try:
                mom_change = (row['value'] - row['value_prev']) / abs(row['value_prev']) * 100
                return mom_change
            except ZeroDivisionError:
                return np.nan

        data['mom_percent_change'] = data.apply(calc_mom_percent_change, axis=1)
        data = data.round(2)
        data = data.drop(columns=['value_prev', 'year_prev', 'month_prev'])
    return data


def _calc_mom_total(data, ner_dict):
    if not data.empty:
        dimension, time, value, metric = ner_dict['dimension'], ner_dict['time'], ner_dict['value'], ner_dict['metric']
        data[value] = pd.to_numeric(data[value], errors='coerce')
        extracted_cols = data[time].str.extract(r'month (\d{1,2}) (\d{4})')
        data['month'] = extracted_cols[0].astype(int)
        data['year'] = extracted_cols[1].astype(int)
        sort_by_cols = dimension + [metric, 'year', 'month']
        group_by_cols = dimension + [metric]
        data['value_prev'] = data.groupby(group_by_cols)[value].shift(1)
        data['year_prev'] = data.groupby(group_by_cols)['year'].shift(1)
        data['month_prev'] = data.groupby(group_by_cols)['month'].shift(1)
        data = data.sort_values(by=sort_by_cols)

        def calc_mom_percent_change(row):
            if pd.isna(row['value_prev']) or row['year'] != row['year_prev'] or row['month'] != row['month_prev'] + 1:
                return np.nan
            try:
                mom_change = (row['value'] - row['value_prev']) / abs(row['value_prev']) * 100
                return mom_change
            except ZeroDivisionError:
                return np.nan

        data['mom_percent_change'] = data.apply(calc_mom_percent_change, axis=1)
        data = data.round(2)
        data = data.drop(columns=['value_prev', 'year_prev', 'month_prev'])
    return data


def _sort_filt_data(data, ner_dict, sort_metric, sort_by_col, rank_check, metric_df):
    '''
    Sort tactics/drivers by metric values
    :param data:
    :param ner_dict:
    :param sort_metric:
    :return: sorted dataframe with rank
    '''
    driver, time, value, dim_col, metric_col = ner_dict['driver'], ner_dict['time'], ner_dict['value'], ner_dict[
        'dimension'], ner_dict['metric']

    ascend = _get_sort_order(metric_df, sort_metric[0], rank_check)
    month_check = 'month' in data.columns

    def process_group(group):
        sort_metric_data = group[group[metric_col].isin(sort_metric)]
        most_recent_year = sort_metric_data[time].max()
        most_recent_data = sort_metric_data[sort_metric_data[time] == most_recent_year]
        if month_check:
            most_recent_year = sort_metric_data['year'].max()
            most_recent_month = sort_metric_data[sort_metric_data['year'] == most_recent_year]['month'].max()
            most_recent_data = sort_metric_data[
                (sort_metric_data['month'] == most_recent_month) & (sort_metric_data['year'] == most_recent_year)]
        if 'sourceofchange' in sort_metric:  # or 'contribution' in sort_metric:
            most_recent_data['abs_value'] = most_recent_data[value].abs()
            ordered_drivers_ranked = most_recent_data.sort_values(by='abs_value', ascending=False)
            ordered_drivers_ranked['rank'] = ordered_drivers_ranked['abs_value'].rank(method='dense', ascending=False)
            ordered_drivers_ranked = ordered_drivers_ranked.drop('abs_value', axis=1)
        else:
            if month_check or most_recent_data[sort_by_col].isna().all():
                agg_data = (most_recent_data.groupby(dim_col + [driver]).agg({'value': 'sum'}).reset_index())
                ordered_drivers_ranked = agg_data.sort_values(by='value', ascending=ascend)
                ordered_drivers_ranked['rank'] = ordered_drivers_ranked['value'].rank(method='dense', ascending=ascend)
            else:
                agg_data = (most_recent_data.groupby(dim_col + [driver]).agg({sort_by_col: 'sum'}).reset_index())
                ordered_drivers_ranked = agg_data.sort_values(by=sort_by_col, ascending=ascend)
                ordered_drivers_ranked['rank'] = ordered_drivers_ranked[sort_by_col].rank(method='dense',
                                                                                          ascending=ascend)
        return ordered_drivers_ranked

    # Apply the function to each dimension group, then merge back to the original dataframe
    if ner_dict['driver_high'] == 'media_channel':
        if data['media_channel'].isnull().all():
            data['media_channel'] = data['tactics']
        ranked_groups = data.groupby(dim_col + ['media_channel']).apply(process_group).reset_index(drop=True)
    else:
        ranked_groups = data.groupby(dim_col).apply(process_group).reset_index(drop=True)
    # Merge the ranked data with original data on the specific columns
    merged_data = pd.merge(data, ranked_groups[dim_col + [driver, 'rank']], on=dim_col + [driver], how='left')
    # Sort the merged data by dimension, rank, and metric column
    merged_data.sort_values(by=dim_col + ['rank', metric_col], inplace=True)
    return merged_data


def _get_granular_table(data, ner_dict, metrics_order, ner_filter, rank_check, sort_by_col, metric_df, client_code, model_group_id):
    time_order = sorted(data['time'].unique().tolist(), reverse=True)
    time_col, dim_col, driver_col_detail, metric_col, value_col = ner_dict['time'], ner_dict['dimension'], ner_dict[
        'driver'], ner_dict['metric'], ner_dict['value']
    if driver_col_detail in ['tactics', 'tactics_detail', 'tactics_concat', 'detail_tactics', 'media_tactics_concat']:
        driver_col_detail_upp = driver_col_detail + '_origin'
    else:
        driver_col_detail_upp = driver_col_detail
    pivot_data, sort_col = _table_display(data, dim_col, driver_col_detail_upp, time_col, metric_col, value_col,
                                          metrics_order, time_order, ner_filter, rank_check, sort_by_col, metric_df,
                                          client_code, model_group_id)
    return pivot_data, sort_col


def _reorder_columns(dynamic_cols, metric_order, quarter_check, month_check, half_check):
    if quarter_check or half_check:
        idx = -3
    elif month_check:
        idx = -4
    else:
        idx = -2

    def parse_column(col):
        parts = col.split()
        if 'QOQ' in parts or 'YOY' in parts or 'HOH' in parts:
            metric = ' '.join(parts[:idx])
            time = ' '.join(parts[idx: -1])
            growth = 2
        else:
            metric = ' '.join(parts[:idx + 1])
            time = ' '.join(parts[idx + 1:])
            growth = 1
        return metric, time, growth

    metric_mapping = {metric: i for i, metric in enumerate(metric_order)}
    columns_df = pd.DataFrame([parse_column(col) for col in dynamic_cols], columns=['metric', 'time', 'growth'])
    columns_df['original_col'] = dynamic_cols
    columns_df['metric_order'] = columns_df['metric'].map(metric_mapping)
    if quarter_check:
        columns_df[['num', 'year']] = columns_df['time'].str.extract(r'Q(\d) (\d{4})')
    elif half_check:
        columns_df[['num', 'year']] = columns_df['time'].str.extract(r'h(\d) (\d{4})')
    elif month_check:
        extract_col = columns_df['time'].str.extract(r'month (\d{1,2}) (\d{4})')
        columns_df['num'] = extract_col[0].astype(int)
        columns_df['year'] = extract_col[1].astype(int)
    else:
        columns_df['year'] = columns_df['time'].str.extract(r'(\d{4})')
        columns_df['num'] = 1
    columns_df.sort_values(by=['metric_order', 'growth', 'year', 'num'], ascending=[True, True, False, False],
                           inplace=True)
    return columns_df['original_col'].tolist()


def _table_display(data, dim_col, driver_col, time_col, metric_col, value_col, metrics_order, time_order, ner_filter,
                   rank_check, sort_by_col, metric_df, client_code, model_group_id):
    data = data.round(2)
    data = data[data[metric_col] != 'margin roi index']
    growth_col = next(
        (col for col in ['yoy_percent_change', 'qoq_percent_change', 'hoh_percent_change'] if col in data.columns),
        None)
    main_metric = ner_filter['main_metric']
    main_metric = ["source of change" if m == "sourceofchange" else m for m in main_metric]
    most_recent_time = str(data['time'].max())
    quarter_check = 'quarter' in data.columns
    half_check = 'half' in data.columns
    month_check = 'month' in data.columns
    ascend = _get_sort_order(metric_df, main_metric[0], rank_check)
    abs_sort = True if sort_by_col == 'value' and any(
        metric in ['source of change'] for metric in main_metric) else False
    value_cols = [value_col]
    if growth_col:
        value_cols.append(growth_col)

    kpi_name_mapping = {'sourceofchange': 'Source of Change'}
    metrics_order = [kpi_name_mapping[m] if m in kpi_name_mapping else m for m in metrics_order]
    if month_check:
        data = data[data[metric_col].isin(main_metric + ['sourceofchange'])]
    pivot_data = data.pivot_table(index=dim_col + [driver_col], columns=[metric_col, time_col],
                                  values=value_cols, aggfunc='first')

    pivot_data.columns = ['{} {} {}'.format(kpi_name_mapping.get(col[1], col[1]), col[2], col[0])
                          for col in pivot_data.columns]
    pivot_data = pivot_data.reset_index()
    # dim_col_order, driver_col_order = data[dim_col].unique().tolist(), data[driver_col].unique().tolist()
    dim_col_order = {col: data[col].unique().tolist() for col in dim_col}
    driver_col_order = data[driver_col].unique().tolist()
    # pivot_data[dim_col] = pd.Categorical(pivot_data[dim_col], categories=dim_col_order, ordered=True)
    for col in dim_col:
        pivot_data[col] = pd.Categorical(pivot_data[col], categories=dim_col_order[col], ordered=True)
    pivot_data[driver_col] = pd.Categorical(pivot_data[driver_col], categories=driver_col_order, ordered=True)
    if month_check:
        most_recent_year = data['year'].max()
        most_recent_month = data.loc[data['year'] == most_recent_year, 'month'].max()
        most_recent_time = f'month {most_recent_month} {most_recent_year}'
        sort_col_m = [col for col in pivot_data.columns if most_recent_time in col]
        if sort_col_m:
            pivot_data = pivot_data.sort_values(by=dim_col + [sort_col_m[0]], ascending=ascend)
    elif quarter_check:
        most_recent_year = data['year'].max()
        most_recent_quarter = data.loc[data['year'] == most_recent_year, 'quarter'].max()
        most_recent_time = f'Q{most_recent_quarter} {most_recent_year}'
        pivot_data = pivot_data.sort_values(by=dim_col + [driver_col])
    else:
        pivot_data = pivot_data.sort_values(by=dim_col + [driver_col])
    pivot_data.columns = pivot_data.columns.str.replace(' value', '')
    pivot_data.columns = pivot_data.columns.str.replace(' yoy_percent_change', ' YOY')
    pivot_data.columns = pivot_data.columns.str.replace(' qoq_percent_change', ' QOQ')
    pivot_data.columns = pivot_data.columns.str.replace(' hoh_percent_change', ' HOH')

    if 'tactics' in driver_col:
        pivot_data.columns = pivot_data.columns.str.replace(driver_col, 'tactics')

    dim_mapping = {'tp': 'TP', 'ahw': 'AHW', 'led': 'LED', 'pen': 'PEN', 'clv': 'CLV'}
    for col in dim_col:
        pivot_data[col] = pivot_data[col].apply(lambda x: _dim_value_map(x, dim_mapping))
    # pivot_data[dim_col] = pivot_data[dim_col].apply(lambda x: _dim_value_map(x, {'tp': 'TP', 'ahw': 'AHW', 'led': 'LED', 'pen': 'PEN', 'clv': 'CLV'}))
    static_columns = pivot_data.columns[:len(dim_col) + 1].tolist()

    dynamic_columns = [col for col in pivot_data.columns if col not in static_columns]
    pivot_data[dynamic_columns] = pivot_data[dynamic_columns].replace([np.inf, -np.inf], np.nan)
    pivot_data = pivot_data.dropna(axis=0, how='all', subset=dynamic_columns)

    sorted_dynamic_columns = _reorder_columns(dynamic_columns, metrics_order, quarter_check, month_check, half_check)
    sorted_columns = static_columns + sorted_dynamic_columns
    pivot_data = pivot_data[sorted_columns]

    def _sort_condition(col):
        def contains_main_metric(col):
            return any(metric in col.lower() for metric in main_metric)

        if sort_by_col != 'value':
            return (most_recent_time in col) and contains_main_metric(col) and (
                    'yoy' in col.lower() or 'qoq' in col.lower() or 'hoh' in col.lower())
        else:
            return (most_recent_time in col) and contains_main_metric(col) and not (
                    'yoy' in col.lower() or 'qoq' in col.lower() or 'hoh' in col.lower())

    pivot_sort_cols = [col for col in pivot_data.columns if
                       _sort_condition(col) and 'share of spend' not in col.lower()]
    pivot_sort_col = None
    if pivot_sort_cols and not month_check:
        pivot_sort_col = pivot_sort_cols[0]

        if abs_sort:
            pivot_data = pivot_data.reindex(pivot_data[pivot_sort_col].abs().sort_values(ascending=ascend).index)
        else:
            pivot_data = pivot_data.sort_values(by=pivot_sort_col, ascending=ascend)
        multi_value_cols = {col: data[col].nunique() for col in dim_col if data[col].nunique() > 1}
        if multi_value_cols:
            # if len(dim_col_order) > 1:
            pivot_data['sort_helper'] = pivot_data.groupby(dim_col).cumcount()
            pivot_data = pivot_data.sort_values(by=dim_col + ['sort_helper'])
            # pivot_data = pivot_data.groupby(dim_col).head(5)
            pivot_data = pivot_data.drop(columns=['sort_helper'])
    original_columns = pivot_data.columns.tolist()

    metric_col_idx = len(dim_col) + 1
    pivot_data = add_unit_sign(data=pivot_data, metric_col_idx=metric_col_idx, to_mm_metrics=["spend", "activity"],
                               client_code=client_code, model_group_id=model_group_id)

    # spend, response, activity with MM
    # def transform_to_mm(df):
    #     for col in pivot_data.columns[len(dim_col) + 1:]:
    #         if (kpi_name_mapping.get('share of spend', 'share of spend') not in col and 'YOY' not in col
    #                 and kpi_name_mapping.get('cost per',
    #                                          'cost per') not in col and 'QOQ' not in col and 'HOH' not in col and 'spend share' not in col):
    #             # if 'spend' in col or 'response' in col or 'activity' in col:
    #             if 'spend' in col or 'activity' in col:
    #                 df[col] = df[col] / 1000000  # Convert to millions
    #                 df[col] = df[col].apply(lambda x: f"{x:.2f}" if not pd.isna(x) else x)
    #                 df.rename(columns={col: col + ' (MM)'}, inplace=True)
    #     return df
    #
    # pivot_data = transform_to_mm(pivot_data)
    if pivot_sort_col in original_columns:
        renamed_cols = {old_col: new_col for old_col, new_col in zip(original_columns, pivot_data.columns) if
                        old_col != new_col}
        pivot_sort_col = renamed_cols.get(pivot_sort_col, pivot_sort_col)

    # YOY, share of spend add %
    # soc, contribution add %
    # def append_percentage(df):
    #     # Iterate over columns and apply transformation if conditions are met
    #     for col in pivot_data.columns[len(dim_col) + 1:]:
    #         if 'YOY' in col or 'share of spend' in col or 'spend share' in col or 'contribution' in col or 'source of change' in col.lower() or 'QOQ' in col or 'HOH' in col:
    #             df[col] = pd.to_numeric(df[col], errors='coerce')
    #             df[col] = df[col].apply(lambda x: f"{x:.1f}%" if not pd.isna(x) else x)
    #     return df
    #
    # pivot_data = append_percentage(pivot_data)
    #
    # # sales add $, 0,000
    # def format_sales_values(df):
    #     for col in pivot_data.columns[len(dim_col) + 1:]:
    #         if ((kpi_name_mapping.get('sales', 'sales') in col
    #              and kpi_name_mapping.get('response', 'response') not in col) or ('gross adds' in col) or (
    #                     'cost per kpi' in col)) and ('YOY' not in col
    #                                                  and 'QOQ' not in col and 'HOH' not in col):
    #             # df[col] = df[col].apply(lambda x: f"${x:,.2f}" if not pd.isna(x) else x)
    #             df[col] = df[col].apply(lambda x: f"{x:,.0f}" if not pd.isna(x) else x)
    #     return df
    #
    # pivot_data = format_sales_values(pivot_data)
    #
    # def response_format(df):
    #     for col in pivot_data.columns[len(dim_col) + 1:]:
    #         if kpi_name_mapping.get('response',
    #                                 'response') in col and 'YOY' not in col and 'QOQ' not in col and 'HOH' not in col:
    #             df[col] = df[col].apply(lambda x: f"${x:,.2f}" if pd.notna(x) else x)
    #     return df
    #
    # pivot_data = response_format(pivot_data)
    #
    # def activity_format(df):
    #     for col in pivot_data.columns[len(dim_col) + 1:]:
    #         if kpi_name_mapping.get('activity',
    #                                 'activity') in col and 'cost per' not in col and 'YOY' not in col and 'QOQ' not in col and 'HOH' not in col:
    #             df[col] = df[col].apply(lambda x: f"{float(x):,.2f}" if not pd.isna(x) else x)
    #     return df
    #
    # pivot_data = activity_format(pivot_data)
    #
    # def add_dollar_sign(df):
    #     for col in pivot_data.columns[len(dim_col) + 1:]:
    #         if ('roi' in col or kpi_name_mapping.get('spend', 'spend') in col or kpi_name_mapping.get(
    #                 'cost per', 'cost per') in col) and kpi_name_mapping.get('share of spend',
    #                                                                          'share of spend') not in col and 'cost per kpi' not in col and 'spend share' not in col and 'YOY' not in col and 'QOQ' not in col and 'HOH' not in col:
    #             df[col] = df[col].apply(lambda x: f"${float(x):,.2f}" if not pd.isna(x) else x)
    #     return df
    #
    # pivot_data = add_dollar_sign(pivot_data)

    # def rename_change_columns(col):
    #     if re.search(r'(YOY|QOQ|HOH|MOM)', col):
    #         col_clean = re.sub(r'\s*(YOY|QOQ|HOH|MOM)', '', col)
    #         return f"% Change in {col_clean}"
    #     return col
    # pivot_data.columns = [rename_change_columns(col) for col in pivot_data.columns]
    # column name uppercase
    def custom_title_case(column_name):
        words = column_name.split()
        final_words = [word.upper() if word in ['YOY', '(MM)', 'roi', 'QOQ', 'HOH', 'kpi', 'cpm'] else word.title() for word in
                       words]
        return ' '.join(final_words)

    pivot_data.columns = [custom_title_case(col) for col in pivot_data.columns]

    if pivot_sort_col:
        # if re.search(r'\b(YOY|QOQ|HOH|MOM)\b', pivot_sort_col, re.IGNORECASE):
        #     pivot_sort_col = re.sub(r'\s*(YOY|QOQ|HOH|MOM)', '', pivot_sort_col)
        #     pivot_sort_col = f"% Change in {pivot_sort_col}"
        pivot_sort_col = custom_title_case(pivot_sort_col)
    check_cols = [c for c in pivot_data.columns if any(m in c.lower() for m in main_metric)]
    if check_cols:
        pivot_data = pivot_data.dropna(subset=check_cols, how='all')
    return pivot_data, pivot_sort_col


def _rename_change_columns(col):
    if re.search(r'(YOY|QOQ|HOH|MOM)', col):
        col_clean = re.sub(r'\s*(YOY|QOQ|HOH|MOM)', '', col)
        return f"% Change in {col_clean}"
    return col


def _top_data(data, ner_dict, top_num='na'):
    '''

    :param data:
    :param ner_dict:
    :param top_num:
    :return: top_num tactics/drivers dataframe
    '''
    rank_num = data['rank'].max()

    if rank_num >= 3:
        top_num = 3 if top_num == 'na' else min(int(top_num), rank_num)
    else:
        top_num = rank_num if top_num == 'na' else min(int(top_num), rank_num)
    top_data = data[data['rank'] <= top_num]
    if ner_dict['driver_high'] == 'media_channel':
        # top_num = 1
        top_data = data[data['rank'] <= top_num]

        def dedup(group):
            if group['tactics_concat'].nunique() > 1:
                first_value = group['tactics_concat'].iloc[0]
                group = group[group['tactics_concat'] == first_value]
            return group

        top_data = top_data.groupby(ner_dict['dimension'] + ['tactics_concat']).apply(dedup).reset_index(drop=True)
    return top_data


def _top_increase_data(data, ner_dict, growth_col, main_metric, top_num=1):
    '''
    :return: top 1 growing tactic/driver data
    '''
    driver, time, value, dim_col, metric_col = ner_dict['driver'], ner_dict['time'], growth_col, ner_dict[
        'dimension'], ner_dict['metric']
    most_recent_time = data[time].sort_values(ascending=False).values[0]
    if 'qoq' in growth_col or 'hoh' in growth_col:
        most_recent_year = data['year'].max()
        t = 'quarter' if 'qoq' in growth_col else 'half'
        most_recent_quarter = data[data['year'] == most_recent_year][t].max()
        most_recent_time = data[(data[t] == most_recent_quarter) & (data['year'] == most_recent_year)][time].values[0]
    growth_data = data[
        (data[time] == most_recent_time) & (data[value].notnull()) & (data[metric_col].isin(main_metric))]

    top_inc_data = growth_data.groupby(dim_col + [metric_col], as_index=False).apply(
        lambda x: x.nlargest(top_num, value)).reset_index(drop=True)

    return top_inc_data


def _top_decrease_data(data, ner_dict, growth_col, main_metric, top_num=1):
    '''
    :return: top 1 decreasing tactic/driver data
    '''
    driver, time, value, dim_col, metric_col = ner_dict['driver'], ner_dict['time'], growth_col, ner_dict[
        'dimension'], ner_dict['metric']
    most_recent_time = data[time].sort_values(ascending=False).values[0]
    if 'qoq' in growth_col or 'hoh' in growth_col:
        most_recent_year = data['year'].max()
        t = 'quarter' if 'qoq' in growth_col else 'half'
        most_recent_quarter = data[data['year'] == most_recent_year][t].max()
        most_recent_time = data[(data[t] == most_recent_quarter) & (data['year'] == most_recent_year)][time].values[0]
    growth_data = data[
        (data[time] == most_recent_time) & (data[value].notnull()) & (data[metric_col].isin(main_metric))]

    top_dec_data = growth_data.groupby(dim_col + [metric_col], as_index=False).apply(
        lambda x: x.nsmallest(top_num, value)).reset_index(drop=True)
    return top_dec_data


def _get_granular_level_pivot(data, ner_dict, ner_filter, trend_check, metric_df):
    time_col, dim_col, driver_col_detail, metric_col, value_col = ner_dict['time'], ner_dict['dimension'], ner_dict[
        'driver'], ner_dict['metric'], ner_dict['value']
    main_metric = ner_filter['main_metric']
    dim_col = [x for x in dim_col if x!='core_dimension_term']
    output_text = _data_to_text(data, dim_col, driver_col_detail, time_col, metric_col, value_col,
                                main_metric, trend_check, metric_df)
    return output_text


def _data_to_text(data, dim_col, driver_col, time_col, metric_col, value_col, main_metric, trend_check, metric_df):
    data = data.round(2)
    dim_unique_combinations = data[dim_col].drop_duplicates()
    growth_col = next(
        (col for col in ['yoy_percent_change', 'qoq_percent_change', 'hoh_percent_change'] if
         col in data.columns), None)
    output_text = ""
    if 'core_dimension_term' in data.columns:
        aka_df = data[['core_dimension_term', driver_col]].drop_duplicates()
        unique_tactics = aka_df.apply(
            lambda row: f"{row[driver_col]} (aka {row['core_dimension_term'].strip(', ')})"
            if pd.notna(row["core_dimension_term"]) and row["core_dimension_term"].strip(", ")
            else None,
            axis=1
        ).dropna().unique()
        output_text += ". ".join(unique_tactics) + '. ' if len(unique_tactics) > 0 else ''
    for _, row in dim_unique_combinations.iterrows():
        values = [f"'{val}'" for val in row if val != 'overall']
        if values:
            output_text += "For " + " - ".join(values) + ": "
        dim_filt = row.to_dict()
        dim_group = data.copy()
        for col, value in dim_filt.items():
            dim_group = dim_group[dim_group[col] == value]
        if 'year' in dim_group.columns:
            most_recent_year = dim_group['year'].max()
            recent_year_values = dim_group[
                (dim_group['year'] == most_recent_year) & (dim_group[metric_col].isin(main_metric))]
        else:
            most_recent_year = dim_group[time_col].sort_values(ascending=False).values[0]
            recent_year_values = dim_group[
                (dim_group[time_col] == most_recent_year) & (dim_group[metric_col].isin(main_metric))]
        # Sort drivers based on the main_metric value in the most recent year
        if 'quarter' in recent_year_values.columns:
            recent_year_values = recent_year_values.sort_values(by=['year', 'quarter'], ascending=False)
            recent_periods = recent_year_values[time_col].unique()[:3].tolist()
        elif 'month' in recent_year_values.columns:
            recent_year_values = dim_group[dim_group[metric_col].isin(main_metric)]
            recent_year_values = recent_year_values.sort_values(by=['year', 'month'], ascending=False)
            recent_periods = recent_year_values[time_col].unique()[:3].tolist()
        else:
            recent_periods = None

        ascend = metric_df.loc[metric_df[metric_col] == main_metric[0], 'ascend'].head(1).iloc[0] if not metric_df.loc[
            metric_df[metric_col] == main_metric[0], 'ascend'].empty else False

        sorted_drivers = recent_year_values.groupby(driver_col)[value_col].mean().sort_values(
            ascending=ascend).index.tolist()
        for driver in sorted_drivers:
            for m_metric in main_metric:
                driver_group = dim_group[(dim_group[driver_col] == driver) & (dim_group[metric_col] == m_metric)]
                if recent_periods:
                    driver_group = driver_group[driver_group[time_col].isin(recent_periods)]
                sorted_group = driver_group.sort_values(by=time_col, ascending=False)
                # values_by_time = sorted_group[sorted_group[metric_col].isin(main_metric)][value_col].tolist()
                values_by_time = sorted_group[value_col].tolist()
                period = sorted_group[time_col].unique().tolist()
                if values_by_time and not all(pd.isna(v) for v in values_by_time):
                    if m_metric not in ['sourceofchange', 'contribution']:
                        rename_values = metric_df.loc[metric_df[metric_col] == m_metric, 'rename'].dropna()
                        rename_metric = rename_values.iloc[0] if not rename_values.empty else m_metric
                        if rename_metric and rename_metric != m_metric:
                            rename_metric = f"{m_metric} ({rename_metric})"
                        output_text += f"'{driver}' '{rename_metric}' is {', '.join([str(v) if pd.notna(v) else 'N/A' for v in values_by_time])} in {', '.join(map(str, period))} "
                    elif 'sourceofchange' == m_metric:
                        output_text += f"'{driver}' drives {', '.join([str(v) + '%' for v in values_by_time])} sales change in {', '.join(map(str, period))}"
                    elif 'contribution' == m_metric:
                        output_text += f"'{driver}' '{m_metric}' is {', '.join([str(v) + '%' for v in values_by_time])} in {', '.join(map(str, period))}"

                if len(period) > 1:
                    output_text += " respectively"
                if growth_col and trend_check != 'no':
                    change_type_mapping = {'qoq': 'QOQ', 'yoy': 'YOY', 'mom': 'monthly', 'hoh': 'half year'}
                    change_type = next((change_type_mapping[key] for key in change_type_mapping if key in growth_col),
                                       "")
                    # growth_by_time = \
                    #     sorted_group[(sorted_group[metric_col] == main_metric) & (sorted_group[growth_col].notnull())][
                    #         [growth_col, time_col]]
                    growth_by_time = sorted_group[sorted_group[growth_col].notnull()][[growth_col, time_col]]
                    growths = growth_by_time[growth_col].tolist()
                    periods = growth_by_time[time_col].tolist()
                    if len(growths) != 0:
                        output_text += f", resulting in {', '.join([f'{g:.2f}%' for g in growths])} {change_type} change in {', '.join([str(t) for t in periods])}"
                    if len(periods) > 1:
                        output_text += " respectively"
                output_text += ". "

            output_text += "\n"

    return output_text


def _truncate_text(input_text, max_length):
    # Truncate the input text based on string length if necessary
    if len(input_text) > max_length:
        truncated_text = input_text[:max_length]
        return truncated_text
    return input_text

def _calc_roi_index(data, benchmark_idx):
    time_roi_map = benchmark_idx[['time', 'benchmarkROI']].drop_duplicates().set_index('time')['benchmarkROI'].to_dict()
    valid_metric = [x for x in benchmark_idx['metric'].unique() if x in data["metric"].unique()]
    if valid_metric:
        metric = valid_metric[0]
    else:
        metric = benchmark_idx['metric'].unique()[0]
    for year in time_roi_map.keys():
        roi_idx = data[(data['time'] == year) & (data['metric'] == metric) & (
                data['media_channel'] != 'non-media')]
        roi_idx['value'] = 100 * roi_idx['value'] / time_roi_map[year]
        roi_idx['value'] = roi_idx['value'].astype(int)
        roi_idx['metric'] = f"{metric} index"
        data = pd.concat([data, roi_idx])
    return data

def _benchmark_output(data, ner_dict, how_many, intention, core_dimension, benchmark_idx):
    if not benchmark_idx.empty and 'benchmarkROI' in benchmark_idx.columns and not data.empty:
        data = _calc_roi_index(data, benchmark_idx)
        benchmark_idx = benchmark_idx[['media_channel', 'time', 'benchmark roi index']]
        top_data_benchmark = _top_data(data, ner_dict, how_many)
        benchmark_str, benchmark_data = _merge_benchmark(intention, None, core_dimension, None,
                                                                   ner_dict, data,
                                                                   top_data_benchmark, benchmark_idx)
    else:
        benchmark_str, benchmark_data = "", pd.DataFrame()
    return benchmark_str, benchmark_data

def _benchmark_data(top_data, benchmark_idx, top_inc_data, top_dec_data, ner_dict):
    driver_col, metric_col, value_col, time_col, dim_col = "tactics", ner_dict['metric'], ner_dict['value'], \
        ner_dict['time'], ner_dict['dimension']
    if not 'time' in benchmark_idx.columns:
        most_recent_time = top_data[time_col].max()
        top_data = top_data[top_data[time_col] == most_recent_time]
    if ner_dict['driver'] == 'media_tactics_concat':
        origin_driver = 'tactics_concat_origin'
    else:
        origin_driver = ner_dict['driver'] + '_origin'
    data = top_data[top_data[metric_col] == 'margin roi index'][
        dim_col + [time_col, metric_col, driver_col, origin_driver, 'media_channel', value_col]]
    data['source'] = 'top'
    if top_inc_data.empty == False:
        inc_data = top_inc_data[top_inc_data[metric_col] == 'margin roi index'][
            dim_col + [time_col, metric_col, driver_col, origin_driver, 'media_channel', value_col]]
        inc_data['source'] = 'top increase'
        data = pd.concat([data, inc_data], axis=0)
    if top_dec_data.empty == False:
        dec_data = top_dec_data[top_dec_data[metric_col] == 'margin roi index'][
            dim_col + [time_col, metric_col, driver_col, driver_col + '_origin', 'media_channel', value_col]]
        dec_data['source'] = 'top decrease'
        data = pd.concat([data, dec_data], axis=0)
    if 'time' in benchmark_idx.columns:
        on_col = ['media_channel', 'time']
    else:
        on_col = ['media_channel']
    data = pd.merge(data, benchmark_idx, how='left', on=on_col)
    # data = pd.merge(data, benchmark_idx, how='left', on=['media_channel', 'time'])
    data = data.dropna(subset=['benchmark roi index'])
    data['over/under index'] = 100 * (data['value'].astype(int) / data['benchmark roi index'].astype(int) - 1)
    data['over/under index'] = data['over/under index'].round(0).astype(int)

    def categorize(row):
        index = row['over/under index']
        if 10 < index <= 100:
            return 'overperforms'
        elif -10 < index <= 10:
            return 'is on par'
        elif -100 < index <= -10:
            return 'underperforms'
        elif index > 100:
            return 'significantly overperforms'
        elif index <= -100:
            return 'significantly underperforms'

    data['respond'] = data.apply(categorize, axis=1)

    return data


def _benchmark_to_text(data, ner_dict):
    driver_col = "Tactics" # ner_dict['driver']
    data = data[[driver_col, 'time', 'respond']].drop_duplicates()
    if data.empty == False:
        str_benchmark = ["Compared to ROI Genome Benchmarks:"]
        multi_year = data['time'].unique().tolist()
        for idx, row in data.iterrows():
            if len(multi_year) > 1:
                str_benchmark.append(f"- In {row['time']}, '{row[driver_col]}' margin ROI index {row['respond']} "
                                     f"relative to industry norms.")
            else:
                str_benchmark.append(f"- '{row[driver_col]}' margin ROI index {row['respond']} "
                                     f"relative to industry norms.")
    else:
        str_benchmark = []

    return '\n'.join(str_benchmark)


def _dim_value_map(value, special_mapping=None):
    if isinstance(value, str):
        components = value.split()
        mapped_components = [
            special_mapping.get(word.lower(), word.capitalize()) if special_mapping else word.capitalize()
            for word in components
        ]
        return ' '.join(mapped_components)
    return value


def _benchmark_table_display(data, ner_dict):
    # data = data.drop(['source', 'respond', 'metric', ner_dict['driver']], axis=1)
    data.drop(['source', 'respond', 'metric', "tactics"], axis=1)
    dim_col = ner_dict['dimension']
    dim_mapping = {'tp': 'TP', 'ahw': 'AHW', 'led': 'LED', 'pen': 'PEN', 'clv': 'CLV'}
    for dim in dim_col:
        data[dim] = data[dim].apply(lambda x: _dim_value_map(x, dim_mapping))

    if ner_dict['driver'] == 'media_tactics_concat':
        origin_driver = 'tactics_concat_origin'
    else:
        origin_driver = ner_dict['driver'] + '_origin'
    data = data.rename(columns={'value': 'margin roi index', origin_driver: 'tactics'})

    def append_percentage(df):
        # Iterate over columns and apply transformation if conditions are met
        for col in df.columns:
            if 'over/under index' in col:
                df[col] = df[col].apply(lambda x: f"{int(x)}%" if not pd.isna(x) else x)
        return df

    data = append_percentage(data)

    def custom_title_case_mediachannel(row):
        words = row.split()
        final_words = [word.upper() if word in ['pr', 'tv', 'ctv', 'dooh'] else word.title() for word in words]
        return ' '.join(final_words)

    data['media_channel'] = data['media_channel'].apply(custom_title_case_mediachannel)

    def custom_title_case(column_name):
        words = column_name.split()
        final_words = [word.upper() if word in ['roi', 'kpi'] else word.title() for word in words]
        return ' '.join(final_words)

    data.columns = [custom_title_case(col) for col in data.columns]

    return data


def _get_total_level_pivot(data, ner_dict, ner_filter, metric_df):
    time_col, dim_col, metric_col, value_col = ner_dict['time'], ner_dict['dimension'], ner_dict['metric'], ner_dict[
        'value']
    main_metric = ner_filter['main_metric']
    output_text = _total_to_text(data, dim_col, time_col, metric_col, value_col, main_metric, metric_df)
    return output_text


def _total_to_text(data, dim_col, time_col, metric_col, value_col, main_metric, metric_df):
    data = data.round(2)
    dim_unique_combinations = data[dim_col].drop_duplicates()
    growth_col = next(
        (col for col in ['yoy_percent_change', 'qoq_percent_change', 'mom_percent_change', 'hoh_percent_change'] if
         col in data.columns), None)

    output_text = ""
    for _, row in dim_unique_combinations.iterrows():
        values = [f"'{val}'" for val in row if val != 'overall']
        if values:
            output_text += "For " + " - ".join(values) + ": "
        dim_filt = row.to_dict()
        dim_group = data[data[metric_col].isin(main_metric)]
        for col, value in dim_filt.items():
            dim_group = dim_group[dim_group[col] == value]
        if 'quarter' in dim_group.columns:# and 'roi' not in main_metric:
            # quarter = dim_group[dim_group['year'] == dim_group['year'].max()]['quarter'].max()
            # filtered = dim_group[dim_group['quarter'] == quarter]
            recent_year_values = dim_group.sort_values(by=['year', 'quarter'], ascending=False)
            recent_periods = recent_year_values[time_col].unique()[:3].tolist()
            dim_group = dim_group[dim_group[time_col].isin(recent_periods)]
        elif 'month' in dim_group.columns:
            recent_year_values = dim_group.sort_values(by=['year', 'month'], ascending=False)
            recent_periods = recent_year_values[time_col].unique()[:3].tolist()
            dim_group = dim_group[dim_group[time_col].isin(recent_periods)]
        for m_metric in main_metric:
            metric_group = dim_group[dim_group[metric_col] == m_metric]
            if not metric_group.empty and 'driver' in metric_group.columns:
                driver = metric_group['driver'].values[0]
            else:
                driver = 'overall' if m_metric not in ['sourceofchange', 'contribution'] else 'marketing'
            sorted_group = metric_group.sort_values(by=time_col, ascending=False)
            values_by_time = sorted_group[value_col].tolist()
            period = sorted_group[time_col].unique().tolist()
            period = [re.sub(r'month (\d{1,2})', _replace_month, str(col)) for col in period]
            if values_by_time and not all(pd.isna(v) for v in values_by_time):
                if m_metric not in ['sourceofchange', 'contribution']:
                    rename_values = metric_df.loc[metric_df[metric_col] == m_metric, 'rename'].dropna()
                    rename_metric = rename_values.iloc[0] if not rename_values.empty else m_metric
                    if rename_metric and rename_metric != m_metric:
                        rename_metric = f"{m_metric} ({rename_metric})"
                    output_text += f"{driver} '{rename_metric}' is {', '.join([str(v) for v in values_by_time])} in {', '.join(map(str, period))}"
                elif 'contribution' == m_metric:
                    output_text += f"{driver} '{m_metric}' is {', '.join([str(v) + '%' for v in values_by_time])} in {', '.join(map(str, period))}"
                # output_text += f"Overall '{m_metric}' is {', '.join([str(v) + ' in ' + str(p) for v, p in zip(values_by_time, period)])}"
                # if len(period) > 1:
                #     output_text += " respectively"
                if growth_col:
                    change_type_mapping = {'qoq': 'QOQ', 'yoy': 'YOY', 'mom': 'monthly', 'hoh': 'half year'}
                    change_type = next((change_type_mapping[key] for key in change_type_mapping if key in growth_col),
                                       "")
                    growth_by_time = \
                        sorted_group[sorted_group[growth_col].notnull()][[growth_col, time_col]]
                    growths = growth_by_time[growth_col].tolist()
                    periods = growth_by_time[time_col].tolist()
                    periods = [re.sub(r'month (\d{1,2})', _replace_month, str(col)) for col in periods]
                    if len(growths) != 0 and any(abs(g) >= 0.5 for g in growths):
                        output_text += f", resulting in {', '.join([f'{g:.2f}%' for g in growths])} {change_type} change in {', '.join([str(t) for t in periods])}"
                        # if len(periods) > 1:
                        #     output_text += " respectively"
                    if len(growths) != 0 and all(abs(g) < 0.5 for g in growths):
                        output_text += ", the changes over time remain flat"
                output_text += ". "
        output_text += "\n"
    return output_text


def _get_total_table(data, ner_dict, ner_filter, rank_check, sort_by_col, metric_df, client_code, model_group_id):
    time_col, dim_col, metric_col, value_col = ner_dict['time'], ner_dict['dimension'], ner_dict['metric'], ner_dict[
        'value']
    pivot_data = _total_table_display(data, dim_col, time_col, metric_col, value_col, ner_filter, rank_check,
                                      sort_by_col, metric_df, client_code, model_group_id)
    return pivot_data


def _total_table_display(data, dim_col, time_col, metric_col, value_col, ner_filter, rank_check, sort_by_col,
                         metric_df, client_code, model_group_id):
    data = data.round(2)
    growth_col = next(
        (col for col in ['yoy_percent_change', 'qoq_percent_change', 'hoh_percent_change'] if col in data.columns),
        None)
    quarter_check = 'quarter' in data.columns
    half_check = 'half' in data.columns
    month_check = 'month' in data.columns
    main_metric = ner_filter['main_metric']
    ascend = _get_sort_order(metric_df, main_metric[0], rank_check)
    most_recent_time = str(data['time'].max())
    value_cols = [value_col]
    if growth_col:
        value_cols.append(growth_col)

    pivot_data = data.pivot_table(index=dim_col, columns=[metric_col, time_col], values=value_cols, aggfunc='first')
    pivot_data.columns = ['{} {} {}'.format(col[1], col[2], col[0]) for col in pivot_data.columns]
    pivot_data = pivot_data.reset_index()
    # dim_col_order = data[dim_col].unique().tolist()
    dim_col_order = {col: data[col].unique().tolist() for col in dim_col}
    # pivot_data[dim_col] = pd.Categorical(pivot_data[dim_col], categories=dim_col_order, ordered=True)
    for col in dim_col:
        pivot_data[col] = pd.Categorical(pivot_data[col], categories=dim_col_order[col], ordered=True)
    if month_check:
        most_recent_year = data['year'].max()
        most_recent_month = data.loc[data['year'] == most_recent_year, 'month'].max()
        target_col = f'month {most_recent_month} {most_recent_year}'
        sort_col_m = [col for col in pivot_data.columns if target_col in col]
        if sort_col_m:
            pivot_data = pivot_data.sort_values(by=dim_col + [sort_col_m[0]], ascending=ascend)
    else:
        pivot_data = pivot_data.sort_values(by=dim_col, ascending=ascend)
    pivot_data.columns = pivot_data.columns.str.replace(' value', '')
    pivot_data.columns = pivot_data.columns.str.replace(' yoy_percent_change', ' YOY')
    pivot_data.columns = pivot_data.columns.str.replace(' qoq_percent_change', ' QOQ')
    pivot_data.columns = pivot_data.columns.str.replace(' hoh_percent_change', ' HOH')

    dim_mapping = {'tp': 'TP', 'ahw': 'AHW', 'led': 'LED', 'pen': 'PEN', 'clv': 'CLV'}
    for col in dim_col:
        pivot_data[col] = pivot_data[col].apply(lambda x: _dim_value_map(x, dim_mapping))
    static_columns = pivot_data.columns[:len(dim_col)].tolist()
    dynamic_columns = [col for col in pivot_data.columns if col not in static_columns]
    sorted_dynamic_columns = _reorder_columns(dynamic_columns, main_metric, quarter_check, month_check, half_check)
    sorted_columns = static_columns + sorted_dynamic_columns
    pivot_data = pivot_data[sorted_columns]

    def _sort_condition(col):
        def contains_main_metric(col):
            return any(metric in col.lower() for metric in main_metric)

        if sort_by_col != 'value':
            return (most_recent_time in col) and contains_main_metric(col) and (
                    'yoy' in col.lower() or 'qoq' in col.lower() or 'hoh' in col.lower())
        else:
            return (most_recent_time in col) and contains_main_metric(col) and not (
                    'yoy' in col.lower() or 'qoq' in col.lower() or 'hoh' in col.lower())

    pivot_sort_cols = [col for col in pivot_data.columns if
                       _sort_condition(col) and 'share of spend' not in col.lower()]
    pivot_sort_col = None
    if pivot_sort_cols and not month_check:
        pivot_sort_col = pivot_sort_cols[0]
        pivot_data = pivot_data.sort_values(by=pivot_sort_col, ascending=ascend)
        multi_value_cols = {col: data[col].nunique() for col in dim_col if data[col].nunique() > 1}
        if multi_value_cols:
            pivot_data['sort_helper'] = pivot_data.groupby(dim_col).cumcount()
            pivot_data = pivot_data.sort_values(by=dim_col + ['sort_helper'])
            # pivot_data = pivot_data.groupby(dim_col).head(5)
            pivot_data = pivot_data.drop(columns=['sort_helper'])
    original_columns = pivot_data.columns.tolist()

    metric_col_index = len(dim_col)
    pivot_data = add_unit_sign(pivot_data, metric_col_index, to_mm_metrics=["spend", "activity"], client_code=client_code, model_group_id=model_group_id)



    # def transform_to_mm(df):
    #     for col in pivot_data.columns[len(dim_col):]:
    #         if 'share of spend' not in col and 'YOY' not in col and 'cost per' not in col and 'QOQ' not in col and 'HOH' not in col:
    #             # if 'spend' in col or 'response' in col or 'activity' in col:
    #             if 'spend' in col or 'activity' in col:
    #                 df[col] = df[col] / 1000000  # Convert to millions
    #                 df[col] = df[col].apply(lambda x: f"{x:.2f}" if not pd.isna(x) else x)
    #                 df.rename(columns={col: col + ' (MM)'}, inplace=True)
    #     return df

    # pivot_data = transform_to_mm(pivot_data)
    if pivot_sort_col in original_columns:
        renamed_cols = {old_col: new_col for old_col, new_col in zip(original_columns, pivot_data.columns) if
                        old_col != new_col}
        pivot_sort_col = renamed_cols.get(pivot_sort_col, pivot_sort_col)


    # yoy add %
    # def append_percentage(df):
    #     # Iterate over columns and apply transformation if conditions are met
    #     for col in pivot_data.columns[len(dim_col):]:
    #         if 'YOY' in col or 'share of spend' in col or 'contribution' in col or 'source of change' in col.lower() or 'QOQ' in col or 'HOH' in col:
    #             df[col] = pd.to_numeric(df[col], errors='coerce')
    #             df[col] = df[col].apply(lambda x: f"{x:.1f}%" if not pd.isna(x) else x)
    #     return df
    #
    # pivot_data = append_percentage(pivot_data)
    #
    # # sales add $, add ,
    # def format_sales_values(df):
    #     for col in pivot_data.columns[len(dim_col):]:
    #         if (('sales' in col and 'response' not in col) or ('gross adds' in col) or ('cost per kpi' in col)) and (
    #                 'YOY' not in col and 'QOQ' not in col and 'HOH' not in col):
    #             # df[col] = df[col].apply(lambda x: f"${x:,.2f}" if not pd.isna(x) else x)
    #             df[col] = df[col].apply(lambda x: f"{x:,.0f}" if not pd.isna(x) else x)
    #     return df
    #
    # pivot_data = format_sales_values(pivot_data)
    #
    # def add_dollar_sign(df):
    #     for col in pivot_data.columns[len(dim_col):]:
    #         if 'spend' in col and 'share of spend' not in col and 'YOY' not in col and 'QOQ' not in col and 'HOH' not in col:
    #             df[col] = df[col].apply(lambda x: f"${float(x):,.2f}" if not pd.isna(x) else x)
    #     return df
    #
    # pivot_data = add_dollar_sign(pivot_data)

    # def rename_change_columns(col):
    #     if re.search(r'(YOY|QOQ|HOH|MOM)', col):
    #         col_clean = re.sub(r'\s*(YOY|QOQ|HOH|MOM)', '', col)
    #         return f"% Change in {col_clean}"
    #     return col
    #
    # pivot_data.columns = [rename_change_columns(col) for col in pivot_data.columns]    # column name uppercase
    def custom_title_case(column_name):
        words = column_name.split()
        final_words = [word.upper() if word in ['YOY', '(MM)', 'roi', 'QOQ', 'HOH'] else word.title() for word in words]
        return ' '.join(final_words)
    pivot_data.columns = [custom_title_case(col) for col in pivot_data.columns]
    return pivot_data


def convert_display_format(text):
    # Pattern to find numbers with more than 7 digits
    pattern = r'\b\d{7,}\.?\d*\b'

    # Function to replace each match
    def replace_with_millions(match):
        number = float(match.group())
        # Convert to millions and format to two decimal places with "MM"
        millions = round(number / 1_000_000, 2)
        return f"{millions} MM"

    # Replace all found numbers with their 'million' equivalents
    res = re.sub(pattern, replace_with_millions, text)

    res = re.sub(r' {2,}', ' ', res)
    res = re.sub(r'\$', '\\$', res)
    res = re.sub(r'#', '', res)

    return res


# biz rule: paid search response and cost per data is non-legit, need to filter out
def _filter_paid_search(intention, data):
    if intention in ['response', 'cost per']:
        data = data[data['media_channel'] != 'paid search']
    return data


# biz rule: exclude pillar data for non-halo and particular tactics/non product media queries
def _filter_pillar(product_halo_check, core_dimension, data):
    if product_halo_check == 'irrelevant' and core_dimension:
        data = data[data['custom_aggregated'] != 'p']
    return data


# biz rule: only keep special media when halo has ahw
def _filter_special_media(product_halo_check, data):
    if 'ahw' not in product_halo_check:
        data = data[data['media_channel'] != 'special media']
    return data


# biz rule: when condition meets, use pillar data to show aggregated values
def _get_pillar_data(period_type, product_media_check, core_dimension, data):
    pillar_data = pd.DataFrame()
    if product_media_check and not core_dimension:
        if 'month' in period_type:
            pillar_data = data[data['tactics'] == 'pillars']
        else:
            pillar_data = data[data['tactics'] == 'comprehensive pillars']
        pillar_data['tactics_concat'] = pillar_data['tactics_detail']
        pillar_data['tactics_concat_origin'] = pillar_data['tactics_detail_origin']
        if not pillar_data.empty:
            data = data[~data['tactics'].isin(['pillars', 'comprehensive pillars'])]
    # for national and language queries only
    agg_data = data[data['tactics'].isin(['national', 'language'])]
    data_copy = data[~data['tactics'].isin(['national', 'language'])]
    if data_copy.empty:
        agg_data = pd.DataFrame()
    else:
        data = data_copy
    if not agg_data.empty:
        pillar_data = agg_data
    return pillar_data, data


# biz rule: product media (in tactics_detail col) as product when condition meets
def _switch_dimension_col(ner_dict, data):
    dim_col = ner_dict['dimension'][0]
    prod_ls = data[dim_col].unique().tolist()
    if (data['custom_aggregated'] == 'p').all():
        if prod_ls == ['overall']:
            data[dim_col] = data['tactics_detail']
        else:
            data[dim_col] = data[dim_col] + ' - ' + data['tactics_detail']
        ner_dict['driver'] = 'tactics'
    return data, ner_dict


def _calc_growth(period_type, ner_dict, data):
    if 'quarter' in period_type:
        data = _calc_qoq(data, ner_dict)
    elif 'year' in period_type:
        data = _calc_yoy(data, ner_dict)
    elif 'month' in period_type:
        data = _calc_mom(data, ner_dict)
    elif 'half' in period_type:
        data = _calc_hoh(data, ner_dict)
    return data


def _get_growth_sort_col(trend_check, data):
    growth_col = next(
        (col for col in ['yoy_percent_change', 'qoq_percent_change', 'hoh_percent_change'] if col in data.columns),
        None)
    sort_by_col = growth_col if (growth_col and trend_check == 'yes') else 'value'
    return growth_col, sort_by_col


# biz rule: ignore growth outlier
def _ignore_growth_outlier(growth_col, data):
    if growth_col and not data.empty:
        data.loc[abs(data[growth_col]) > 1000, growth_col] = np.nan
    return data


def _merge_planner(intention, media_channel_check, core_dimension, core_dimension_composite, data, planner_driver_data):
    try:
        if intention in ['margin roi', 'performance'] and not media_channel_check in ['all', 'irrelevant'] and (
                core_dimension or core_dimension_composite) and not planner_driver_data.empty:
            if (data['custom_aggregated'] == 'p').all():
                sub_data = data[['time', 'tactics', 'tactics_detail']].drop_duplicates()
                planner_data = pd.merge(planner_driver_data, sub_data, how='inner',
                                        left_on=['time', 'product', 'to_driver'],
                                        right_on=['time', 'tactics_detail', 'tactics'])
                planner_data = planner_data[
                    ['time', 'product', 'from_driver', 'to_driver', 'planning']].drop_duplicates()
            else:
                sub_data = data[data['product'] == 'overall'][['time', 'product', 'tactics_detail']].drop_duplicates()
                planner_data = pd.merge(planner_driver_data, sub_data, how='inner',
                                        left_on=['time', 'product', 'to_driver'],
                                        right_on=['time', 'product', 'tactics_detail'])
                planner_data = planner_data[
                    ['time', 'product', 'from_driver', 'to_driver', 'planning']].drop_duplicates()
        else:
            planner_data = pd.DataFrame()
    except Exception as e:
        logger.warning(f"An error occurred while processing data: {e}")
        planner_data = pd.DataFrame()
    return planner_data


def _merge_benchmark(intention, media_channel_check, core_dimension, core_dimension_composite, ner_dict, data, top_data,
                     benchmark_idx):
    if intention in ['margin roi', 'performance'] and not media_channel_check in ['all', 'irrelevant'] and (
            core_dimension or core_dimension_composite) and (
            'margin roi index' in top_data['metric'].unique().tolist()) and not benchmark_idx.empty:
        dim_col = ner_dict['dimension']
        benchmark_data_text = _benchmark_data(top_data, benchmark_idx, pd.DataFrame(), pd.DataFrame(), ner_dict)
        benchmark_str = _benchmark_to_text(benchmark_data_text, ner_dict)
        benchmark_data_display = _benchmark_data(data, benchmark_idx, pd.DataFrame(), pd.DataFrame(), ner_dict)
        benchmark_data = _benchmark_table_display(benchmark_data_display, ner_dict)
        benchmark_data = benchmark_data.sort_values(by=['Media_Channel', 'Margin ROI Index'], ascending=[True, False])
        benchmark_data = _drop_single_dim(dim_col, benchmark_data)
    else:
        benchmark_str = ""
        benchmark_data = pd.DataFrame()
    return benchmark_str, benchmark_data


def _get_trend_data(growth_col, trend_check, rank_check, how_many, ner_dict, ner_filter, data):
    if growth_col and trend_check == 'yes' and not data.empty:
        top_num = int(how_many) if how_many != 'na' else 3
        if rank_check == 'top':
            top_inc_data = _top_increase_data(data, ner_dict, growth_col, ner_filter['main_metric'], top_num)
            top_dec_data = pd.DataFrame()
        elif rank_check == 'bottom':
            top_dec_data = _top_decrease_data(data, ner_dict, growth_col, ner_filter['main_metric'], top_num)
            top_inc_data = pd.DataFrame()
        else:
            top_inc_data = _top_increase_data(data, ner_dict, growth_col, ner_filter['main_metric'], top_num)
            top_dec_data = _top_decrease_data(data, ner_dict, growth_col, ner_filter['main_metric'], top_num)
            if len([d for d in top_inc_data[ner_dict['driver']].unique().tolist() if
                    d in top_dec_data[ner_dict['driver']].unique().tolist()]) >= 1:
                top_dec_data = pd.DataFrame()
    else:
        top_inc_data, top_dec_data = pd.DataFrame(), pd.DataFrame()
    return top_inc_data, top_dec_data


def _filter_top(group, top_num, value_col, main_metric):
    rank_metric_data = group[group['metric'].isin(main_metric)]
    most_recent_year = rank_metric_data['time'].max()
    most_recent_data = rank_metric_data[rank_metric_data['time'] == most_recent_year]
    if len(most_recent_data) < top_num:
        return most_recent_data.nlargest(len(most_recent_data), value_col)
    else:
        return most_recent_data.nlargest(top_num, value_col)


def _filter_top_media_channel(driver_high, how_many, ner_filter, ner_dict, sort_by_col, top_data):
    if driver_high == 'media_channel':
        dim_col, driver_col = ner_dict['dimension'], ner_dict['driver']
        top_num = 3 if how_many == 'na' else int(how_many)
        # Apply the function to each group
        if top_data['media_channel'].nunique() > 1:
            top_drivers = top_data.groupby(dim_col).apply(
                lambda x: _filter_top(x, int(top_num), sort_by_col, ner_filter['main_metric'])).reset_index(
                drop=True)
            # Filter original data to include only top drivers across all groups
            top_data = top_data.merge(top_drivers[dim_col + [driver_col]],
                                      on=dim_col + [driver_col], how='inner')
    return top_data

def _metrics_aka(str_q, ner_filter):
    metrics_aka_str = ""
    acronym_metric = ner_filter.get('acronym', {})
    main_metric = ner_filter.get('main_metric', [])
    if main_metric and acronym_metric:
        metrics_aka_list = []
        for m in main_metric:
            m = 'cost per' if m == 'cost per activity' else m
            metrics_aka = [key for key, value in acronym_metric.items() if m in value]
            if metrics_aka:
                metrics_aka_list.append(f"'{m}' aka '{metrics_aka[0]}'. ")
        metrics_aka_str = " ".join(metrics_aka_list)
    str_q = metrics_aka_str + str_q
    return str_q

def _get_readout(intention, growth_col, trend_check, rank_check, sort_metric, how_many, ner_dict, ner_filter, data,
                 top_data, metric_df):
    def get_verbiage(metric_row, col_name):
        if not len(metric_row):
            return ""
        val = metric_row[col_name].values[0] if col_name in metric_row else ''
        return '' if pd.isna(val) or val == '' else val
    def aka_substring(sub_str):
        # aka_phrase = re.findall(r'\b[^.()]+\s*\(aka [^)]+\)', sub_str)
        aka_phrase = re.findall(r'\b[^.()]*?(?:\([^)]*\))?\s*\(aka [^)]+\)', sub_str)
        if aka_phrase:
            for phrase in aka_phrase:
                sub_str = sub_str.replace(phrase, '').strip()
        sub_str = re.sub(r'\.\s*\.\s*\.+', '.', sub_str)  # Fix . . . to .
        sub_str = re.sub(r'(^|\s)\.(\s|$)', ' ', sub_str)  # Remove isolated periods
        sub_str = re.sub(r'\s{2,}', ' ', sub_str).strip()  # Normalize whitespace
        # sub_str = re.sub(r'\s+\.', '.', sub_str)
        aka_str = ". ".join(aka_phrase) + '.' if aka_phrase else ''
        return aka_str, sub_str
    str_q = ""
    top_inc_data, top_dec_data = _get_trend_data(growth_col, trend_check, rank_check, how_many, ner_dict, ner_filter,
                                                 data)
    metric_row = metric_df[metric_df['metric'] == ner_filter['main_metric'][0]]
    if not top_inc_data.empty:
        top_inc_data = top_inc_data[top_inc_data['metric'].isin(sort_metric)]
        str_q_inc = _get_granular_level_pivot(top_inc_data, ner_dict, ner_filter, trend_check, metric_df)
        open_verbiage = get_verbiage(metric_row, 'trend_top_verbiage')
        aka_str, str_q_inc = aka_substring(str_q_inc)
        str_q += open_verbiage + str_q_inc if str_q_inc else ""
        str_q = aka_str + str_q
    if not top_dec_data.empty:
        top_dec_data = top_dec_data[top_dec_data['metric'].isin(sort_metric)]
        str_q_dec = _get_granular_level_pivot(top_dec_data, ner_dict, ner_filter, trend_check, metric_df)
        open_verbiage = get_verbiage(metric_row, 'trend_bottom_verbiage')
        aka_str, str_q_dec = aka_substring(str_q_dec)
        str_q += open_verbiage + str_q_dec if str_q_dec else ""
        str_q = aka_str + str_q

    if not top_data.empty and (trend_check != 'yes' or (growth_col and top_data[growth_col].isna().all()) or not str_q):
        sub_str = _get_granular_level_pivot(top_data, ner_dict, ner_filter, trend_check, metric_df)
        aka_str, sub_str = aka_substring(sub_str)
        col_name = 'notrend_bottom_verbiage' if rank_check == 'bottom' else 'notrend_top_verbiage'
        open_verbiage = get_verbiage(metric_row, col_name)
        str_q += open_verbiage + sub_str if sub_str else ""
        str_q = aka_str + str_q
    return str_q

def _get_readout_com(readout, readout_dict, result_combo_match, data_type):
    if readout:
        if 'remove_noshow_readout' in result_combo_match and result_combo_match['remove_noshow_readout'] == 'Y':
            agg_data = result_combo_match['agg_view'].split('_')[-1]
            readout_dict[agg_data]['readout'] = _remove_noshow_readout(readout_dict[agg_data]['readout'])
        readout = [ro.split('_')[-1] for ro in readout.split(',')]
        readout = [readout_dict[ro]['readout'] for ro in readout]
        if all(r.strip() == '' for r in readout):
            return ''
        if data_type=='bi':
            return "At Tactics level: ".join(readout)
        else:
            return "More granularly: ".join(readout)
    else:
        return readout


def _remove_noshow_readout(texts):
    seen_statements = {}
    cutoff_index = None
    lines = texts.split("\n")
    if lines:
        for i, line in enumerate(lines):
            parts = line.split(":", 1)
            if len(parts) > 1:
                statement = parts[0].strip()
                if statement in seen_statements:
                    cutoff_index = i
                    break
                else:
                    seen_statements[statement] = True
        cleaned_text = "\n".join(lines[:cutoff_index]) if cutoff_index is not None else texts
    else:
        cleaned_text = texts
    return cleaned_text

def _add_na_term(core_dimension, readout):
    no_data_term = find_top_level_empty_keys(core_dimension)
    if no_data_term:
        suffix = ', '.join(no_data_term) + " not available in data."
        readout = (readout or '') + suffix
    return readout


def _get_metrics_rename(pivot_sort_col, metric_df, detail_view):
    if pivot_sort_col:
        matching_rows = metric_df[metric_df['metric'].apply(lambda x: pivot_sort_col.lower().startswith(x))]
        rename = matching_rows['rename'].iloc[0] if not matching_rows.empty else ""
        rename = "" if pd.isna(rename) else rename
        original_metric = matching_rows['metric'].iloc[0] if not matching_rows.empty else ""
        if not detail_view.empty and rename:
            detail_view.columns = detail_view.columns.str.replace(original_metric, rename.title(), case=False,
                                                                  regex=True)
    return detail_view


def _rename_product_media(product_halo_check, pivot_table):
    if not pivot_table.empty and 'Product' in pivot_table.columns:
        pivot_table['Product Media'] = pivot_table['Product'].str.split(' - ').str[1]
        if pivot_table['Product Media'].isna().all():
            pivot_table = pivot_table.drop(columns='Product Media')
            if product_halo_check != 'irrelevant':
                pivot_table = pivot_table.rename(columns={'Product': 'Product Media'})
        else:
            pivot_table['Product'] = pivot_table['Product'].str.split(' - ').str[0]
    if 'Product Media' in pivot_table.columns:
        cols_order = ['Product Media'] + [col for col in pivot_table.columns if col != 'Product Media']
        pivot_table = pivot_table[cols_order]
    return pivot_table


def _rename_product_media_other_category(ner_filter, dim_col, pivot_table):
    if 'product_focused_media' in ner_filter and dim_col.capitalize() in pivot_table.columns:
        pivot_table = pivot_table.rename(columns={dim_col.capitalize(): 'Product Media'})
    elif 'product_focused_media' not in ner_filter:
        pivot_table = pivot_table.rename(columns={'Product': 'Channel'})
    return pivot_table


def _core_dim_readout(search_level, search_level_com, pivot_table_m_com, pivot_table_ag, pivot_table,
                      pillar_pivot_table, str_q_ag, str_q_m_com, str_q):
    if search_level['measure'] or search_level_com['measure']:
        if pivot_table_m_com.empty:
            if not pivot_table_ag.empty:
                pillar_pivot_table = pivot_table_ag
                if str_q_ag != str_q:
                    str_q = str_q_ag + "At tactics level: " + str_q
        else:
            pillar_pivot_table = pivot_table
            pivot_table = pivot_table_m_com
            if str_q_m_com != str_q:
                str_q = str_q + "At tactics level: " + str_q_m_com
    else:
        if not pivot_table_ag.empty:
            pillar_pivot_table = pivot_table_ag
            if str_q != str_q_ag:
                str_q = str_q_ag + "At tactics level: " + str_q
    if str_q:
        str_q = str_q.replace("Top performing tactics: ", "")
    return str_q, pillar_pivot_table, pivot_table


def _colgate_specific_readout_reformat(str_q):
    unwanted_str = ['marketing', 'media', 'language', 'national', 'comprehensive tactic', 'comprehensive pillars',
                    'pillars']
    str_pattern = r'\b(' + '|'.join(re.escape(word) for word in unwanted_str) + r')\s*\((.*?)\)'
    str_q = re.sub(str_pattern, r'\2', str_q)
    return str_q


# TO DO: can be combined with colgate func
def _hilsp_specific_readout_reformat(str_q):
    unwanted_str = ['brand focus', 'campaign totals', 'marketing type', 'marketing type with brand detail',
                    'total marketing', 'total shopper', 'shopper marketing type']
    str_pattern = r'\b(' + '|'.join(re.escape(word) for word in unwanted_str) + r')\s*\((.*?)\)'
    str_q = re.sub(str_pattern, r'\2', str_q)
    return str_q

def _readout_reformat(str_q, lookup_df):
    if lookup_df:
        lookup_df_setting = lookup_df['setting'].fillna("")
        filtered_values = lookup_df_setting.loc[~lookup_df_setting['value'].isin(['yes', 'no']), 'value']
        if not filtered_values.empty:
            v_ls = [v.replace("'", "").split(',') for v in filtered_values.values]
            # deduplicate
            unwanted_str = list({str.strip() for ls in v_ls for str in ls})
            str_pattern = r'\b(' + '|'.join(re.escape(word) for word in unwanted_str) + r')\s*\((.*?)\)'
            str_q = re.sub(str_pattern, r'\2', str_q)
    return str_q


def _get_agg_data_hilsp(product_media_check, data):
    unwanted_str = ['brand focus', 'campaign totals', 'marketing type', 'marketing type with brand detail',
                    'total marketing', 'total shopper', 'shopper marketing type']
    if 'yes' in data['custom_aggregated'].unique().tolist() and not product_media_check:
        agg_data = data[data['custom_aggregated'] != 'no']
        data = data[data['custom_aggregated'] == 'no']
        higher_agg = agg_data[agg_data['tactics'].isin(unwanted_str)]
        if not higher_agg.empty:
            agg_data = higher_agg
    elif product_media_check:
        agg_data = data[data['tactics'].isin(unwanted_str)]
        data = data[~data['tactics'].isin(unwanted_str)]
    else:
        agg_data = pd.DataFrame()
    return agg_data, data

def _get_diag_agg(data, file_type, diag_tagging):
    if diag_tagging and file_type=='mg':
        diag_data = data.copy()
        diag_data = diag_data[diag_data["custom_aggregated"] == 'diagnostic']
        return diag_data, True
    else:
        return data, False

def _get_agg_data(data, agg_view_df, ner_dict):
    if agg_view_df:
        setting_df = agg_view_df['setting'].copy()
        mapping_df = agg_view_df['mapping'].copy()
        rule_match = {}
        for idx, row in setting_df.iterrows():
            if 'rule' in row['idx']:
                col = row['column']
                value_ls = row['value'].replace("'", "").split(',')
                value_ls = [v.strip() for v in value_ls]
                if row['condition_or_action'] == 'any':
                    rule_match[row['idx']] = 'Y' if data[col].isin(value_ls).any() else 'N'
                elif row['condition_or_action'] == 'all':
                    rule_match[row['idx']] = 'Y' if data[col].isin(value_ls).all() else 'N'
        rules = [col for col in mapping_df if 'rule' in col]
        mapping_df['rule_key'] = mapping_df[rules].astype(str).agg('~'.join, axis=1)
        r = '~'.join([rule_match[rule] for rule in rules])
        if r in mapping_df['rule_key'].tolist():
            agg_view = mapping_df[mapping_df['rule_key'] == r]['agg_view'].values[0]
            detail_view = mapping_df[mapping_df['rule_key'] == r]['detail_view'].values[0]
        else:
            return pd.DataFrame(), data, None, ner_dict
        def view_idx_to_df(idx):
            if not pd.isna(idx):
                view_rule = setting_df[setting_df['idx'] == idx]
                c = view_rule['column'].values[0]
                v_ls = view_rule['value'].values[0].replace("'", "").split(',')
                v_ls = [v.strip() for v in v_ls]
                if view_rule['condition_or_action'].values[0] == 'include':
                    return data[data[c].isin(v_ls)]
                else:
                    return data[~data[c].isin(v_ls)]
            else:
                return pd.DataFrame(columns=data.columns)

        agg_data = view_idx_to_df(agg_view)
        data = view_idx_to_df(detail_view)
        if 'concat_dimension' in mapping_df.columns:
            if_switch_dimension = mapping_df[mapping_df['rule_key'] == r]['concat_dimension'].values[0]
            if if_switch_dimension == 'Y':
                dim_col = ner_dict['dimension'][0]
                prod_ls = data[dim_col].unique().tolist()
                if prod_ls == ['overall']:
                    data[dim_col] = data['tactics_detail']
                else:
                    data[dim_col] = data[dim_col] + ' - ' + data['tactics_detail']
                ner_dict['driver'] = 'tactics'
        return agg_data, data, (agg_view, detail_view), ner_dict
    else:
        return pd.DataFrame(), data, None, ner_dict


def _drop_single_dim(dim_cols, pivot_table):
    if not pivot_table.empty:
        cols_to_drop = [col for col in pivot_table.columns if
                        (col.lower() in dim_cols or col in dim_cols) and pivot_table[col].unique().tolist() == [
                            'Overall']]
        cols_to_drop = cols_to_drop+['Core_Dimension_Term']
        pivot_table = pivot_table.drop(columns=cols_to_drop, errors='ignore')
    return pivot_table


def _rename_unwanted_tactics(pivot_table):
    unwanted_str = ['marketing', 'media', 'language', 'national', 'comprehensive tactic', 'comprehensive pillars',
                    'pillars']
    table_pattern = r'(?i)\b(' + '|'.join(map(re.escape, unwanted_str)) + r')\s*\((.*?)\)'
    if not pivot_table.empty and 'Tactics' in pivot_table.columns:
        pivot_table['Tactics'] = pivot_table['Tactics'].str.replace(table_pattern, r'\2', regex=True)
    return pivot_table


def _hilsp_specific_table_format(product_media_check, pivot_table, pivot_biz_table, agg_data):
    unwanted_str = ['brand focus', 'campaign totals', 'marketing type', 'marketing type with brand detail',
                    'total marketing', 'total shopper', 'shopper marketing type']
    if product_media_check and not pivot_table.empty and 'Tactics' in pivot_table.columns:
        pivot_table['Marketing Focus'] = pivot_table['Tactics'].str.extract(r'^(.*?)\s*\(')
        pivot_table['Tactics'] = pivot_table['Tactics'].str.extract(r'\((.*?)\)')
        cols = list(pivot_table.columns)
        tactics_index = cols.index('Tactics')
        cols.insert(tactics_index, cols.pop(cols.index('Marketing Focus')))
        pivot_table = pivot_table[cols]

    if not agg_data.empty and not pivot_biz_table.empty and 'Tactics' in pivot_biz_table.columns:
        pivot_biz_table['Marketing Focus'] = pivot_biz_table['Tactics'].str.extract(r'^(.*?)\s*\(')
        pivot_biz_table['Tactics'] = pivot_biz_table['Tactics'].str.extract(r'\((.*?)\)')
        cols = list(pivot_biz_table.columns)
        tactics_index = cols.index('Tactics')
        cols.insert(tactics_index, cols.pop(cols.index('Marketing Focus')))
        pivot_biz_table = pivot_biz_table[cols]
        if pivot_biz_table['Marketing Focus'].str.lower().isin(unwanted_str).any():
            pivot_biz_table = pivot_biz_table.drop(columns=['Marketing Focus'])

    return pivot_table, pivot_biz_table


def _agg_table_format(pivot_table, pivot_biz_table, lookup_df, views, concatenate, raw_col_name='Tactics'):
    if lookup_df:
        lookup_df_setting = lookup_df['setting'].fillna("")
        lookup_df_mapping = lookup_df['mapping']
        def rename_df(df, rename_col):
            if concatenate == 'yes':
                df[rename_col] = df[raw_col_name].str.extract(r'^(.*?)\s*\(')
                df[raw_col_name] = df[raw_col_name].str.extract(r'\((.*?)\)')

                cols = list(df.columns)
                tactics_index = cols.index(raw_col_name)
                cols.insert(tactics_index, cols.pop(cols.index(rename_col)))
                df = df[cols]
            else:
                df=df.rename(columns={raw_col_name:rename_col})
            return df

        if views:
            agg_view, detail_view = views
            agg_rename = lookup_df_setting[lookup_df_setting['idx'] == agg_view][f'{raw_col_name.lower()}_col_rename'].values[0] if not pd.isna(agg_view) else None
            detail_rename = lookup_df_setting[lookup_df_setting['idx'] == detail_view][f'{raw_col_name.lower()}_col_rename'].values[0] if not pd.isna(detail_view) else None
            if agg_rename and not pivot_biz_table.empty:
                # rename agg
                pivot_biz_table = rename_df(pivot_biz_table, agg_rename)
                drop_condition = lookup_df_mapping[(lookup_df_mapping['agg_view'] == agg_view) &
                                                   ((lookup_df_mapping['detail_view'].isna() if pd.isna(detail_view)
                                                     else lookup_df_mapping['detail_view'] == detail_view))]
                if not drop_condition.empty and drop_condition['drop_agg_rename_col'].values[0] == 'Y':
                    pivot_biz_table = pivot_biz_table.drop(columns=[agg_rename])

            if detail_rename and not pd.isna(detail_rename) and not pivot_table.empty:
                if not concatenate:
                    if pivot_table[raw_col_name].str.contains(r'^.+ - .+$', regex=True).all():
                        concatenate = 'yes'
                        pivot_table[raw_col_name] = pivot_table[raw_col_name].apply(
                            lambda x: x.split(' - ')[1] + '(' + x.split(' - ')[0] + ')')

                pivot_table = rename_df(pivot_table, detail_rename)

    return pivot_table, pivot_biz_table

def _get_dim_col_in_pivot_table(df_dict: dict[str, pd.DataFrame], ner_filter: dict):
    dfs = ['ag_df', 'mg_df', 'm_df']
    dim_col = None
    for df in dfs:
        curr_df = df_dict[df]
        if not curr_df.empty:
            data = _read_and_process_csv(curr_df, ner_filter)
            if 'tactics' in data.columns:
                dim_col = _get_dimension_cols(data)
            elif 'business_driver' in data.columns:
                dim_col = _get_dimension_cols(data, idx_col='business_driver')
            return dim_col
    return None

def _get_dim_col_in_pivot_table_unique(df_dict: dict[str, pd.DataFrame], ner_filter: dict, client_code: str):
    dfs = ['ag_df', 'mg_df', 'm_df']
    dim_col = None
    for df in dfs:
        curr_df = df_dict[df]
        if not curr_df.empty:
            data = _read_and_process_csv(curr_df, ner_filter)
            if 'tactics' in data.columns:
                dim_col = _get_dimension_cols_unique(data, client_code, ner_filter)
            elif 'business_driver' in data.columns:
                dim_col = _get_dimension_cols_unique(data, client_code, ner_filter, idx_col='business_driver')
            return dim_col
    return None


def _dynamic_top(dim_cols, pivot_sort_col, pivot_table, metric_df, rank_check='na', total_entry=20):
    if not pivot_table.empty:
        dim_col = [col for col in pivot_table.columns if (col.lower() in dim_cols or col in dim_cols)]
        # unique_combinations = pivot_table[dim_col].drop_duplicates().shape[0]
        # if unique_combinations > 1 and total_entry>unique_combinations:
        #     # unique_num = pivot_table[dim_col].nunique()
        #     row_per_group = total_entry // unique_combinations
        #     pivot_table = pivot_table.groupby(dim_col).head(row_per_group)
        # elif unique_combinations > 1 and total_entry<=unique_combinations:
        #     pivot_table = pivot_table.groupby(dim_col).head(5)
        if pivot_sort_col:
            pivot_table["sort_helper"] = (
                pivot_table[pivot_sort_col]
                .astype(str)
                .str.replace(r"[$%,]", "", regex=True)
                .astype(float)
            )

            if "source of change" in pivot_sort_col.lower():  # or "contribution" in pivot_sort_col.lower():
                # Sort by absolute values
                pivot_table = pivot_table.sort_values(
                    by=dim_col + ["sort_helper"],
                    ascending=[True for i in dim_col] + [False],
                    key=lambda col: col.abs() if col.name == "sort_helper" else col
                )
            else:
                matching_rows = metric_df[metric_df['metric'].apply(lambda x: pivot_sort_col.lower().startswith(x))]
                metric = matching_rows['metric'].iloc[0] if not matching_rows.empty else ""
                ascend = _get_sort_order(metric_df, metric, rank_check)
                # ascend = matching_rows['ascend'].iloc[0] if not matching_rows.empty else False
                pivot_table = pivot_table.sort_values(by=dim_col + ["sort_helper"],
                                                      ascending=[True for i in dim_col] + [ascend])
            pivot_table = pivot_table.drop(columns=["sort_helper"])
    return pivot_table


def _pillar_pivot_table_rename(core_dimension_composite, core_dimension, pillar_pivot_table):
    if not core_dimension_composite and not core_dimension and not pillar_pivot_table.empty:
        pillar_pivot_table = pillar_pivot_table.rename(columns={'Tactics': 'Product Media Overall'})
    return pillar_pivot_table


def _colgate_specific_table_reformat(dim_cols, product_halo_check, pivot_sort_col, core_dimension_composite,
                                     core_dimension, pivot_table, pillar_pivot_table, metric_df):
    pivot_table = _dynamic_top(dim_cols, pivot_sort_col, pivot_table, metric_df)
    pivot_table = _drop_single_dim(dim_cols, pivot_table)
    pivot_table = _rename_product_media(product_halo_check, pivot_table)
    pivot_table = _rename_unwanted_tactics(pivot_table)
    pillar_pivot_table = _drop_single_dim(dim_cols, pillar_pivot_table)
    pillar_pivot_table = _rename_unwanted_tactics(pillar_pivot_table)
    pillar_pivot_table = _pillar_pivot_table_rename(core_dimension_composite, core_dimension, pillar_pivot_table)
    return pillar_pivot_table, pivot_table


def _planner_intention_res(ner_filter, planner_driver_data, product_halo_check, media_channel_check, media_channel_ls):
    additional_info_options = ['For additional guidance, budget scenarios can be run in Strategic Planning.',
                               'To pressure test budget changes further, various scenarios can be run in Strategic Planning to quantify business impacts.',
                               'Running budget scenarios in Strategic Planning can help provide further guidance on budgeting decisions.']
    fixtext = ""
    if not planner_driver_data.empty:
        fixtext = "Colgate Toothpaste media optimization opportunities for 2024 are identified as: \n"
        product = ner_filter.get('product', [])
        product_ls = planner_driver_data['product'].unique().tolist()
        if 'optic white' in product_halo_check:
            product_halo_check.append('optic white tp')
        if product_halo_check != 'irrelevant' and any(p in product_ls for p in product_halo_check):
            product = [p for p in product_halo_check if p in product_ls]
        if product == ['overall']:
            planner_filter_data_p = planner_driver_data
        else:
            planner_filter_data_p = planner_driver_data[planner_driver_data['product'].isin(product)]
        if media_channel_check not in ['irrelevant', 'all'] and not planner_filter_data_p.empty:
            planner_filter_data_m = planner_filter_data_p[
                (planner_filter_data_p['from_media_channel'].isin(media_channel_ls)) | (
                    planner_filter_data_p['to_media_channel'].isin(media_channel_ls))]
        else:
            planner_filter_data_m = planner_filter_data_p
        if planner_filter_data_m.empty:
            planner_filter_data = planner_driver_data if planner_filter_data_p.empty else planner_filter_data_p
        else:
            planner_filter_data = planner_filter_data_m
        fixtext += planner_filter_data['planning'].str.cat(sep='\n')
    fixtext = fixtext + '\n' + np.random.choice(additional_info_options)
    str_q, pivot_table, benchmark_data, benchmark_str, planner_data = "", pd.DataFrame(), pd.DataFrame(), "", pd.DataFrame()
    return str_q, fixtext, pivot_table, benchmark_data, benchmark_str, planner_data


def _overall_metric_other_category(ner_filter, driver_high, file_type, data_type, driver_detail, data):
    product_media_check = 'product_focused_media' in ner_filter.keys()
    if data.columns[0] not in ['product', 'category']:
        data['product'] = 'overall'
    else:
        data = data.rename(columns={'product': 'channel'})
    if 'product' in data.columns:
        cols = ['product'] + [col for col in data.columns if col != 'product']
        data = data[cols]
    ner_dict = _get_ner_key_value_other_category(data, driver_high, file_type, data_type, driver_detail)
    if product_media_check:
        ner_dict['dimension'] = ['channel']
    if 'product' in ner_filter:
        product_check = ner_filter['product']
    elif 'category' in ner_filter:
        product_check = ner_filter['category']
    else:
        product_check = ['overall']

    return data, product_check, ner_dict


def _overall_metric(client_code, model_group_id, ner_filter, ner_res_dict, driver_tags, dim_cols, metric_df):
    try:
        file_type, data_type, driver_high, driver_detail = driver_tags
        trend_check, rank_check = ner_res_dict.get('trend', 'no'), ner_res_dict.get('rank', 'na')
        metric = ner_filter['main_metric']
        data = load_total_data(client_code, model_group_id)
        ner_dict = _get_ner_key_value(data, driver_high, file_type, data_type, driver_detail, dim_cols)
        dimension_check = {dim: ner_filter.get(dim, 'overall') for dim in ner_dict['dimension']}

        # dimension_check = ner_filter[ner_dict['dimension']]
        if 3042 <= int(model_group_id) <= 3045:
            data, dimension_check, ner_dict = _overall_metric_other_category(ner_filter, driver_high, file_type,
                                                                             data_type, driver_detail, data)
        metric_col, dim_col, time_col, value_col, period_col = ner_dict['metric'], ner_dict['dimension'], ner_dict[
            'time'], ner_dict['value'], ner_dict['period']
        period_type = ner_filter.get('period_type', ['year'])[0]
        flat = False
        data = data[(data[metric_col].isin(metric)) & (data[time_col].isin(ner_filter['time']))]
        if isinstance(dimension_check, list):
            data = data[data[dim_col[0]].isin(dimension_check)]
        else:
            for col, value in dimension_check.items():
                data = data[data[col].isin(value)]
        data = data.sort_values(by=dim_col + [metric_col, period_col, time_col])
        if 'quarter' in period_type:
            data = _convert_quarter(data, time_col)
            data = _calc_qoq_total(data, ner_dict)
        elif 'year' in period_type:
            data = _convert_time(data, time_col)
            data = _calc_yoy_total(data, ner_dict)
        elif 'half' in period_type:
            data = _convert_half(data, time_col)
            data = _calc_hoh_total(data, ner_dict)
        growth_col, sort_by_col = _get_growth_sort_col(trend_check, data)
        if growth_col in data.columns:
            growth_values = data[growth_col].dropna()
            if not growth_values.empty and (abs(growth_values) < 0.5).all():
                data = data.drop(growth_col, axis=1)
                flat = True
        str_q = _get_total_level_pivot(data, ner_dict, ner_filter, metric_df)
        if flat:
            str_q = str_q.rstrip() + "The value changes over time remain flat.\n"
        pivot_table = _get_total_table(data, ner_dict, ner_filter, rank_check, sort_by_col, metric_df, client_code, model_group_id)
    except Exception as e:
        logger.warning(f"Failed to process overall metrics: {e}")
        str_q = ""
        pivot_table = pd.DataFrame()
    return str_q, pivot_table


def _prepare_overall_driver_data(client_code, model_group_id, ner_filter, driver_tags, dim_dict):
    file_type, data_type, driver_high, driver_detail = driver_tags
    metric = ner_filter['main_metric']
    data = load_total_data(client_code, model_group_id)
    if 'driver' not in data.columns:
        data.insert(loc=len(data.columns) - 4, column='driver', value='overall')

    level, _ = _get_level_tagging(ner_filter)

    if 3042 <= int(model_group_id) <= 3045:
        data, _, ner_dict = _overall_metric_other_category(
            ner_filter, driver_high, file_type, data_type, driver_detail, data)
        dim_cols = _get_dimension_cols(data, 'driver')
    else:
        dim_cols = _get_dimension_cols(data, 'driver')
        data, dim_cols = _level_to_drop(level, dim_cols, data)
        ner_dict = _get_ner_key_value(data, driver_high, file_type, data_type, driver_detail, dim_cols)
    ner_dict['driver'] = 'driver'
    if dim_dict:
        dimension_check = {col: dim_dict.get(col, ["overall"]) for col in dim_cols}
    else:
        dimension_check = {dim: ner_filter.get(dim, ['overall']) for dim in ner_dict['dimension']}
    metric_col, dim_col, time_col, _, period_col = (
        ner_dict['metric'], ner_dict['dimension'], ner_dict['time'],
        ner_dict['value'], ner_dict['period']
    )
    period_type = ner_filter.get('period_type', ['year'])[0]
    data = data[(data[metric_col].isin(metric)) & (data[time_col].isin(ner_filter['time']))]
    for col, value in dimension_check.items():
        if 'overall' in value and 'overall' in data[col].values:
            data = data[data[col].isin(value)]
        elif 'overall' not in value:
            data = data[data[col].isin(value)]

    if data.empty:
        return pd.DataFrame(), ner_dict, dim_cols
    data = data.sort_values(by=dim_col + [metric_col, period_col, time_col])
    # Time-based trend calculation
    if 'quarter' in period_type:
        data = _convert_quarter(data, time_col)
        data = _calc_qoq(data, ner_dict)
    elif 'year' in period_type:
        data = _convert_time(data, time_col)
        data = _calc_yoy(data, ner_dict)
    elif 'half' in period_type:
        data = _convert_half(data, time_col)
        data = _calc_hoh(data, ner_dict)
    elif 'month' in period_type:
        data = _convert_month(data, time_col)
        data = _calc_mom(data, ner_dict)
    return data, ner_dict, dim_cols


def _overall_time_filter(data, time_col, metric):
    if data.empty:
        return pd.DataFrame()
    if 'year' in data.columns:
        most_recent_year = data['year'].max()
        recent_year_values = data[(data['year'] == most_recent_year)]
    else:
        most_recent_year = data[time_col].sort_values(ascending=False).values[0]
        recent_year_values = data[(data[time_col] == most_recent_year)]
        # Sort drivers based on the main_metric value in the most recent year
    if 'quarter' in recent_year_values.columns and metric[0] != 'sales':
        recent_year_values = recent_year_values.sort_values(by=['year', 'quarter'], ascending=False)
        recent_periods = recent_year_values[time_col].unique()[:3].tolist()
    elif 'month' in recent_year_values.columns and metric[0] != 'sales':
        recent_year_values = recent_year_values.sort_values(by=['year', 'month'], ascending=False)
        recent_periods = recent_year_values[time_col].unique()[:3].tolist()
    else:
        recent_periods = None
    if recent_periods:
        data = data[data[time_col].isin(recent_periods)]
    return data

def _overall_metric_w_driver(client_code, model_group_id, ner_filter, ner_res_dict, driver_tags, metric_df, dim_dict):
    trend_check, rank_check = ner_res_dict.get('trend', 'no'), ner_res_dict.get('rank', 'na')
    biz_driver, biz_driver_detail = ner_filter.get('business_driver', []), ner_filter.get('business_driver_detail', [])
    flat = False
    str_q = ""
    pivot_table = pd.DataFrame()
    data, ner_dict, dim_cols = _prepare_overall_driver_data(client_code, model_group_id, ner_filter, driver_tags, dim_dict)
    data = _overall_time_filter(data, ner_dict['time'], ner_filter['main_metric'])
    if data.empty:
        return str_q, pivot_table
    # uniq_dim_data = data[dim_cols].drop_duplicates().shape[0] if not data.empty else 1
    # if uniq_dim_data>=4:
    #     return str_q, pivot_table
    growth_col, sort_by_col = _get_growth_sort_col(trend_check, data)
    if growth_col in data.columns:
        growth_values = data[growth_col].dropna()
        if not growth_values.empty and (abs(growth_values) < 0.5).all():
            data = data.drop(growth_col, axis=1)
            flat = True
    if 'driver' in data.columns:
        driver_ls = data['driver'].unique().tolist()
        if biz_driver and isinstance(biz_driver, list):
            drivers = [d for d in driver_ls if d in biz_driver]
        elif biz_driver_detail and isinstance(biz_driver_detail, list):
            drivers = [d for d in driver_ls if d in biz_driver_detail]
        else:
            drivers = driver_ls
        if not drivers:
            drivers = driver_ls
        for d in drivers:
            sub_data = data[data['driver'] == d]
            str_q_driver = _get_total_level_pivot(sub_data, ner_dict, ner_filter, metric_df)
            str_q += str_q_driver
            sub_pivot_table = _get_total_table(sub_data, ner_dict, ner_filter, rank_check, sort_by_col,
                                               metric_df, client_code, model_group_id)
            sub_pivot_table['Driver'] = d.title()
            pivot_table = pd.concat([pivot_table, sub_pivot_table])
        fixed_pos = len(dim_cols)
        cols = [col for col in pivot_table.columns if col != 'Driver']
        cols.insert(fixed_pos, 'Driver')
        pivot_table = pivot_table[cols]
    if flat:
        str_q = str_q.rstrip() + " The value changes over time remain flat.\n"
    return str_q, pivot_table

def _get_pretext_overall_trend(client_code, model_group_id, ner_filter, ner_res_dict, driver_tags, metric_df, df, driver_col= 'Tactics'):
    trend_check, rank_check = ner_res_dict.get('trend', 'no'), ner_res_dict.get('rank', 'na')
    biz_driver, biz_driver_detail = ner_filter.get('business_driver', []), ner_filter.get('business_driver_detail', [])
    metric = ner_filter['main_metric']
    pivot_table, target_df = pd.DataFrame(), pd.DataFrame()
    tactics_idx = df.columns.get_loc(driver_col)
    left_cols = df.columns[:tactics_idx].tolist()
    if left_cols:
        df = df.rename(columns={col: col.lower() for col in left_cols})
        left_cols = [col.lower() for col in left_cols]
        for col in left_cols:
            df[col] = df[col].str.lower()
        dim_dict = {col: df[col].unique().tolist() for col in left_cols}
    else:
        dim_dict = None
    data, ner_dict, dim_cols = _prepare_overall_driver_data(client_code, model_group_id, ner_filter, driver_tags, dim_dict)
    if data.empty:
        return "", pd.DataFrame()
    growth_col, sort_by_col = _get_growth_sort_col(trend_check, data)
    if growth_col in data.columns:
        growth_values = data[growth_col].dropna()
        if not growth_values.empty and (abs(growth_values) < 0.5).all():
            data = data.drop(growth_col, axis=1)
    if 'driver' in data.columns:
        driver_ls = data['driver'].unique().tolist()
        if biz_driver and isinstance(biz_driver, list):
            drivers = [d for d in driver_ls if d in biz_driver]
        elif biz_driver_detail and isinstance(biz_driver_detail, list):
            drivers = [d for d in driver_ls if d in biz_driver_detail]
        else:
            drivers = driver_ls
        if not drivers:
            drivers = driver_ls
        for d in drivers:
            sub_data = data[data['driver'] == d]
            sub_pivot_table = _get_total_table(sub_data, ner_dict, ner_filter, rank_check, sort_by_col,
                                               metric_df, client_code, model_group_id)
            sub_pivot_table['Driver'] = d.title()
            pivot_table = pd.concat([pivot_table, sub_pivot_table])
        fixed_pos = len(dim_cols)
        cols = [col for col in pivot_table.columns if col != 'Driver']
        cols.insert(fixed_pos, 'Driver')
        pivot_table = pivot_table[cols]

    if not pivot_table.empty:
        # get trend
        metric_cols = [col for col in pivot_table.columns if metric[0] in col.lower() and 'Share' not in col]
        trend_cols = [col for col in metric_cols if any(trend_key in col for trend_key in ['QOQ', 'YOY', 'HOH'])]
        value_cols = [col for col in metric_cols if col not in trend_cols]
        target_df = pivot_table[list(pivot_table.columns[:pivot_table.columns.get_loc('Driver') + 1]) + metric_cols]
        target_df = _clean_numeric_after_tactics(target_df, metric_cols, 'Driver')

        target_df["all_mean_abs"] = target_df[value_cols].astype(float).abs().mean(axis=1)
        target_df = target_df.loc[target_df["all_mean_abs"].notna()]
        target_df[f"{metric[0]} trend type"] = [
            analyze_trend(
                trend_values=row[value_cols].values,
                all_mean_abs=row["all_mean_abs"],
                period_type=ner_filter.get('period_type', ['year'])[0],
                spike_ratio=2,
                flat_multiplier=0.04, )
            for _, row in target_df.iterrows()]
        target_df = target_df.drop(columns=['all_mean_abs'])
        summary = ""
        if not target_df.empty:
            trend_phrase_map = {
                "consistently increasing": [
                    "was consistently increasing",
                    "showed steady growth",
                    "maintained an upward trend",
                    "kept rising steadily"
                ],
                "consistently decreasing": [
                    "was consistently decreasing",
                    "showed a steady decline",
                    "maintained a downward trend",
                    "kept falling steadily"
                ],
                "spiking": [
                    "experienced sharp spikes",
                    "saw sudden surges",
                    "had abrupt peaks"
                ],
                "volatile": [
                    "was volatile",
                    "showed high variability",
                    "fluctuated sharply"
                ],
                "flat": [
                    "remained flat",
                    "stayed stable",
                    "showed little movement"
                ],
                "mild fluctuated (increase)": [
                    "fluctuated mildly with an overall increase",
                    "showed minor fluctuations, trending upward",
                    "moved slightly but ended higher overall"
                ],
                "mild fluctuated (decrease)": [
                    "fluctuated mildly with an overall decrease",
                    "showed minor fluctuations, trending downward",
                    "moved slightly but ended lower overall"
                ]}
            phrases = []
            last_phrase = None
            for d, trend in zip(target_df['Driver'], target_df[f"{metric[0]} trend type"]):
                if trend:
                    options = trend_phrase_map.get(trend, [f"showed a {trend} trend"])
                    phrase = random.choice(options)
                    # retry once if it's the same as last time
                    if phrase == last_phrase and len(options) > 1:
                        phrase = random.choice([p for p in options if p != last_phrase])

                    phrases.append(f"{d.lower()} {metric[0]} {phrase}")
                    last_phrase = phrase
            summary = f"Over the observed periods, {', '.join(phrases)}." if phrases else ""
            # unique_trends = {phrase for _, phrase in trend_pairs}
            #
            # if len(phrases) == 1:
            #     summary = f"Over the observed periods, {phrases[0]}."
            # elif len(unique_trends) == 1:
            #     drivers = [driver for driver, _ in trend_pairs]
            #     summary = f"Over the observed periods, {', '.join(drivers[:-1])} and {drivers[-1]} {metric[0]} {list(unique_trends)[0]}."
            # else:
            #     summary = f"Over the observed periods, {', '.join(phrases[:-1])}, while {phrases[-1]}."
    else:
        summary = ""
    return summary, target_df


def _empty_mg_replace(df_dict, file_mapping_br, curr_df_name, curr_df):
    if curr_df_name == 'mg_df' and curr_df.empty:
        file_type, driver_detail = 'm', 'tactics'
        curr_df_name = file_mapping_br[driver_detail]
        curr_df = df_dict[curr_df_name]
    return curr_df


def _empty_mg_replace_other_category(df_dict, file_mapping_br, curr_df_name, curr_df):
    if curr_df_name == 'mg_df' and curr_df.empty:
        file_type, driver_detail = 'm', 'detail_tactics'
        curr_df_name = file_mapping_br[driver_detail]
        curr_df = df_dict[curr_df_name]
    return curr_df


def _get_fixtext_br(intention, biz_driver_check, media_channel_check, ner_dict1, ner_filter, trend_check, driver_detail,
                    curr_df, data1, metric_df):
    fixtext = ""
    if intention == 'contribution' and 'other' in curr_df['business_driver'].unique().tolist():
        fixtext += 'We only provide contribution by marketing drivers. '

    if intention == 'source of change':
        if (biz_driver_check in [['media'], ['promotions'], ['media', 'promotions']] or (
                biz_driver_check == 'all' and media_channel_check == 'all')) and not data1.empty:
            fixtext += "Key business drivers of sales change are: " + _get_granular_level_pivot(data1, ner_dict1,
                                                                                                ner_filter, trend_check,
                                                                                                metric_df)
    elif intention != 'source of change' and driver_detail == 'tactics':
        if intention == 'contribution' and ('media' in biz_driver_check or 'promotions' in biz_driver_check or (
                biz_driver_check == 'all' and media_channel_check == 'all')) and not data1.empty:
            fixtext += "Business drivers contribution: " + _get_granular_level_pivot(data1, ner_dict1, ner_filter,
                                                                                     trend_check, metric_df)
    return fixtext


def _get_readout_br(intention, driver_detail, str_q):
    str_q = str_q.replace("Top performing tactics: ", "")
    if str_q:
        if intention == 'source of change':
            str_q = "Key drivers of sales change are: " + str_q
        elif intention != 'source of change' and driver_detail != 'tactics':
            str_q = "Top performing drivers: " + str_q
    return str_q


def _filter_media_promotion(intention, biz_driver_check, media_channel_check, data):
    if intention == 'contribution' or (
            intention == 'source of change' and biz_driver_check == 'all' and media_channel_check == 'all'):
        data = data[data['business_driver'].isin(['media', 'promotions'])]
    return data


def _filter_contribution_driver(intention, data):
    df = data.copy()
    if intention == 'contribution':
        df = df[df['business_driver'].isin(['media', 'marketing', 'promotions', 'non-media'])]
    trigger_pretext = True if len(df)<len(data) else False
    return df, trigger_pretext


def _calc_growth_br(intention, ner_res_dict, period_type, ner_dict, data):
    # if not (intention == 'source of change' and ner_res_dict.get('time', 'irrelevant') == 'irrelevant'):
    ner_dict_copy = ner_dict.copy()
    time = ner_res_dict.get('start', 'irrelevant')=='irrelevant' and ner_res_dict.get('end', 'irrelevant') =='irrelevant'
    if time and intention =='source of change' and 'end' in data.columns:
        data = data[data['end'] == data['end'].max()]
    if ner_dict['driver'] == 'business_driver_detail':
        ner_dict_copy['dimension'] = ner_dict['dimension'].copy()
        ner_dict_copy['dimension'].append('business_driver')
    data = _calc_growth(period_type, ner_dict_copy, data)
    return data


def _exclude_system(intention, data):
    if intention != 'source of change':
        data = data[data['business_driver'] != 'system']
    return data


def _get_pivot_biz_table(client_code, model_group_id,intention, biz_driver_check, media_channel_check, ner_dict1, ner_filter, rank_check,
                         sort_by_col, metrics_order, data1, metric_df):
    if biz_driver_check == 'all' and media_channel_check == 'irrelevant' and intention != 'sales' and not data1.empty:
        pivot_table_biz, _ = _get_granular_table(data1, ner_dict1, metrics_order, ner_filter, rank_check, sort_by_col,
                                                 metric_df, client_code, model_group_id)
    else:
        pivot_table_biz = pd.DataFrame()
    return pivot_table_biz


def _colgate_specific_soc(intention, ner_dict1, ner_filter, trend_check, data1, pivot_table_biz, str_q, fixtext,
                          metric_df):
    if not pivot_table_biz.empty and intention == 'source of change':
        str_q = "Key business drivers of sales change are: " + _get_granular_level_pivot(data1, ner_dict1, ner_filter,
                                                                                         trend_check, metric_df)
        fixtext = ""
    return str_q, fixtext


def _br_table_reformat(pivot_table, pivot_table_biz):
    if not pivot_table.empty:
        pivot_table = pivot_table.rename(
            columns={'Business_Driver_Detail': 'Business Driver', 'Tactics': 'Business Driver',
                     'Tactics_Detail': 'Business Driver', 'Driver': 'Business Driver'})
        if 'Business Driver' in pivot_table.columns:
            pivot_table['Business Driver'] = pivot_table['Business Driver'].astype(str)
    if not pivot_table_biz.empty:
        pivot_table_biz = pivot_table_biz.rename(
            columns={'Business_Driver_Detail': 'Business Driver', 'Tactics': 'Business Driver',
                     'Tactics_Detail': 'Business Driver', 'Driver': 'Business Driver'})
        if 'Business Driver' in pivot_table_biz.columns:
            pivot_table_biz['Business Driver'] = pivot_table_biz['Business Driver'].str.title()
    return pivot_table, pivot_table_biz


def _planner_format(planner_data):
    if not planner_data.empty:
        for col in planner_data.columns:
            col_lower = col.lower()
            if '%' in col_lower:
                planner_data[col] = planner_data[col].apply(
                    lambda x: f"{x:.2f}%" if x != float("inf") else f"0")
            elif 'spend' in col_lower and 'level' not in col_lower:
                planner_data[col] = planner_data[col].apply(lambda x: f"${int(x):,}")
            elif 'cost per acquisition' in col_lower or 'roi' in col_lower or 'margin' in col_lower:
                planner_data[col] = planner_data[col].apply(lambda x: f"${x:,.2f}")
            elif ('gross adds' in col_lower or 'activity' in col_lower) and not 'group' in col_lower:
                planner_data[col] = planner_data[col].apply(lambda x: f"{int(x):,}")
            # planner_data = planner_data.drop(columns=['Scenario ID', 'measure'], errors='ignore')
    planner_data.columns = planner_data.columns.str.replace('Incremental', 'Absolute Change', regex=True)
    return planner_data


def _get_top_planner_drivers(data, sort_ascending, top_n=5):
    if sort_ascending:
        filtered_data = data[data['Spend Incremental'] < 0]
    else:
        filtered_data = data[data['Spend Incremental'] > 0]
    sorted_data = filtered_data.sort_values(by=['Scenario ID', 'Spend Incremental'], ascending=[False, sort_ascending])
    top_drivers_df = sorted_data.groupby('Scenario ID').head(top_n)
    top_drivers_dict = top_drivers_df.groupby('Scenario ID')['Driver'].apply(list).to_dict()
    return top_drivers_df, top_drivers_dict


def _planner_curve(response_mod, curve_data, overview, scenario_ids):
    res = {}
    if "diminishing return" not in response_mod:
        for id in scenario_ids:
            res[id] = ""
        return res
    curve_df = curve_data.astype(float)
    avg_col = [col for col in curve_df.columns if col.lower().startswith('average')][0]
    marginal_col = [col for col in curve_df.columns if col.lower().startswith('marginal')][0]
    for scenario_id in scenario_ids:
        scenario_slice = curve_df[curve_df['Model ID'].astype(float) == scenario_id]
        scenario_slice = scenario_slice.astype(float)
        scenario_slice['diff'] = abs(scenario_slice[avg_col] - scenario_slice[marginal_col])

        min_index = scenario_slice['diff'].idxmin()
        argmin_spend = scenario_slice.loc[min_index, 'Spend']

        overview['Scenario ID'] = overview['Scenario ID'].astype(float)
        total_spend = overview[(overview['Scenario ID'] == scenario_id) &
                               (overview['KPI'] == 'Total Spend')]['Historical'].iloc[0].replace('$', '').replace(',',
                                                                                                                  '').strip()

        res[scenario_id] = (
            f"the current total spend of \${float(total_spend):,.0f} is {'below' if float(total_spend) < argmin_spend else 'above'} "
            f"the optimal level of spending of \${argmin_spend:,.0f} to optimally leverage diminishing returns.")

    return res


def _planner_open_statement(id, overview, curve_text):
    scenario_overview = overview[overview['Scenario ID'].astype(float) == id]
    period = scenario_overview['Forecast Period'].unique().tolist()
    description = scenario_overview['Scenario Description'].unique().tolist()
    scenario_fixtext = (
        f"From {period[0]}:\n"
        f"- When {description[0]}: {curve_text[id]}\n"
    )
    return scenario_fixtext, scenario_overview


def _format_inc(df_slice, spend_net, kpi_net, spend_change, kpi_change, target_kpi):
    return (f"    - {df_slice['Driver'].iloc[0]}: \t"
            + f"Spend absolute change of \${int(spend_net):,} "
            + (f"({spend_change:.1f}%)" if spend_change != float('inf') else "0")
            + f", resulting in {target_kpi} net change of {kpi_net:,.2f} units "
            + (f"({kpi_change:.1f}%)\n" if kpi_change != float('inf') else "0\n"))


def _format_dec(df_slice, spend_net, kpi_net, spend_change, kpi_change, target_kpi):
    return (f"    - {df_slice['Driver'].iloc[0]}: \t"
            + f"Spend decrease of {'-' if spend_net < 0 else ''}\${abs(int(spend_net)):,} "
              f"({spend_change:.1f}%), resulting in "
            + f"{target_kpi} net change of {kpi_net:,.2f} units ({kpi_change:.1f}%)\n")


def _driver_format_str(scenario_fixtext, drivers, driver_df, target_kpi, text_top_n, direction):
    if drivers:
        for driver in drivers[:text_top_n]:
            df_slice = driver_df[driver_df['Driver'] == driver]
            spend_net, kpi_net = (float(df_slice['Spend Incremental'].iloc[0]),
                                  float(df_slice[f'{target_kpi} Incremental'].iloc[0]))
            spend_change, kpi_change = (float(df_slice['Spend % Change'].iloc[0]),
                                        float(df_slice[f'{target_kpi} % Change'].iloc[0]))
            if direction == 'increase':
                format_str = _format_inc(df_slice, spend_net, kpi_net, spend_change, kpi_change, target_kpi)
            else:
                format_str = _format_dec(df_slice, spend_net, kpi_net, spend_change, kpi_change, target_kpi)
            scenario_fixtext += format_str
    return scenario_fixtext


def _not_core_dim_fixtext(scenario_ids, overview, curve_text, inc_driver_dict, dec_driver_dict, text_top_n):
    fixtext_dict = {}
    for id in scenario_ids:
        scenario_fixtext, scenario_overview = _planner_open_statement(id, overview, curve_text)
        scenario_overview['Growth %'] = scenario_overview['Growth %'].str.replace('%', '').astype(float)
        scenario_overview_filtered = scenario_overview[scenario_overview['KPI'] != 'Total Spend']
        metric_col = 'Metrics' if 'Metrics' in scenario_overview_filtered.columns else 'KPI'
        statements = [
            f"{row[metric_col]} {'increased' if row['Growth %'] > 0 else 'decreased'} by {abs(row['Growth %'])}%."
            for _, row in scenario_overview_filtered.iterrows()
        ]
        kpi_statements = " ".join(statements)
        scenario_fixtext += f"  - {kpi_statements}\n"

        scenario_fixtext += f"  - Top drivers with the most spend increase: {', '.join(inc_driver_dict[id][:text_top_n])}.\n" if id in inc_driver_dict else ""
        scenario_fixtext += f"  - Top drivers with the most spend decrease: {', '.join(dec_driver_dict[id][:text_top_n])}.\n" if id in dec_driver_dict else ""
        fixtext_dict[f'fixtext_{id}'] = scenario_fixtext
    fixtext = "\n\n".join(fixtext_dict[f'fixtext_{id}'] for id in scenario_ids)
    return fixtext


def _core_dim_fixtext(scenario_ids, overview, curve_text, inc_driver_dict, inc_driver_df, dec_driver_dict,
                      dec_driver_df, target_kpi, text_top_n):
    fixtext_dict = {}
    for id in scenario_ids:
        scenario_fixtext, _ = _planner_open_statement(id, overview, curve_text)
        if id in inc_driver_dict:
            scenario_fixtext += "  - The top drivers with the most spend increase are:\n" if id in inc_driver_dict else ""
            scenario_fixtext = _driver_format_str(scenario_fixtext, inc_driver_dict[id],
                                                  inc_driver_df[inc_driver_df['Scenario ID'] == id], target_kpi,
                                                  text_top_n,
                                                  direction='increase')
        if id in dec_driver_dict:
            scenario_fixtext += "  - The top drivers with the most spend decrease are:\n" if id in dec_driver_dict else ""
            scenario_fixtext = _driver_format_str(scenario_fixtext, dec_driver_dict[id],
                                                  dec_driver_df[dec_driver_df['Scenario ID'] == id], target_kpi,
                                                  text_top_n,
                                                  direction='decrease')
        fixtext_dict[f'fixtext_{id}'] = scenario_fixtext
    fixtext = "\n".join(fixtext_dict[f'fixtext_{id}'] for id in scenario_ids)
    return fixtext


def response_generate(query: str, readout: str) -> str:
    prompt_template = PromptTemplates()
    readout = convert_display_format(readout)
    prompt = prompt_template.build_readout_prompt(query, readout)
    text = generate_text(prompt)
    return text


def _planner_principle_summary(spend_by_channel, genome_principle):
    summaries = []
    multiple_scenarios = spend_by_channel['Scenario Description'].nunique() > 1
    for desc, group in spend_by_channel.groupby('Scenario Description'):
        scenario_summaries = []
        fore_or_opt_col = 'optimized_spend_share' if 'optimized_spend_share' in group.columns else 'forecast_spend_share'
        for _, row in group.iterrows():
            channel = row['marketingChannel']
            hist_share = row['historical_spend_share']
            fore_share = row[fore_or_opt_col]
            median = row['medianPoint']
            left = row['maintainSpendShareRangeLeft']
            right = row['maintainSpendShareRangeRight']
            # Historical interpretation
            # hist_comp = "higher than" if hist_share > median else "lower than" if hist_share < median else "equal to"
            hist_range = "within" if left <= hist_share <= right else "out of"
            # Forecast interpretation
            # fore_comp = "higher than" if fore_share > median else "lower than" if fore_share < median else "equal to"
            fore_range = "within" if left <= fore_share <= right else "out of"

            # summary = (
            #     f"- **{channel.title()}** historical spend share is {hist_share:.2f}%, which is {hist_comp} the median point "
            #     f"({median:.2f}%), and {hist_range} the maintain spend share range ({left:.2f}%–{right:.2f}%). "
            #     f"The {fore_or_opt_col.replace('_', ' ')} is {fore_share:.2f}%, which is {fore_comp} the media point, "
            #     f"and {fore_range} the maintain spend share range ({left:.2f}%–{right:.2f}%)."
            # )
            if genome_principle.empty:
                genome_str = ''
            else:
                matched = genome_principle[
                    (genome_principle['matchingIntention'] == 'planner') &
                    (genome_principle['marketingChannel'] == channel)]
                genome_str = matched['genomePrinciple'].iloc[0] if not matched.empty else ''
            summary = (
                f"- **Total {channel.title()}** historical spend share is {hist_share:.1f}%, "
                f"which is {hist_range} the maintain spend share range ({left:.1f}%–{right:.1f}%). "
                f"The {fore_or_opt_col.replace('_', ' ')} is {fore_share:.1f}%,  "
                f"and {fore_range} the spend share range ({left:.1f}%–{right:.1f}%). "
                f"{genome_str}")
            scenario_summaries.append(summary)

        if multiple_scenarios:
            # Add scenario header only if there are multiple
            scenario_block = f"**Scenario: {desc}**\n\n" + "\n".join(scenario_summaries)
        else:
            scenario_block = "\n".join(scenario_summaries)

        summaries.append(scenario_block)
    return "\n\n".join(summaries)


def _planner_spend_range(planner_data, planner_principle, genome_principle, overview_all):
    summaries = []
    if not planner_principle.empty and not overview_all.empty and 'marketing_channel' in planner_data.columns:
        scenario_ids = planner_data["Scenario ID"].unique().tolist()
        multiple_scenarios = len(scenario_ids) > 1
        spend_by_channel_list = []
        for id in scenario_ids:
            planner_df = planner_data[planner_data['Scenario ID']==id]
            overview = overview_all[overview_all['Scenario ID']==id]
            total_spend_his = float(
                overview.loc[overview['KPI'] == 'Total Spend', 'Historical'].values[0].replace('$', '').replace(',', ''))
            total_spend_fore = float(
                overview.loc[overview['KPI'] == 'Total Spend', 'Forecast'].values[0].replace('$', '').replace(',', ''))
            # drop_cols = ['KPI', 'activity_group', 'measure_group', 'measure'] if 'KPI' in planner_df.columns else ['activity_group', 'measure_group', 'measure']
            # df = planner_df.drop(columns=drop_cols, errors='ignore').drop_duplicates()
            df = planner_df.copy()
            if not 'Spend - Optimized' in df.columns:
                calc_col, new_col = 'Spend - Forecast', 'forecast_spend_share'
            else:
                calc_col, new_col = 'Spend - Optimized', 'optimized_spend_share'
            df = df[['Scenario ID', 'Scenario Description', 'marketing_channel','Spend - Historical', calc_col]].drop_duplicates()
            spend_by_channel = df.groupby(['Scenario ID', 'Scenario Description', 'marketing_channel'])[
                ['Spend - Historical', calc_col]].sum().reset_index()
            spend_by_channel['historical_spend_share'] = 100 * spend_by_channel['Spend - Historical'] / total_spend_his
            spend_by_channel[new_col] = 100 * spend_by_channel[calc_col] / total_spend_fore
            spend_by_channel = spend_by_channel.rename(columns={'marketing_channel': 'marketingChannel'})
            spend_by_channel = pd.merge(spend_by_channel, planner_principle, how='inner', on='marketingChannel')
            if spend_by_channel.empty:
                continue
            spend_by_channel_list.append(spend_by_channel)
            summary = _planner_principle_summary(spend_by_channel, genome_principle)
            # ---- scenario label (ONLY for multi-scenario) ----
            if multiple_scenarios:
                scenario_name = planner_df["Scenario Description"].iloc[0]
                summaries.append(f"**When {scenario_name}**\n{summary}")
            else:
                summaries.append(summary)
        spend_by_channel_all = pd.concat(spend_by_channel_list, ignore_index=True)
        if not summaries:
            return "", pd.DataFrame()
        return "**Additional Insights from ROI Genome.**\n\n" + "\n\n".join(summaries), spend_by_channel_all
    else:
        return "", pd.DataFrame()


def _compute_reduce_budget(data, overview_df, core_dim):
    if core_dim:
        return data["Spend - Forecast"].sum() < data["Spend - Historical"].sum()

    if overview_df.empty:
        return False

    row0 = overview_df.iloc[0]
    hist = float(str(row0["Historical"]).replace("$", "").replace(",", ""))
    fcst = float(str(row0["Forecast"]).replace("$", "").replace(",", ""))
    return fcst < hist


def _build_planner_fixtext_and_table(data, curve_df, overview_df, scenario_ids, response_mod, core_dim, text_top_n, target_kpi=None):
    reduce_budget = _compute_reduce_budget(data, overview_df, core_dim)
    data = data.copy()
    data.loc[:, data.columns[:-1]] = data.loc[:, data.columns[:-1]].fillna(np.inf)

    inc_df, inc_dict = _get_top_planner_drivers(data, False)
    dec_df, dec_dict = _get_top_planner_drivers(data, True)

    curve_text = _planner_curve(response_mod, curve_df, overview_df, scenario_ids)

    if not core_dim:
        fixtext = _not_core_dim_fixtext(scenario_ids, overview_df, curve_text, inc_dict, dec_dict, text_top_n)
    else:
        fixtext = _core_dim_fixtext(scenario_ids, overview_df, curve_text, inc_dict, inc_df, dec_dict, dec_df, target_kpi, text_top_n)

    inc_df = _planner_format(inc_df)
    dec_df = _planner_format(dec_df)

    if not inc_df.empty:
        inc_df["Increase/Decrease"] = "Increase"
    if not dec_df.empty:
        dec_df["Increase/Decrease"] = "Decrease"

    planner_table = pd.concat([inc_df, dec_df], ignore_index=True)

    return fixtext, planner_table, reduce_budget


def _planner_output(planner_df, m_df, metric_df, planner_principle, genome_principle, core_dim, response_mod,
                    planner_response_curve, overview,
                    planner_filter,
                    text_top_n=5):
    principle_summary, spend_by_channel = _planner_spend_range(planner_df, planner_principle, genome_principle, overview)
    reduce_budget = False
    call_llm = False
    planner_data = planner_df.copy()
    # if m_df.empty:
    # planner_data = pd.DataFrame()
    if response_mod and core_dim:
        heads = ["This tool does not have access to the data source required to generate insights in this version. Please use the GPSE Planner application.",
                 "This version of the tool does not have access to the required data sources to generate insights. Please use the GPSE Planner application.",
                 "Insight generation is unavailable in this version due to data access limitations. Please use the GPSE Planner application instead.",
                 "This version currently does not support insight generation due to data access constraints. Please use the GPSE Planner application."]
        return '', pd.DataFrame(), reduce_budget, call_llm, np.random.choice(heads)
    if not m_df.empty and core_dim:
        # m_ls = m_df['measure'].unique().tolist()
        # planner_data = planner_df[planner_df['measure'].isin(m_ls)]
        join_cols = (["measure"] if not m_df["measure"].isna().any()
            else ["activity_group", "measure_group", "measure"])
        planner_data = planner_data.merge(m_df[join_cols].drop_duplicates(), how="inner", on=join_cols)
        principle_summary,_ = _planner_spend_range(planner_data, planner_principle, genome_principle, overview)
    planner_data = (planner_data
        .drop(columns=["activity_group", "measure_group", "measure", "marketing_channel"], errors="ignore")
        .dropna(axis=1, how="all").drop_duplicates())

    if planner_data.empty:
        return "", pd.DataFrame(), reduce_budget, call_llm
    uniq_kpi = planner_filter['KPI'].unique().tolist() if 'KPI' in planner_filter else []
    uniq_scenarios = planner_filter['Scenario ID'].unique().tolist() if 'Scenario ID' in planner_filter else []
    if len(uniq_kpi) == 0:
        target_kpi = planner_filter['target_kpi'].iloc[0]
        scenario_ids = planner_data['Scenario ID'].unique().astype(float).tolist()
        fixtext, planner_table, reduce_budget = _build_planner_fixtext_and_table(planner_data,
            planner_response_curve, overview, scenario_ids, response_mod, core_dim, text_top_n, target_kpi)

        planner_table = _dynamic_top(["Scenario ID"],"Spend % Change",planner_table,metric_df)
    else:
        fixtext_parts = []
        planner_table_all = []
        reduce_budget_flags = []
        for kpi in uniq_kpi:
            target_kpi = planner_filter.loc[planner_filter["KPI"] == kpi, "target_kpi"].iloc[0]

            data_sub = planner_data[planner_data["KPI"] == kpi]
            curve_sub = planner_response_curve[planner_response_curve["KPI"] == kpi]
            overview_sub = overview[overview["KPI"] == kpi]
            scenario_ids = data_sub["Scenario ID"].astype(float).unique().tolist()

            fixtext_sub, table_sub, reduce_budget_sub = _build_planner_fixtext_and_table(data_sub, curve_sub,
                overview_sub, scenario_ids, response_mod, core_dim, text_top_n, target_kpi)

            fixtext_parts.append(f"For {kpi}:\n\n{fixtext_sub}")
            planner_table_all.append(table_sub.assign(KPI=kpi))
            reduce_budget_flags.append(reduce_budget_sub)

        fixtext = "\n\n".join(fixtext_parts)
        planner_table = pd.concat(planner_table_all, ignore_index=True)

        planner_table = _dynamic_top(["Scenario ID", "KPI"],"Spend % Change", planner_table,
            metric_df, total_entry=0)
        reduce_budget = any(reduce_budget_flags)
    if not planner_table.empty:
        planner_table = planner_table.drop(columns=["Scenario ID"], errors="ignore")
        planner_table = (planner_table.replace(r"^\$?-?inf$", np.nan, regex=True)
            .replace([np.inf, -np.inf], np.nan))
        pct_cols = [c for c in planner_table.columns if "% Change" in c]
        planner_table[pct_cols] = (planner_table[pct_cols].replace(["nan%", "0%", 0, "0"], np.nan))
    # remove principle summary from pretext
    # if fixtext:
    #     fixtext = f"{fixtext}\n\n{principle_summary}"
    call_llm = bool(fixtext and len(uniq_kpi) > 1 and len(uniq_scenarios) > 1)
    heads = ["The insights presented in this tool are derived from published scenarios and are intended for reference only. For Re-optimization or scenario development, please use the GPSE Planner application",
             "All insights in this version are derived from published scenarios and should be used as reference. For scenario planning or re-optimization activities, please access the GPSE Planner application.",
             "The analysis presented here reflects published scenarios only. For scenario development or optimization adjustments, please proceed to the GPSE Planner application.",
             "This tool provides reference insights based on published scenarios. To create or refine scenarios, please use the GPSE Planner application."]
    return fixtext, planner_table, reduce_budget, call_llm, np.random.choice(heads), spend_by_channel


def _categorize_columns(df):
    dimension_cols = []
    driver_cols = []
    metric_cols = []

    columns = list(df.columns)

    # Identify metric columns (contains 4-digit year in the name)
    metric_pattern = re.compile(r'\b\d{4}\b')
    metric_cols = [col for col in columns if metric_pattern.search(col)]

    # Identify the dimension and driver columns
    for i, col in enumerate(columns):
        if col in metric_cols:
            continue  # Skip metric columns

        if i < len(columns) - 1 and columns[i + 1] in metric_cols:
            driver_cols.append(col)
        elif not metric_cols:
            driver_cols.append(col)
        else:
            dimension_cols.append(col)

    return dimension_cols, driver_cols, metric_cols


def _output_table_to_text(df):
    if any('Source Of Change' in col for col in df.columns):
        cols_drop = [col for col in df.columns if any(key in col for key in ["YOY", "QOQ", "HOH"])]
        if cols_drop:
            df = df.drop(columns=cols_drop)
    dimension_cols, driver_cols, metric_cols = _categorize_columns(df)
    text_output = []

    # Define patterns to extract metric names
    metric_pattern = re.compile(r'^(.*?)(?=\sQ\d|\sMonth|\s\d{4})')  # Extract text before 'Q', 'Month', or year
    mm_pattern = re.compile(r'\(MM\)$')  # Identify (MM) columns

    # Group the dataframe by dimension columns (if they exist)
    if dimension_cols:
        grouped_df = df.groupby(dimension_cols)
    else:
        grouped_df = [(None, df)]  # If no dimensions, treat all rows as one group

    for dimension_values, group in grouped_df:
        row_text = []

        # Format dimension text (only displayed once per unique combination)
        if dimension_cols:
            dimension_text = " | ".join(f"{dim}: {value}" for dim, value in zip(dimension_cols, dimension_values))
            row_text.append(dimension_text)

        # Collect drivers together under the same dimension
        driver_metrics = {}

        for _, row in group.iterrows():
            driver_text = " | ".join(f"{driver}: {row[driver]}" for driver in driver_cols if pd.notna(row[driver]))

            if driver_text not in driver_metrics:
                driver_metrics[driver_text] = {}

            for metric in metric_cols:
                match = metric_pattern.match(metric)
                if match and pd.notna(row[metric]):  # Only include non-missing values
                    base_metric = match.group(1).strip()
                    is_mm = bool(mm_pattern.search(metric))  # Check if it's an MM column

                    if is_mm:
                        base_metric += " (MM)"  # Keep (MM) in the metric name, not in the Qx year part

                    if base_metric not in driver_metrics[driver_text]:
                        driver_metrics[driver_text][base_metric] = []

                    # Extract Qx and year part, ensuring MM columns keep Qx labels
                    metric_parts = metric.split()
                    if ("QOQ" in metric):
                        q_part = " ".join(metric_parts[-3:])  # Q4 2024 QOQ
                    elif ('YOY' in metric):
                        q_part = " ".join(metric_parts[-2:])  # 2024 YOY
                    else:
                        q_number = next((part for part in metric_parts if "Q" in part), "")
                        year = next((part for part in metric_parts if part.isdigit() and len(part) == 4), "")
                        if q_number and year:
                            q_part = f"{q_number} {year}"
                        else:
                            # Detect Month x yyyy or Month x yyyy - Month y yyyy
                            month_range_match = re.search(
                                r"Month\s(\d{1,2})\s(\d{4})(?:\s-\sMonth\s(\d{1,2})\s(\d{4}))?", metric
                            )
                            if month_range_match:
                                m1 = int(month_range_match.group(1))
                                y1 = month_range_match.group(2)
                                m2 = month_range_match.group(3)
                                y2 = month_range_match.group(4)

                                # Optionally convert to month names:
                                q_part = f"Month {m1} {y1}"
                                if m2 and y2:
                                    q_part += f" - Month {m2} {y2}"
                            else:
                                if is_mm:
                                    q_part = " ".join(metric_parts[-2:])
                                else:
                                    q_part = " ".join(metric_parts[-1:])

                    driver_metrics[driver_text][base_metric].append(f"{q_part}: {row[metric]}")

        # Convert grouped drivers and metrics to text
        for driver_text, metrics in driver_metrics.items():
            row_text.append(driver_text)
            for metric_name, values in metrics.items():
                row_text.append(f"{metric_name}: {', '.join(values)}")
            row_text.append("")  # Add an extra newline after each driver block

        # Append row text if it contains data
        if row_text:
            text_output.append("\n".join(row_text))

    return "\n\n".join(text_output)


def _soc_split_table(table):
    if table.empty:
        return table.copy(), table.copy()
    tactics_col = 'Business Driver'
    soc_cols = [col for col in table.columns if 'source of change' in col.lower()]
    df = _clean_numeric_after_tactics(table, soc_cols, tactics_col)
    if not soc_cols:
        return table.copy(), table.copy()
    metric_col = soc_cols[0]
    pos_driver = df[df[metric_col] >= 0][tactics_col].values.tolist()
    pos_df = table[table[tactics_col].isin(pos_driver)]
    neg_df = table[~table[tactics_col].isin(pos_driver)]
    return pos_df, neg_df


def table_to_text(agg_table, detail_table, take_head=True, buffer=800):
    max_tokens = int(os.getenv("LLM_MAX_INPUT_TOKENS")) - buffer

    def process_table(table, take_head=take_head):
        if table.empty:
            return "No data available. ", [], table
        if take_head:
            head = table.head(20)
        else:
            head = table.copy()
        table_str = _output_table_to_text(head)
        # TODO: remove truncation
        tokens = tokenize_text(table_str)
        return table_str, tokens, head

    if any('Source Of Change' in col for col in agg_table.columns):
        pos_agg_df, neg_agg_df = _soc_split_table(agg_table)
        pos_agg_str, pos_agg_tokens, _ = process_table(pos_agg_df, take_head)
        neg_agg_str, neg_agg_tokens, _ = process_table(neg_agg_df, take_head)
        agg_str = 'Positive SOC drivers:\n' + pos_agg_str + '\nNegative SOC drivers:\n' + neg_agg_str
        agg_tokens = pos_agg_tokens + neg_agg_tokens
        _, _, agg_head = process_table(agg_table, take_head)
    else:
        agg_str, agg_tokens, agg_head = process_table(agg_table, take_head)

    if any('Source Of Change' in col for col in detail_table.columns):
        pos_detail_df, neg_detail_df = _soc_split_table(detail_table)
        pos_detail_str, pos_detail_tokens, _ = process_table(pos_detail_df, take_head)
        neg_detail_str, neg_detail_tokens, _ = process_table(neg_detail_df, take_head)
        detail_str = 'Positive SOC drivers:\n' + pos_detail_str + '\nNegative SOC drivers:\n' + neg_detail_str
        detail_tokens = pos_detail_tokens + neg_detail_tokens
        _, _, detail_head = process_table(detail_table, take_head)
    else:
        detail_str, detail_tokens, detail_head = process_table(detail_table, take_head)

    total_tokens = agg_tokens + detail_tokens
    if len(total_tokens) <= max_tokens:
        return agg_str, detail_str, agg_head, detail_head

    if max_tokens < 1:
        raise RuntimeError(f"Text cannot be truncated. LLM max input too small: {os.getenv('LLM_MAX_INPUT_TOKENS')}")

    agg_limit = int(max_tokens * len(agg_tokens) / max(1, len(total_tokens)))
    detail_limit = max_tokens - agg_limit

    def truncate(text, tokens, limit, head):
        if not tokens:
            return text, head
        truncated_tokens = tokens[:limit]
        truncated_text = text[:truncated_tokens[-1]["stop"]]
        row_count = max(1, int(len(truncated_tokens) / (len(tokens) / len(head))))
        return truncated_text, head#.head(row_count)

    agg_str, agg_head = truncate(agg_str, agg_tokens, agg_limit, agg_head)
    detail_str, detail_head = truncate(detail_str, detail_tokens, detail_limit, detail_head)

    logger.warning(f"Prompt was truncated from {len(total_tokens)} tokens to {max_tokens} tokens.")

    return agg_str, detail_str, agg_head, detail_head


def _get_level_tagging(ner_filter, client_code=''):
    general_level, halo_level = ner_filter.get('level', {}).get('general_level', []), ner_filter.get('level', {}).get(
        'halo_level', [])
    general_tag, halo_tag = ner_filter.get('level', {}).get('general_tagging', []), ner_filter.get('level', {}).get(
        'halo_tagging', [])
    if client_code == 'LINKEDIN':
        level = general_level + halo_level + halo_tag
    else:
        level = general_level + halo_level
    tagging = general_tag + halo_tag
    return level, tagging

def _custom_tagging_condition(ner_filter, tagging):
    if tagging:
        for t in tagging:
            if ner_filter.get(t, []):
                return False
    return True

def _level_to_drop(level, dim_cols, data):
    if len(level) > 0:
        col_drop = [d for d in dim_cols if d not in level]
        data = data.drop(columns = col_drop)
        dim_cols = [d for d in dim_cols if d in level]
    return data, dim_cols

def _halo_tagging_filter(data, ner_filter, core_dim_composite):
    halo_tag = ner_filter.get('level', {}).get('halo_tagging', [])
    if (data['custom_aggregated']=='no').all():
        return data
    if halo_tag:
        for t in halo_tag:
            tag_hit = ner_filter.get(t, [])
            if tag_hit and not data.empty and not core_dim_composite:
                data = data[(data[t].notna()) & (~data['custom_aggregated'].isin(['no', 'diagnostic']))]
    return data


# def _planner_filter(planner_filter, dim_cols, m_df, planner_df):
#     if dim_cols:
#         filter_dim = [d for d in dim_cols if d in planner_filter.columns]
#         if filter_dim:
#             dim_id_map = dict(zip(planner_filter[filter_dim[0]], planner_filter['Scenario ID']))
#             dims = m_df[filter_dim[0]].unique().tolist()
#             mapped_id = [dim_id_map[d] for d in dims if d in dim_id_map]
#             if mapped_id:
#                 planner_df = planner_df[planner_df['Scenario ID'].isin(mapped_id)]
#     return planner_df

def _planner_filter(planner_filter, dim_cols, ner_filter, planner_df):
    level_type = ner_filter.get("level", {})

    if dim_cols:
        filter_dim = [d for d in dim_cols if d in planner_filter.columns]
        if filter_dim:
            dim_values = {}
            for d in filter_dim:
                filter_val = ner_filter.get(d, [])
                # dim_values[d] = ner_filter.get(d)

                # if d in level_type.get("halo_level", []):
                #     halo_tagging = level_type.get("halo_tagging", [])
                #     if halo_tagging:
                #         filter_val += ner_filter.get(halo_tagging[0], [])
                if filter_val:
                    dim_values[d] = filter_val
            # prioritize non-overall dimensions
            filter_dim = sorted(filter_dim, key=lambda d: dim_values.get(d, ["overall"]) == ["overall"])
            value_id_map = planner_filter.groupby(filter_dim)["Scenario ID"].apply(list).to_dict()
            # start with all keys as candidates
            candidate_items = list(value_id_map.items())

            for dim_idx, dim in enumerate(filter_dim):
                allowed_values = dim_values.get(dim, ["overall"])

                # values available in CURRENT candidate set
                available_values = {k[dim_idx] for k, _ in candidate_items}

                # case: only "overall" requested
                if allowed_values == ["overall"]:
                    if "overall" in available_values:
                        # enforce overall ONLY within current candidates
                        candidate_items = [
                            (k, ids) for k, ids in candidate_items
                            if k[dim_idx] == "overall"
                        ]
                    else:
                        # skip this dimension entirely
                        continue
                else:
                    # normal filtering
                    candidate_items = [
                        (k, ids) for k, ids in candidate_items
                        if k[dim_idx] in allowed_values
                    ]

                if not candidate_items:
                    break

            matched_ids = set()
            for _, ids in candidate_items:
                matched_ids.update(ids)
            mapped_id = list(matched_ids)
            if mapped_id:
                planner_df = planner_df[planner_df['Scenario ID'].isin(mapped_id)]
    return planner_df

def _generate_driver_tags(tactics_level, df_dict):
    driver_tags = {}
    driver_tags_br = {}
    record_level_mapping = {
        'measure_group': 'tactics_detail',
        'measure': 'detail_tactics',
        'activity_group': 'tactics'}

    for _, row in tactics_level.iterrows():
        key = row['file_type']
        value = (row['file_type'], row['dataType'], row['tactics_level'], row['tactics_level'])

        if row['dataType'] == 'bi':
            driver_tags[key] = value
        elif row['dataType'] == 'br':
            driver_tags_br[key] = value
    for key in ['ag', 'mg', 'm']:
        df = df_dict.get(f'{key}_df')
        if df is not None and not df.empty:
            record_level = df['record_level'].unique()[0]
            if record_level != driver_tags_br[key][2]:
                mapped_level = record_level_mapping[record_level] if record_level in record_level_mapping else record_level
                driver_tags_br[key] = (key, 'br', mapped_level, mapped_level)
    return driver_tags, driver_tags_br

def _br_custom_agg_filter(core_dimension, data, ner_filter):
    halo_levels = ner_filter.get('level', {}).get('halo_level', [])
    halo_level = halo_levels[0] if halo_levels else None
    halo_filter = ner_filter.get(halo_level, []) if halo_level else []
    if not (halo_filter or (core_dimension and not halo_filter)) and 'custom_aggregated' in data.columns:
        data = data[data['custom_aggregated'] != 'yes']
    return data


def _table_readout_prompt(readout, agg_table_str, detail_table_str):
    no_data = 'No data available. '
    if agg_table_str == no_data and detail_table_str == no_data:
        return readout
    return readout + (
        f"\nFirst, analyze this high level aggregated market data: \n{agg_table_str}\n"
        f"Then, analyze this detailed market data: \n{detail_table_str}\n"
        "Ensure your answer considers both data levels and synthesizes insights clearly.")


def _replace_month(match):
    month_num = int(match.group(1))
    return month_name[month_num]

def _table_month_transform(df):
    new_columns = [re.sub(r'Month (\d{1,2})', _replace_month, col) for col in df.columns]
    df.columns = new_columns
    df = df.dropna(axis=1, how='all')
    df = df.loc[:, ~(df.apply(lambda col: col.astype(str).str.lower().eq('overall').all()))]
    return df


def _percent_change_summary(intention, target_col, tactics_col, df):
    tactics_idx = df.columns.get_loc(tactics_col)
    left_cols = df.columns[:tactics_idx].tolist()
    final_cols = left_cols + [tactics_col, target_col]
    df_filtered = df[final_cols].dropna(subset=[target_col])

    if df_filtered.empty or intention == 'sales':
        return ""
    df_filtered[target_col] = df_filtered[target_col].replace(r'%', '', regex=True).astype(float)
    summary_parts = []
    if left_cols:
        grouped = df_filtered.groupby(left_cols)
    else:
        grouped = [(None, df_filtered)]
    for group_key, group_df in grouped:
        try:
            max_row = group_df.loc[group_df[target_col].idxmax()]
            min_row = group_df.loc[group_df[target_col].idxmin()]
        except ValueError:
            continue  # skip empty group

        if isinstance(group_key, tuple):
            group_label = ", ".join(str(g) for g in group_key)
        else:
            group_label = ''

        if intention == 'source of change':
            if pd.notna(max_row[target_col]) and max_row[target_col] > 0:
                summary_parts.append(
                    f"{group_label} Top positive {target_col}: {max_row[tactics_col]} ({max_row[target_col]:.2f})."
                )
            if pd.notna(min_row[target_col]) and min_row[target_col] < 0:
                summary_parts.append(
                    f"{group_label} Top negative {target_col}: {min_row[tactics_col]} ({min_row[target_col]:.2f})."
                )
        else:
            if pd.notna(max_row[target_col]) and max_row[target_col] > 0:
                summary_parts.append(
                    f"{group_label} Driver with highest {intention} increase in {target_col}: {max_row[tactics_col]} ({max_row[target_col]:.2f})."
                )
            if pd.notna(min_row[target_col]) and min_row[target_col] < 0:
                summary_parts.append(
                    f"{group_label} Driver with highest {intention} decrease in {target_col}: {min_row[tactics_col]} ({min_row[target_col]:.2f})."
                )
    return " ".join(summary_parts)


def _br_custom_instruction(intention, time, df, ner_filter_dict):
    inst_parts = []
    outlier = ""
    if intention in ['contribution']:
        if len(time) >= 3:
            trend_data, _ = trend_contribution_sales(df, ner_filter_dict)
            if trend_data:
                inst_parts.append(
                    f"Trend analysis, break marketing drivers into 5 groups: "
                    f"1) Consistently increasing at generally high and moderate {intention} level; "
                    f"2) Consistently decreasing at high and moderate {intention} level; "
                    f"3) Consistently low {intention} level; "
                    f"4) Flat at high and moderate {intention} level; "
                    f"5) Volatile."
                )
                inst_parts.append(trend_data)
        target_col = next((c for c in df.columns if 'YOY' in c or 'QOQ' in c or 'HOH' in c), None)
        if target_col:
            summary = _percent_change_summary(intention, target_col, 'Business Driver', df)
            if summary:
                inst_parts.append(summary)
        outlier = _get_outlier_pretext(df, ner_filter_dict, tactics_col='Business Driver',
                                       percent_change_col=target_col)
    if intention == 'source of change':
        # if len(time) >= 3:
        #     trend_data = trend_soc(df)
        #     if trend_data:
        #         inst_parts.append(
        #             "Trend analysis, break drivers into 6 groups: "
        #             "1) Consistently positive and high impact; "
        #             "2) Consistently negative and high impact; "
        #             "3) Consistently positive; "
        #             "4) Consistently negative; "
        #             "5) Highly Volatile; "
        #             "6) Consistently low impact."
        #         )
        #         inst_parts.append(trend_data)
        target_col = next((c for c in df.columns if 'Source Of Change' in c), None)
        summary = _percent_change_summary(intention, target_col, 'Business Driver', df)
        if summary:
            inst_parts.append(summary)
    return "\n" + "\n".join(inst_parts) if inst_parts else "", outlier


def _bi_custom_instruction(intention, time, df, client_code, model_group_id, ner_filter_dict, prompt_inst):
    inst_parts = []
    outlier = ''
    if intention == 'spending':
        if len(time) >= 3:
            trend_data, _ = trend_bi(df, ner_filter_dict)
            if trend_data:
                inst_parts.append(
                    f"Trend analysis, break drivers into 5 groups: "
                    f"1) Consistently increasing at generally high and moderate {intention} level; "
                    f"2) Consistently decreasing at high and moderate {intention} level; "
                    f"3) Consistently low {intention} level; "
                    f"4) Flat at high and moderate {intention} level; "
                    f"5) Volatile."
                )
                inst_parts.append(trend_data)
        target_col = next((c for c in df.columns if 'YOY' in c or 'QOQ' in c or 'HOH' in c), None)
        if target_col:
            df_filtered = df[['Tactics', target_col]].dropna()
            summary = _percent_change_summary(intention, target_col, 'Tactics', df_filtered)
            if summary:
                inst_parts.append(summary)
        outlier = _get_outlier_pretext(df, ner_filter_dict, percent_change_col=target_col)
    if intention in ['margin roi', 'performance']:
        recent_data, _, recent_data_df = _recent_roi_spend(df, client_code, model_group_id, ner_filter_dict, roi_type = ner_filter_dict['main_metric'][0])
        recent_inst = prompt_inst[prompt_inst['intentionName']=='ROI-Spend latest']['instruction'].values[0]
        if recent_data:
            inst_parts.append(recent_inst)
            inst_parts.append(recent_data)
        target_col = next((c for c in df.columns if 'YOY' in c or 'QOQ' in c or 'HOH' in c), None)
        outlier = _get_outlier_pretext(df, ner_filter_dict, percent_change_col=target_col)
        if target_col:
            df_filtered = df[['Tactics', target_col]].dropna()
            summary = _percent_change_summary(intention, target_col, 'Tactics', df_filtered)
            if summary:
                inst_parts.append(summary)
            if not (ner_filter_dict.get('trend', 'no') == 'yes' and len(time) > 2):
                # roi & spend -> increase / decrease
                trend_data, _ = _trend_roi_spend(df, ner_filter_dict, roi_type = ner_filter_dict['main_metric'][0])
                trend_inst = prompt_inst[prompt_inst['intentionName'] == 'ROI-Spend trend test']['instruction'].values[0]
                if trend_data:
                    inst_parts.append(trend_inst)
                    inst_parts.append(trend_data)
            if not (ner_filter_dict.get('trend', 'no')=='yes' and len(time)>=2):
                other_data, _ = _trend_roi_other(df, ner_filter_dict, roi_type = ner_filter_dict['main_metric'][0])
                other_inst = prompt_inst[prompt_inst['intentionName'] == 'ROI-Response-Cost Per Trend']['instruction'].values[0]
                if other_data:
                    inst_parts.append(other_inst)
                    inst_parts.append(other_data)
        if len(time) >= 3:
            trend_data, _ = trend_bi(df, ner_filter_dict, roi_level_df = recent_data_df)
            if trend_data:
                inst_parts.append(
                    f"Summarize the data into the following trend patterns: "
                    f"1) Consistently increasing at excellent level; "
                    f"2) Consistently Increasing at good and justifiable level; "
                    f"3) Consistently decreasing at excellent Level; "
                    f"4) Consistently decreasing at justifiable level; "
                    f"5) Flat; "
                    f"5) Volatile.")
                inst_parts.append(trend_data)
    return "\n" + "\n".join(inst_parts) if inst_parts else "", outlier


# def _bi_overall_custom_instruction(intention, time, overall_table):
#     inst_parts = []
#     if intention in ['margin roi', 'performance'] and len(time)>=3:
#         inst_parts.append('Identify overall ROI trend if it is flat, increasing, decreasing or fluctuating.')
#         metric_cols = [col for col in overall_table.columns if intention in col.lower()]
#         trend_cols = [col for col in overall_table.columns if any(trend_key in col for trend_key in ['QOQ', 'YOY', 'HOH'])]
#         value_cols = [col for col in metric_cols if col not in trend_cols]
#         overall_table = _clean_numeric_after_tactics(overall_table, metric_cols, 'Driver')
#         overall_table[f'{intention} trend type'] = overall_table[value_cols].apply(bi_overall_trend_analyze, axis=1)
#         tactics_idx = overall_table.columns.get_loc('Driver')
#         left_cols = overall_table.columns[:tactics_idx].tolist()
#         cols_to_drop = [col for col in left_cols if overall_table[col].unique().tolist() == ['Overall']]
#         overall_table = overall_table.drop(columns=cols_to_drop, errors='ignore')
#         left_cols = [col for col in left_cols if col not in cols_to_drop]
#         overall_table = overall_table[left_cols+['Driver']+trend_cols+[f'{intention} trend type']]
#         overall_trend = _describe_df_by_row(overall_table, left_cols)
#         inst_parts.append(overall_trend)
#     return "\n" + "\n".join(inst_parts) if inst_parts else ""

def _outlier_filter(df, tactics_col='Tactics', percent_change_col=None):
    if len(df) < 2:
        return df, ""

    col_index = df.columns.get_loc(tactics_col) + 1
    if not percent_change_col:
        target_col = df.columns[col_index]
    else:
        target_col = percent_change_col
    if re.search(r'(YOY|QOQ|HOH|MOM)', target_col):
        col_clean = re.sub(r'\s*(YOY|QOQ|HOH|MOM)', '', target_col)
        show_col = f"{col_clean} % change"
    else:
        show_col = target_col
    show_col = re.sub(r'Month (\d{1,2})', _replace_month, show_col)
    show_col = show_col.replace("(MM)", "").strip()
    # Step 2: Remove outliers using IQR
    Q1 = df[target_col].quantile(0.25)
    Q3 = df[target_col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    # Identify outliers
    low_mask = df[target_col] <= lower_bound
    high_mask = df[target_col] >= upper_bound
    low_outliers = df.loc[low_mask, [tactics_col, target_col]]
    high_outliers = df.loc[high_mask, [tactics_col, target_col]]

    # Collect outlier tactic info
    def format_outlier_name(row):
        value = f"{row[target_col]:.1f}"
        if any(keyword in show_col.lower() for keyword in ["%", "contribution"]):
            suffix = "%"
        elif 'spend' in show_col.lower():
            suffix = "MM"
        else:
            suffix = ""
        return f"{row[tactics_col]} ({value}{suffix})"

    outliers_info = {
        f'Significantly lower {show_col.lower()}': [
            format_outlier_name(row) for _, row in low_outliers.iterrows()
        ],
        f'Significantly higher {show_col.lower()}': [
            format_outlier_name(row) for _, row in high_outliers.iterrows()
        ]
    }
    parts = [
        f"- {category}: {', '.join(tactics)}"
        for category, tactics in outliers_info.items() if tactics
    ]
    if not parts:
        outlier_line = ""
    else:
        outlier_line = "\n".join(parts)
    # Filter out outliers
    filtered_df = df[~(low_mask | high_mask)]
    return filtered_df, outlier_line


def _outlier_filter_roi(df, tactics_col='Tactics', roi_level_col = 'roi level'):
    col_index = df.columns.get_loc(tactics_col) + 1
    target_col = df.columns[col_index]
    if re.search(r'(YOY|QOQ|HOH|MOM)', target_col):
        col_clean = re.sub(r'\s*(YOY|QOQ|HOH|MOM)', '', target_col)
        show_col = f"{col_clean} % change"
    else:
        show_col = target_col
    show_col = re.sub(r'Month (\d{1,2})', _replace_month, show_col)
    show_col = show_col.replace("(MM)", "").strip()

    low_mask = df[roi_level_col] == 'low outlier'
    high_mask = df[roi_level_col] == 'high outlier'
    low_outliers = df.loc[low_mask, [tactics_col, target_col]]
    high_outliers = df.loc[high_mask, [tactics_col, target_col]]

    # Collect outlier tactic info
    def format_outlier_name(row):
        value = f"{row[target_col]:.1f}"
        if any(keyword in show_col.lower() for keyword in ["%", "contribution"]):
            suffix = "%"
        elif 'spend' in show_col.lower():
            suffix = "MM"
        else:
            suffix = ""
        return f"{row[tactics_col]} ({value}{suffix})"

    outliers_info = {
        f'Significantly lower {show_col.lower()}': [
            format_outlier_name(row) for _, row in low_outliers.iterrows()
        ],
        f'Significantly higher {show_col.lower()}': [
            format_outlier_name(row) for _, row in high_outliers.iterrows()
        ]
    }
    parts = [
        f"- {category}: {', '.join(tactics)}"
        for category, tactics in outliers_info.items() if tactics
    ]
    if not parts:
        outlier_line = ""
    else:
        outlier_line = "\n".join(parts)
    return outlier_line


def _get_outlier_pretext(df, ner_filter_dict, tactics_col='Tactics', percent_change_col=None):
    metric = ner_filter_dict['main_metric'][0]
    metric_cols = [col for col in df.columns if metric in col.lower()]
    df = _clean_numeric_after_tactics(df, metric_cols, tactics_col)
    df = _quartile_filter(df, metric=metric)
    _, outlier_recent = _outlier_filter(df, tactics_col, None)
    header = "Tactics that stood out:" if tactics_col == "Tactics" else "Drivers that stood out:"
    if percent_change_col:
        _, outlier_change = _outlier_filter(df, tactics_col, percent_change_col)
        return (header + "\n" + outlier_recent + "\n" + outlier_change) if outlier_recent or outlier_change else ''
    else:
        return (header + "\n" + outlier_recent) if outlier_recent else ''


def _quartile_filter(df, metric='spend', intention='not spend'):
    spend_cols = [col for col in df.columns if metric in col.lower()]
    if not spend_cols:
        return df
    if intention == 'spend' and any('month' in col.lower() for col in spend_cols):
        spend_cols = [col for col in spend_cols if 'month' in col.lower()]
        df['_spend_avg'] = df[spend_cols].mean(axis=1)
        spend_col = '_spend_avg'
    else:
        spend_col = spend_cols[0]
    spend_Q1 = df[spend_col].quantile(0.25)
    # for new roi level rule
    if not 'roi' in intention and len(df) > 2:
        filtered_df = df[df[spend_col] > spend_Q1]
    else:
        filtered_df = df
    spend_Q2 = df[spend_col].quantile(0.50)
    spend_Q3 = df[spend_col].quantile(0.75)

    def classify_spend(value):
        if value < spend_Q2:
            return 'low'
        elif value <= spend_Q3:
            return 'moderate'
        else:
            return 'high'

    filtered_df[f'{metric} level'] = df[spend_col].fillna(0).apply(classify_spend)
    level_rank = {'high': 0, 'moderate': 1, 'low': 2}
    filtered_df['level_sort'] = filtered_df[f'{metric} level'].map(level_rank)
    filtered_df = filtered_df.sort_values(by='level_sort').drop(columns='level_sort')
    if '_spend_avg' in filtered_df.columns:
        filtered_df = filtered_df.drop(columns=['_spend_avg'])
    return filtered_df


def _metric_level(df, metric):
    # tactics_col = 'Business Driver'
    # col_index = df.columns.get_loc(tactics_col) + 1
    # first_col_after_tactics = df.columns[col_index]
    metric_col = next((col for col in df.columns if metric in col.lower()), None)
    # metric_Q1 = df[metric_col].quantile(0.25)
    # filtered_df = df[df[metric_col] > metric_Q1]
    metric_Q2 = df[metric_col].quantile(0.50)
    metric_Q3 = df[metric_col].quantile(0.75)

    def classify_br_metric(value):
        if value <= metric_Q2:
            return 'low'
        elif value <= metric_Q3:
            return 'moderate'
        else:
            return 'high'

    df[f'{metric} level'] = df[metric_col].apply(classify_br_metric)
    return df

def _filter_total(metric_col, ner_filter_dict, total, dim_dict):
    # Extract metric and time parts from metric_col
    metric_col = re.sub(r'\bM(\d{1,2})\b', r'Month \1', metric_col)
    time_pattern = r'(Q[1-4]|H[1-2]|Month \d+)?\s*\d{4}$'
    metric = re.sub(time_pattern, '', metric_col).strip().lower()
    time_match = re.search(time_pattern, metric_col)
    metric_time_part = time_match.group(0).strip().lower() if time_match else ''
    metric_time_part = re.sub(r'q([1-4])', r'quarter \1', metric_time_part)

    total['metric_norm'] = total['metric'].str.strip().str.lower()
    total['time_norm'] = total['time'].astype(str).str.strip().str.lower()

    filtered = total[
        (total['metric_norm'] == metric) &
        (total['time_norm'].str.contains(metric_time_part)) & (
                    total['period_type'] == ner_filter_dict['period_type'][0])
        ]

    if 'media' in filtered['driver'].unique().tolist():
        filtered = filtered[filtered['driver'] == 'media']

    driver_idx = total.columns.get_loc('driver')
    left_cols = total.columns[:driver_idx]
    valid_cols = []
    if dim_dict:
        for col, val in dim_dict.items():
            # values = ner_filter_dict.get(col, ['overall'])
            if col in filtered.columns:
                filtered = filtered[filtered[col].isin(val)]
                valid_cols.append(col)
        for col in [col for col in left_cols if col not in dim_dict]:
            if 'overall' in filtered[col].unique():
                filtered = filtered[filtered[col] == 'overall']
                valid_cols.append(col)
    else:
        for col in left_cols:
            if 'overall' in filtered[col].unique():
                filtered = filtered[filtered[col]=='overall']
                valid_cols.append(col)

    result = filtered[list(set(valid_cols)) + ['value']]  # .to_dict(orient='records')
    return result


def _filter_ag_most_recent_metric(metric_cols, ner_filter_dict, df):
    metric_cols = [re.sub(r'\bM(\d{1,2})\b', r'Month \1', metric_col) for metric_col in metric_cols]
    time_pattern = r'(Q[1-4]|H[1-2]|Month \d+)?\s*\d{4}$'
    metrics = [re.sub(time_pattern, '', metric_col).strip().lower() for metric_col in metric_cols]
    time_match = re.search(time_pattern, metric_cols[0])
    metric_time_part = time_match.group(0).strip().lower() if time_match else ''
    metric_time_part = re.sub(r'q([1-4])', r'quarter \1', metric_time_part)

    df['metric_norm'] = df['metric'].str.strip().str.lower()
    df['time_norm'] = df['time'].astype(str).str.strip().str.lower()

    filtered = df[
        (df['metric_norm'].isin(metrics)) &
        (df['time_norm'].str.contains(metric_time_part)) & (
                df['period_type'] == ner_filter_dict['period_type'][0])
        ]

    left_cols = list(df.columns[: filtered.columns.get_loc("activity_group")])
    filtered = filtered.pivot_table(index = left_cols + ['activity_group_origin'], columns = "metric", values = "value").reset_index()

    filtered = filtered.rename({"activity_group_origin": "Tactics"}, axis = 1)



    return filtered





def _describe_df_by_row(df, left_cols):
    output = []
    if left_cols and not df.empty:
        grouped = df.groupby(left_cols, dropna=False)
        for group_vals, group_df in grouped:
            if not isinstance(group_vals, tuple):
                group_vals = (group_vals,)
            group_header = ", ".join(
                f"{col}: {val}" for col, val in zip(left_cols, group_vals)
            )
            output.append(group_header)
            for _, row in group_df.iterrows():
                row_lines = [
                    f"- {col}: {row[col]}"
                    for col in df.columns
                    if col not in left_cols
                ]
                output.append("\n".join(row_lines))
    else:
        for _, row in df.iterrows():
            row_lines = [f"- {col}: {row[col]}" for col in df.columns]
            output.append("\n".join(row_lines))
    return "\n\n".join(output)


def _clean_numeric_after_tactics(df, metric_cols, target_col='Tactics', charting=False):
    df_cleaned = df.copy()
    tactics_idx = df_cleaned.columns.get_loc(target_col)
    cols_to_clean = df_cleaned.columns[tactics_idx + 1:]  # All columns after 'Tactics'

    def parse_value(x):
        if isinstance(x, str):
            x = x.replace('%', '').replace('$', '').replace(',', '').strip()
            try:
                return float(x)
            except ValueError:
                return None
        return x

    for col in cols_to_clean:
        if df_cleaned[col].dtype == 'object':
            df_cleaned[col] = df_cleaned[col].apply(parse_value)

    if not charting:
        return df_cleaned

    # Drop rows with >75% NaNs in cols_to_clean
    threshold = int(len(metric_cols) * 0.56) + 1
    row_nulls_or_zeros = (df_cleaned[metric_cols].isna() | (df_cleaned[metric_cols] == 0)).sum(axis=1)
    df_cleaned = df_cleaned[row_nulls_or_zeros < threshold]
    return df_cleaned


def _recent_roi_spend(df, client_code, model_group_id, ner_filter_dict, roi_type):
    total = load_total_data(client_code, model_group_id)
    tactics_idx = df.columns.get_loc('Tactics')
    left_cols = df.columns[:tactics_idx].tolist()
    tactics_col = ['Tactics']
    # first_right_col = [df.columns[tactics_idx + 1]] # roi
    first_right_col = next((c for c in df.columns[tactics_idx + 1:] if roi_type in c.lower()), None)
    spend_col = next((col for col in df.columns if 'Spend' in col), None)
    other_main_metrics = [m for m in ner_filter_dict['main_metric'] if not m in roi_type]
    roi_cols = [col for col in df.columns if roi_type in col.lower()
                and all(m not in col.lower() for m in other_main_metrics)]
    spend_cols = [col for col in df.columns if 'spend' in col.lower()]
    if not roi_cols:
        return '', '', pd.DataFrame()
    if spend_col is None:
        return '', '', pd.DataFrame()
    if first_right_col is None:
        return "", "", pd.DataFrame()
    first_roi_col = [roi_cols[0]]
    first_spend_col = spend_cols[0]

    df = _clean_numeric_after_tactics(df, roi_cols)

    if len(df) > 3:
        df = _quartile_filter(df, intention=roi_type)
        # df, outlier = _outlier_filter(df)
        outlier = ""
    else:
        outlier = ""
        df['spend level'] = ""
    final_cols = left_cols + tactics_col + list(set([first_right_col] + first_roi_col)) + [spend_col, 'spend level']
    target_df = df[final_cols]
    if left_cols:
        target_df = target_df.rename(columns={col: col.lower() for col in left_cols})
        left_cols = [col.lower() for col in left_cols]
        for col in left_cols:
            target_df[col] = target_df[col].astype(str).str.lower()
        dim_dict = {col: target_df[col].unique().tolist() for col in left_cols}
    else:
        dim_dict = None

    # overall_roi = _filter_total(first_right_col[0], ner_filter_dict, total, dim_dict)
    # if overall_roi.empty:
    #     tmp_df = target_df.copy()
    #     tmp_df[f"{roi_type} index"] = 0
    #     tmp_df[f"{roi_type} level"] = ""
    #     # return '', '', pd.DataFrame()
    #     return '', '', tmp_df
    #
    # if left_cols and left_cols != ['marketing focus']:
    #     merged_df = target_df.merge(overall_roi[left_cols+['value']], on=left_cols, how='left')
    # else:
    #     overall_value = overall_roi['value'].iloc[0]
    #     merged_df = target_df.copy()
    #     merged_df['value'] = overall_value
    # merged_df = merged_df.rename(columns={'value': 'index'})
    # merged_df[f'{roi_type} index'] = merged_df[first_right_col[0]] / merged_df['index']
    # merged_df[f'{roi_type} index'] = merged_df[f'{roi_type} index'].round(2)

    def classify_roi_level(x):
        if x >= 1:
            return 'high'
        elif x >= 0.8:
            return 'moderate'
        else:
            return 'low'

    total_spend = _filter_total(first_spend_col.replace(" (MM)", ""), ner_filter_dict, total, dim_dict)
    # all_tactics_df = _filter_ag_most_recent_metric([first_right_col[0], first_spend_col.replace(" (MM)", "")], ner_filter_dict, all_df)

    # if left_cols and left_cols != ['marketing focus']:
    #     merged_df = merged_df.merge(total_spend[left_cols + ['value']], on= left_cols, how='left')
    # else:
    #     merged_df['total_spend'] = total_spend['value'].values[0]
    if left_cols and left_cols != ['marketing focus']:
        filtered_cols = [x for x in left_cols if x in total_spend.columns]
        merged_df = target_df.merge(total_spend[filtered_cols + ['value']], on=filtered_cols, how='left')
    else:
        merged_df = target_df.copy()
        merged_df['total_spend'] = total_spend['value'].values[0]
    merged_df = merged_df.rename(columns={'value': 'total_spend'})

    def classify_roi_level_new(df, roi_col, spend_col, new_col=f'{roi_type} level'):
        if len(df) < 3:
            df[new_col] = ""
            return df
        total_spend = df['total_spend'].values[0]
        data = df.copy()
        data = data.sort_values(roi_col)
        roi_s = data[roi_col]

        if all(roi_s.isna()):
            df[new_col] = ""
            return df

        spend_s = data[spend_col] / (total_spend / 1e6)
        spend_s_cum = spend_s.cumsum()
        outlier_level = 0.05
        spend_sum_threshold = 0.1
        spend_share_threshold = 0.05

        q05, q95 = roi_s.quantile([outlier_level, 1 - outlier_level])
        # low_roi_ls, high_roi_ls = roi_s[spend_s_cum <= max(spend_s_cum) * spend_sum_threshold], roi_s[spend_s_cum >= max(spend_s_cum) * (1 - spend_sum_threshold)]

        # cutoff_low = list(low_roi_ls)[-1] if len(low_roi_ls) else 0
        # cutoff_high = list(high_roi_ls)[0] if len(high_roi_ls) else max(roi_s) + 1
        large_spend_roi_ls = roi_s[spend_s > spend_share_threshold]
        cutoff_low = list(large_spend_roi_ls)[0] if len(large_spend_roi_ls) else q05
        cutoff_high = list(large_spend_roi_ls)[-1] if len(large_spend_roi_ls) else q95

        cutoff_low = min(q05, cutoff_low)
        cutoff_high = max(q95, cutoff_high)

        data[new_col] = ""

        data.loc[roi_s > cutoff_high, new_col] = "high outlier"
        data.loc[roi_s < cutoff_low, new_col] = "low outlier"

        # mask_inefficient = data[new_col].isna() & (roi_s < 1)
        # data.loc[mask_inefficient, new_col] = "inefficient"

        mask_rest = (data[new_col] == "") & roi_s.notna()
        if mask_rest.any():
            # by quantile
            rest = roi_s[mask_rest]
            bins = pd.qcut(rest, q=4, labels=False, duplicates="drop")
            label_map = {0: "inefficient", 1: "justifiable", 2: "good", 3: "excellent"}
            data.loc[mask_rest, new_col] = bins.map(label_map).astype(object)

            # by cum spend share
            # remaining_spend_share_l = list(spend_s_cum[roi_s < cutoff_low])[-1] if len(spend_s_cum[roi_s < cutoff_low]) else 0
            # remaining_spend_share_r = list(spend_s_cum[mask_rest])[-1]
            # cutoff_ls = np.linspace(remaining_spend_share_l, remaining_spend_share_r, 5)[1:-1]
            # bins = [-np.inf] + list(cutoff_ls) + [np.inf]
            # labels = ["inefficient", "justifiable", "good", "excellent"]
            #
            # data.loc[mask_rest, new_col] = pd.cut(spend_s_cum[mask_rest], bins=bins, labels=labels)

        # data = data[~data['roi level'].isin(["high outlier", "low outlier"])]

        return data

    # merged_df['roi level'] = merged_df['roi index'].apply(classify_roi_level)

    # merged_df = classify_roi_level_new(merged_df, first_right_col[0], spend_col)
    if left_cols:
        merged_df = merged_df.groupby(left_cols, group_keys=False).apply(classify_roi_level_new,
                                                                         roi_col=first_right_col, spend_col=spend_col)
    else:
        merged_df = classify_roi_level_new(merged_df, first_right_col, spend_col)

    # merged_df = merged_df.drop(columns=['index', 'total_spend'])
    merged_df = merged_df.drop(columns=['total_spend'])
    # new roi level
    level_rank_spend = {'high': 0, 'moderate': 1, 'low': 2}
    level_rank_roi = {"low outlier": 5, "inefficient": 4, "justifiable": 3, "good": 2, "excellent": 1,
                      "high outlier": 0}
    merged_df['roi_sort'] = merged_df[f'{roi_type} level'].map(level_rank_roi)
    merged_df['spend_sort'] = merged_df['spend level'].map(level_rank_spend)
    merged_df = merged_df.sort_values(by=['roi_sort', 'spend_sort'])
    merged_df = merged_df.drop(columns=['roi_sort', 'spend_sort'])
    # merged_df = merged_df.select_dtypes(exclude=['number'])
    inst = _describe_df_by_row(merged_df, left_cols)
    outlier = _outlier_filter_roi(merged_df, roi_level_col=f'{roi_type} level')
    return inst, outlier, merged_df


def _trend_roi_spend(df, ner_filter_dict, roi_type):
    tactics_idx = df.columns.get_loc('Tactics')
    left_cols = df.columns[:tactics_idx].tolist()
    tactics_col = ['Tactics']
    # roi = ner_filter_dict['main_metric'][0]
    other_main_metrics = [m for m in ner_filter_dict['main_metric'] if not m in roi_type]
    # roi_cols = [col for col in df.columns if roi in col.lower()
    #             and all(m not in col.lower() for m in ner_filter_dict['main_metric'][1:])]
    roi_cols = [col for col in df.columns if roi_type in col.lower()
                and all(m not in col.lower() for m in other_main_metrics)]
    if not roi_cols:
        return '', pd.DataFrame()
    spend_cols = [col for col in df.columns if 'spend' in col.lower()]
    if not spend_cols:
        return '', pd.DataFrame()
    df = _clean_numeric_after_tactics(df, roi_cols)
    if len(df) > 3:
        df = _quartile_filter(df, intention = 'roi')
        # df, _ = _outlier_filter(df)
    final_cols = left_cols + tactics_col + roi_cols + spend_cols
    target_df = df[final_cols]
    roi_trend_cols = [col for col in roi_cols if any(trend_key in col for trend_key in ['QOQ', 'YOY', 'HOH'])]
    spend_trend_cols = [col for col in spend_cols if any(trend_key in col for trend_key in ['QOQ', 'YOY', 'HOH'])]
    roi_value_cols = [col for col in roi_cols if col not in roi_trend_cols]
    spend_value_cols = [col for col in spend_cols if col not in spend_trend_cols]
    # if any(extract_latest_date(col) for col in roi_value_cols):
    #     sorted_roi_value_cols = sorted(
    #         roi_value_cols,
    #         key=lambda col: extract_latest_date(col) or (0, 0),
    #         reverse=True
    #     )
    # else:
    #     sorted_roi_value_cols = roi_value_cols
    # if any(extract_latest_date(col) for col in spend_value_cols):
    #     sorted_spend_value_cols = sorted(
    #         spend_value_cols,
    #         key=lambda col: extract_latest_date(col) or (0, 0),
    #         reverse=True
    #     )
    # else:
    #     sorted_spend_value_cols = spend_value_cols
    target_df[f'{roi_type} change type'] = target_df[roi_value_cols].apply(roi_spend_trend_analyze, axis=1)
    target_df['spend change type'] = target_df[spend_value_cols].apply(roi_spend_trend_analyze, axis=1)
    target_df_filtered = target_df[~((target_df[f'{roi_type} change type'].str.strip() == '') &
                (target_df['spend change type'].str.strip() == ''))]
    # target_df = target_df.select_dtypes(exclude=['number'])
    inst = _describe_df_by_row(target_df_filtered, left_cols)
    return inst, target_df


def _trend_roi_other(df, ner_filter_dict, roi_type):
    tactics_idx = df.columns.get_loc('Tactics')
    left_cols = df.columns[:tactics_idx].tolist()
    tactics_col = ['Tactics']
    # roi = ner_filter_dict['main_metric'][0]
    # roi_cols = [col for col in df.columns if roi in col.lower()
    #             and all(m not in col.lower() for m in ner_filter_dict['main_metric'][1:])]
    other_main_metrics = [m for m in ner_filter_dict['main_metric'] if not m in roi_type]
    roi_cols = [col for col in df.columns if roi_type in col.lower()
                and all(m not in col.lower() for m in other_main_metrics)]
    if not roi_cols:
        return '', pd.DataFrame()
    response_change_cols = [col for col in df.columns
    if 'response' in col.lower() and any(key in col for key in ['YOY', 'QOQ', 'HOH', '% Change'])]
    cost_per_change_cols = [col for col in df.columns
    if ('cost per' in col.lower() or 'cpm' in col.lower()) and any(key in col for key in ['YOY', 'QOQ', 'HOH', '% Change'])]
    response_col = next((col for col in df.columns if "response" in col.lower()), None)
    cost_per_col = next((col for col in df.columns if ("cost per" in col.lower() or 'cpm' in col.lower())), None)
    df = _clean_numeric_after_tactics(df, roi_cols)
    if len(df) > 3:
        df = _quartile_filter(df, intention = 'roi')
        # df, _ = _outlier_filter(df)
    final_cols = left_cols + tactics_col + roi_cols + response_change_cols + cost_per_change_cols
    if df.empty:
        return '', pd.DataFrame()
    target_df = df[final_cols]
    inst = _describe_df_by_row(target_df, left_cols)
    table_output = df[final_cols + [response_col, cost_per_col]]
    return inst, table_output


def _trend_roi_other_for_insights(df, ner_filter_dict, roi_type):
    tactics_idx = df.columns.get_loc('Tactics')
    left_cols = df.columns[:tactics_idx].tolist()
    tactics_col = ['Tactics']
    other_main_metrics = [m for m in ner_filter_dict['main_metric'] if not m in roi_type]
    roi_cols = [col for col in df.columns if roi_type in col.lower()
                and all(m not in col.lower() for m in other_main_metrics)]
    if not roi_cols:
        return '', pd.DataFrame()
    response_change_cols = [col for col in df.columns
    if 'response' in col.lower() and any(key in col for key in ['YOY', 'QOQ', 'HOH', '% Change'])]
    cost_per_change_cols = [col for col in df.columns
    if ('cost per' in col.lower() or 'cpm' in col.lower()) and any(key in col for key in ['YOY', 'QOQ', 'HOH', '% Change'])]
    activity_change_cols = [col for col in df.columns
    if 'activity' in col.lower() and 'cost' not in col.lower() and any(key in col for key in ['YOY', 'QOQ', 'HOH', '% Change'])]
    response_col = next((col for col in df.columns if "response" in col.lower()), None)
    cost_per_col = next((col for col in df.columns if ("cost per" in col.lower() or 'cpm' in col.lower())), None)
    activity_col = next((col for col in df.columns if 'activity' in col.lower() and 'cost' not in col.lower()
                         and not any(key in col for key in ['YOY', 'QOQ', 'HOH', '% Change'])), None)
    df = _clean_numeric_after_tactics(df, roi_cols)
    if len(df) > 3:
        df = _quartile_filter(df, intention = 'roi')
    final_cols = left_cols + tactics_col + roi_cols + response_change_cols + cost_per_change_cols + activity_change_cols
    if df.empty:
        return '', pd.DataFrame()
    target_df = df[final_cols]
    inst = _describe_df_by_row(target_df, left_cols)
    value_cols = [c for c in [response_col, cost_per_col, activity_col] if c is not None]
    table_output = df[final_cols + value_cols]
    return inst, table_output


def analyze_trend(trend_values, all_mean_abs, period_type, spike_ratio=4.0, volatile_ratio=1.5, flat_multiplier=0.06,
                  sign_ratio=0.8):
    # flat_threshold = flat_multiplier * all_mean_abs
    values = np.array(trend_values, dtype=np.float64)
    values = values[~np.isnan(values)]
    mean = np.mean(values)
    max_deviation = np.max(np.abs(values - mean))
    max_deviation_ratio = max_deviation / (abs(mean) + 1e-6)
    std = np.std(values)

    if len(values) < 3:
        return ""
    ordered = values[::-1]
    flat_res_std_threshold = flat_multiplier * (all_mean_abs + 1e-6)
    volatile_res_std_threshold = 0.35 * (all_mean_abs + 1e-6)
    rho, p_value = spearmanr(list(range(len(ordered))), ordered)
    if 'year' not in period_type:
        if max_deviation_ratio > spike_ratio:
            return 'spiking'
        elif std > volatile_res_std_threshold:
            return 'volatile'
        elif p_value < 0.05 and rho > 0:
            return 'consistently increasing'
        elif p_value < 0.05 and rho < 0:
            return 'consistently decreasing'
        elif std < flat_res_std_threshold:
            return 'flat'
        elif ordered[-1] > ordered[0]:
            return 'mild fluctuated (increase)'
        else:
            return 'mild fluctuated (decrease)'
    else:
        if max_deviation_ratio > spike_ratio:
            return 'spiking'
        elif p_value < 0.05 and rho > 0:
            return 'consistently increasing'
        elif p_value < 0.05 and rho < 0:
            return 'consistently decreasing'
        elif std < flat_res_std_threshold:
            return 'flat'
        elif std > volatile_res_std_threshold:
            return 'volatile'
        elif ordered[-1] > ordered[0]:
            return 'mild fluctuated (increase)'
        else:
            return 'mild fluctuated (decrease)'
    # diffs = np.diff(ordered)
    # signs = np.sign(diffs)
    # mean_abs = np.mean(np.abs(values))
    # std = np.std(values)
    # std_ratio = std / (all_mean_abs + 1e-6)
    # mean = np.mean(values)
    # max_deviation = np.max(np.abs(values - mean))
    # max_deviation_ratio = max_deviation / (abs(mean) + 1e-6)
    #
    # if len(values) < 5:
    #     if np.sum(signs >= 0) >= sign_ratio*len(signs) and ordered[-1] > ordered[0]:
    #         return "consistently increasing"
    #     elif np.sum(signs <= 0) >= sign_ratio*len(signs) and ordered[-1] < ordered[0]:
    #         return "consistently decreasing"
    #     elif std < flat_threshold:
    #         return "flat"
    #     elif max_deviation_ratio > spike_ratio:
    #         return 'sudden spike'
    #     elif std_ratio > volatile_ratio:# or (std_ratio > 0.5*all_mean_abs and all_mean_abs<0.5):
    #         return 'volatile'
    #     elif ordered[-1] > ordered[0]:
    #         return "mild fluctuation (increase)"
    #     else:
    #         return "mild fluctuation (decrease)"
    # else:
    #     y = ordered.reshape(-1)
    #     X = np.arange(len(ordered))
    #     X = sm.add_constant(X)
    #     model = sm.OLS(y, X).fit()
    #     slope = model.params[1]
    #     p_value = model.pvalues[1]
    #
    #     y_pred = model.predict(X)
    #     r2 = r2_score(y, y_pred)
    #     residuals = y - y_pred
    #     # residual_std = np.std(residuals)
    #     rmse = np.sqrt(np.mean(residuals ** 2))
    #     max_abs_residual = np.max(np.abs(residuals))
    #     residual_signs = np.sign(residuals)
    #     switches = np.sum(residual_signs[:-1] != residual_signs[1:])
    #     switch_ratio = switches / (len(y) - 1)
    #
    #     flat_slope_threshold = 0.5 * std / len(values)
    #     flat_res_std_threshold = 0.1 * (mean_abs + 1e-6)
    #     volatile_res_std_threshold = 0.35 * (mean_abs + 1e-6)
    #     spike_threshold = 3 * std
    #
    #     # slope_threshold = 0.01 * np.mean(np.abs(values))
    #     # slope_threshold = 2 *  std / len(values)
    #
    #     if abs(slope) < flat_slope_threshold and rmse < flat_res_std_threshold:
    #     # if abs(slope) < flat_slope_threshold and r2 > 0.9:
    #         return 'flat'
    #     elif max_abs_residual > spike_threshold:
    #         return 'sudden spike'
    #     elif rmse > volatile_res_std_threshold: #and switch_ratio>0.6
    #     # elif r2 < 0.3:
    #         return 'volatile'
    #     # elif slope > slope_threshold:
    #     elif slope > 0 and p_value < 0.05:
    #         return "consistently increasing"
    #     # elif slope < -slope_threshold:
    #     elif slope < 0 and p_value < 0.05:
    #         return "consistently decreasing"
    #     elif ordered[-1] > ordered[0]:
    #         return "mild fluctuation (increase)"
    #     else:
    #         return "mild fluctuation (decrease)"


def roi_spend_trend_analyze(trend_values, positive_threshold=0.75, negative_threshold=0.75):
    trend_values = trend_values.astype(float).dropna()
    if len(trend_values) < 3:
        return ''
    total = len(trend_values)
    values = np.array(trend_values, dtype=np.float64)
    values = values[::-1]
    rho, p_value = spearmanr(list(range(len(values))), values)
    if p_value < 0.5 and rho > 0 and values[-1] > values[0]:
        return 'increase'
    elif p_value < 0.5 and rho < 0 and values[-1] < values[0]:
        return 'decrease'
    else:
        return ''
    # if total < 5:
    #     diffs = np.diff(values)
    #     signs = np.sign(diffs)
    #     if np.sum(signs >= 0) >= positive_threshold * len(signs):
    #         return "increase"
    #     elif np.sum(signs <= 0) >= negative_threshold * len(signs):
    #         return "decrease"
    #     else:
    #         return ''
    # else:
    #     # y = values.reshape(-1, 1)
    #     # X = np.arange(total).reshape(-1, 1)
    #     #
    #     # model = LinearRegression()
    #     # model.fit(X, y)
    #     # slope = model.coef_[0][0]
    #     # slope_threshold = 0.01 * np.mean(np.abs(values))
    #     y = values.reshape(-1)
    #     X = np.arange(len(values))
    #     X = sm.add_constant(X)
    #     model = sm.OLS(y, X).fit()
    #     slope = model.params[1]
    #     # p_value = model.pvalues[1]
    #     slope_threshold = 0.5 * np.std(values) / len(values)
    #     if slope > slope_threshold and values[-1] > values[0]:
    #         return 'increase'
    #     elif slope < -slope_threshold and values[-1] < values[0]:
    #         return 'decrease'
    #     else:
    #         return ''


# def bi_overall_trend_analyze(trend_values):
#     #TODO: overall table only have most recent 3 period, so at most 2 YOY, in this case
#     #TODO: is it necessary to have flat type? Is it correct to use else as fluctuate?
#
#     trend_values = trend_values.astype(float).dropna()
#     if len(trend_values) == 0:
#         return ''
#     # total = len(trend_values)
#     values = np.array(trend_values, dtype=np.float64)
#     values = values[::-1]
#     diffs = np.diff(values)
#     # signs = np.sign(diffs)
#     if values[0]<values[-1]:
#         return "increase"
#     elif values[0]>values[-1]:
#         return "decrease"
#     else:
#         return ''
#     # if (trend_values>0).all():
#     #     return 'increase'
#     # elif (trend_values<0).all():
#     #     return 'decrease'
#     # else:
#     #     return 'fluctuate'

def level_trend_sort(df, metric):
    level_rank = {'high': 3, 'moderate': 2, 'low': 1}
    trend_rank = {
        'consistently increasing': 7,
        'consistently decreasing': 6,
        'volatile': 5,
        'flat': 4,
        'sudden spike': 3,
        'mild fluctuation (increase)': 2,
        'mild fluctuation (decrease)': 1
    }
    # Apply ranks
    df['level_rank'] = df[f'{metric} level'].map(level_rank)
    df['trend_rank'] = df[f'{metric} trend type'].map(trend_rank)
    # Custom combined rank logic
    # High/Moderate always outrank low
    df['combined_rank'] = df.apply(
        lambda row: (1, -row['level_rank'], -row['trend_rank']) if row['level_rank'] > 1
        else (2, -row['trend_rank']), axis=1
    )
    df_sorted = df.sort_values(by='combined_rank')
    df_sorted = df_sorted.drop(columns=['level_rank', 'trend_rank', 'combined_rank'])
    return df_sorted


def trend_contribution_sales(df, ner_filter_dict):
    metric = ner_filter_dict['main_metric'][0]
    tactics_col = ['Business Driver']
    tactics_idx = df.columns.get_loc(tactics_col[0])
    left_cols = df.columns[:tactics_idx].tolist()
    metric_cols = [col for col in df.columns if metric in col.lower()]
    trend_cols = [col for col in metric_cols if any(trend_key in col for trend_key in ['QOQ', 'YOY', 'HOH'])]
    value_cols = [col for col in metric_cols if col not in trend_cols]
    if any(extract_latest_date(col) for col in value_cols):
        sorted_value_cols = sorted(
            value_cols,
            key=lambda col: extract_latest_date(col) or (0, 0),
            reverse=True
        )
    else:
        sorted_value_cols = value_cols
    # if not trend_cols:
    #     return ''
    df = _clean_numeric_after_tactics(df, metric_cols, tactics_col[0])
    df = _quartile_filter(df, metric='contribution')
    # df = _metric_level(df, metric)
    df, _ = _outlier_filter(df, tactics_col='Business Driver')
    final_cols = left_cols + tactics_col + metric_cols + [f'{metric} level']
    target_df = df[final_cols]
    target_df = reorder_columns(target_df, value_cols, sorted_value_cols)
    target_df["all_mean_abs"] = target_df[sorted_value_cols].astype(float).abs().mean(axis=1)
    target_df[f"{metric} trend type"] = [
        analyze_trend(
            trend_values=row[sorted_value_cols].values,
            all_mean_abs=row["all_mean_abs"],
            period_type=ner_filter_dict.get('period_type', ['year'])[0],
            spike_ratio=3,
        )
        for _, row in target_df.iterrows()
    ]
    target_df = target_df.drop(columns=['all_mean_abs'])
    # if metric == 'contribution':
    #     target_df['max_contri'] = target_df[value_cols].max(axis = 1)
    #     target_df[f'{metric} trend type'] = target_df.apply(lambda x: "flat" if x['max_contri'] < 0.2 else x[f'{metric} trend type'], axis = 1)
    #     target_df = target_df.drop(['max_contri'], axis = 1)
    # target_df = target_df[target_df[f'{metric} trend type'].str.strip() != '']
    target_df = level_trend_sort(target_df, metric)
    target_df = target_df.rename(columns={'Business Driver': 'Marketing Driver'})
    inst = _describe_df_by_row(target_df, left_cols)
    return inst, target_df


# def trend_soc(df):
#     metric = 'source of change'
#     tactics_col = ['Business Driver']
#     tactics_idx = df.columns.get_loc(tactics_col[0])
#     left_cols = df.columns[:tactics_idx].tolist()
#     metric_cols = [col for col in df.columns if metric in col.lower()]
#     trend_cols = [col for col in metric_cols if any(trend_key in col for trend_key in ['QOQ', 'YOY', 'HOH'])]
#     metric_cols = [col for col in metric_cols if col not in trend_cols]
#
#     df = _clean_numeric_after_tactics(df, metric_cols, tactics_col[0])
#     df = _metric_level(df, metric)
#     final_cols = left_cols + tactics_col + metric_cols + [f'{metric} level']
#     target_df = df[final_cols]
#     target_df["all_mean_abs"] = target_df[metric_cols].astype(float).abs().mean(axis=1)
#
#     target_df[f"{metric} trend type"] = [
#         analyze_trend(
#             trend_values=row[metric_cols].values,
#             all_mean_abs=row["all_mean_abs"],
#             spike_ratio=4,
#             volatile_ratio=1.5,
#             flat_multiplier=0.5,
#             sign_ratio=0.75
#         )
#         for _, row in target_df.iterrows()
#     ]
#     target_df = target_df.drop(columns=['all_mean_abs'])
#     # target_df = target_df[target_df[f'{metric} trend type'].str.strip() != '']
#     inst = _describe_df_by_row(target_df, left_cols)
#     return inst

def extract_latest_date(col):
    matches = re.findall(r'Month (\d{1,2}) (\d{4})', col)
    if matches:
        dates = [(int(y), int(m)) for m, y in matches]
        return max(dates)
    else:
        return None


def reorder_columns(df, value_cols, sorted_value_cols):
    # Find the start and end index of the value_cols block
    start_idx = df.columns.get_loc(value_cols[0])
    end_idx = df.columns.get_loc(value_cols[-1])

    # Columns before, reordered target, and after
    before = df.columns[:start_idx].tolist()
    after = df.columns[end_idx + 1:].tolist()

    # Reconstruct the column order
    new_col_order = before + sorted_value_cols + after
    return df[new_col_order]


def trend_bi(df, ner_filter_dict, roi_level_df = None):
    metric = ner_filter_dict['main_metric'][0]
    tactics_col = ['Tactics']
    tactics_idx = df.columns.get_loc(tactics_col[0])
    left_cols = df.columns[:tactics_idx].tolist()
    metric_cols = [col for col in df.columns if metric in col.lower() and 'Share' not in col]
                   # and all(m not in col.lower() for m in ner_filter_dict['main_metric'][1:])]
    trend_cols = [col for col in metric_cols if any(trend_key in col for trend_key in ['QOQ', 'YOY', 'HOH'])]
    value_cols = [col for col in metric_cols if col not in trend_cols]
    if any(extract_latest_date(col) for col in value_cols):
        sorted_value_cols = sorted(
            value_cols,
            key=lambda col: extract_latest_date(col) or (0, 0),
            reverse=True
        )
    else:
        sorted_value_cols = value_cols
    # if not trend_cols:
    #     return ''
    df = _clean_numeric_after_tactics(df, metric_cols, tactics_col[0])
    df = _quartile_filter(df, intention=metric)
    if left_cols:
        df = df.rename(columns={col: col.lower() for col in left_cols})
        left_cols = [col.lower() for col in left_cols]
        for col in left_cols:
            df[col] = df[col].astype(str).str.lower()
    if 'roi' in metric:
        # if not roi_level_df.empty:
        if (
                not roi_level_df.empty
                and f"{metric} level" in roi_level_df.columns
                and roi_level_df[f"{metric} level"].astype(str).str.strip().ne("").any()
        ):
            df = pd.merge(df, roi_level_df[left_cols + ['Tactics', f'{metric} level']], on=left_cols + ['Tactics'],
                          how='left')
        else:
            df[f'{metric} level'] = ""
    elif metric != 'spend':
        df = _metric_level(df, metric)
    final_cols = left_cols + tactics_col + metric_cols + [f'{metric} level']
    target_df = df[final_cols]
    target_df = reorder_columns(target_df, value_cols, sorted_value_cols)
    target_df["all_mean_abs"] = target_df[sorted_value_cols].astype(float).abs().mean(axis=1)

    target_df[f"{metric} trend type"] = [
        analyze_trend(
            trend_values=row[sorted_value_cols].values,
            all_mean_abs=row["all_mean_abs"],
            period_type=ner_filter_dict.get('period_type', ['year'])[0],
            spike_ratio=3,
        )
        for _, row in target_df.iterrows()
    ]
    target_df = target_df.drop(columns=['all_mean_abs'])
    target_df = level_trend_sort(target_df, metric)
    # target_df = target_df.select_dtypes(exclude=['number'])
    # target_df = target_df[target_df[f'{metric} trend type'].str.strip() != '']
    inst = _describe_df_by_row(target_df, left_cols)
    return inst, target_df


def _parse_time_labels(time, period_type):
    fiscal_mappings = {
        'month': lambda y: [(y * 100 + m, [y * 100 + m], f"{month_name[m]} {y}") for m in range(1, 13)],
        'quarter': lambda y: [
            (y * 100 + 1, [y * 100 + m for m in range(1, 4)], f"q1 {y}"),
            (y * 100 + 4, [y * 100 + m for m in range(4, 7)], f"q2 {y}"),
            (y * 100 + 7, [y * 100 + m for m in range(7, 10)], f"q3 {y}"),
            (y * 100 + 10, [y * 100 + m for m in range(10, 13)], f"q4 {y}")
        ],
        'half': lambda y: [
            (y * 100 + 1, [y * 100 + m for m in range(1, 7)], f"h1 {y}"),
            (y * 100 + 7, [y * 100 + m for m in range(7, 13)], f"h2 {y}")
        ],
        'year': lambda y: [(y * 100 + 1, [y * 100 + m for m in range(1, 13)], f"year {y}")]}
    yyyymm_label_map = {}
    for t in time:
        match = re.match(rf'(?:fiscal\s+)?{re.escape(period_type)}(?: (\d+))? (\d{{4}})', t.strip(), re.IGNORECASE)
        if match:
            num_str, year = match.group(1), int(match.group(2))
            all_periods = fiscal_mappings[period_type](year)
            if num_str:
                idx = int(num_str) - 1
                if 0 <= idx < len(all_periods):
                    _, months, label = all_periods[idx]
                    for yyyymm in months:
                        yyyymm_label_map[yyyymm] = label
            else:
                for _, months, label in all_periods:
                    for yyyymm in months:
                        yyyymm_label_map[yyyymm] = label
    return yyyymm_label_map


def _get_closest_label(yyyymm, label_map, period_type):
    # Try to find exact match first
    if yyyymm in label_map:
        return label_map[yyyymm]
    # Else, infer label
    year = yyyymm // 100
    month = yyyymm % 100
    if period_type == 'month':
        return f"{month_name[month]} {year}"
    elif period_type == 'quarter':
        quarter = (month - 1) // 3 + 1
        return f"q{quarter} {year}"
    elif period_type == 'half':
        half = 1 if month <= 6 else 2
        return f"h{half} {year}"
    elif period_type == 'year':
        return f"year {year}"
    return str(yyyymm)

def _validate_time_range(ner_res_dict, ner_filter_dict, data=None):
    start, end = ner_res_dict.get('start', 'irrelevant'), ner_res_dict.get('end', 'irrelevant')
    time = ner_filter_dict.get('time', None)
    curr_time = int(datetime.now().strftime("%Y%m"))
    period_type = ner_filter_dict.get('period_type', ['year'])[0]
    latest_time = ner_filter_dict.get('latest_time', 0)
    latest_period_type = ner_filter_dict.get('latest_period_type', 'irrelevant')
    mapping = {
        'fiscal year': 'year',
        'fiscal quarter': 'quarter',
        'fiscal month': 'month',
        'fiscal half': 'half',
        'year': 'year',
        'quarter': 'quarter',
        'month': 'month',
        'half': 'half'
    }
    period_type = mapping.get(period_type, period_type)

    if time is None or start=='irrelevant' or end=='irrelevant':
        return ""
    if data is not None and not data.empty:
        end_data = int(data['end'].max())
    else:
        end_data = max(int(e) for e in end)
    label_map = _parse_time_labels(time, period_type)
    if not label_map:
        return ""
    all_yyyymm = sorted(label_map.keys())
    min_yyyymm = min(all_yyyymm)
    max_yyyymm = max(all_yyyymm)
    time_str=""
    incomplete_str = ""
    period_str = ""
    valid_start = _get_closest_label(min_yyyymm, label_map, period_type)
    valid_end = _get_closest_label(max_yyyymm, label_map, period_type)

    # requested period type (raw ner) differs from the resolved period type
    req_p = ner_res_dict.get('period_type', 'irrelevant')
    if isinstance(req_p, list):
        req_p = req_p[0] if req_p else 'irrelevant'
    if isinstance(req_p, str) and req_p.strip() and req_p != 'irrelevant':
        req_p_norm = mapping.get(req_p.split(',')[0].strip(), req_p.split(',')[0].strip())
        if req_p_norm != period_type:
            period_str = (f"Requested period type '{req_p_norm}' is not available in the data; "
                          f"results are shown by {period_type}. ")


    requested_min = min(int(s) for s in start)
    requested_max = max(int(e) for e in end)
    if requested_min < min_yyyymm or requested_max > max_yyyymm:
        req_start_label = _get_closest_label(requested_min, label_map, period_type)
        req_end_label = _get_closest_label(requested_max, label_map, period_type)
        if req_start_label == req_end_label:
            requested_str = f"data for {req_start_label}"
        else:
            requested_str = f"data from {req_start_label} to {req_end_label}"
        if valid_start == valid_end:
            valid_range_str = f"in {valid_start}"
        else:
            valid_range_str = f"from {valid_start} to {valid_end}"
        time_str = (f"You requested {requested_str}, which falls outside of the valid time range "
                    f"({valid_range_str}). ")

    if end_data >= curr_time:
        end_year = str(end_data)[:4]

        if latest_time!=0 and latest_period_type!='irrelevant':
            latest_year = int(str(latest_time)[:4])
            latest_month = int(str(latest_time)[4:])

            if "quarter" in latest_period_type:
                latest_quarter = (latest_month - 1) // 3 + 1
                latest_label = f"Q{latest_quarter} {latest_year}"
            elif "half" in latest_period_type:
                latest_half = 1 if latest_month <= 6 else 2
                latest_label = f"H{latest_half} {latest_year}"
            else:  # default to month label
                latest_label = datetime.strptime(str(latest_time), "%Y%m").strftime("%B %Y")

            incomplete_str = f"{end_year} data is YTD through {latest_label}."
        else:
            curr_label = datetime.now().strftime("%B %Y")
            incomplete_str = (
                f"Some of your requested dates include future periods "
                f"(current date is {curr_label}).")

    time_str = period_str + time_str + incomplete_str
    return time_str


def _genome_fixtext(principle_df, data, intention, reduce_budget=False):
    if principle_df.empty or data.empty:
        return ''
    if intention=='planner':
        related_m= None
    else:
        related_m = data['metric'].unique().tolist()
    col = next((c for c in ('marketing_channel', 'media_channel') if c in data.columns), None)
    mkt_channels = data[col].unique().tolist()
    d = principle_df.copy()
    for col in ['matchingIntention','marketingChannel','relatedMetric','genomePrinciple']:
        d[col] = d[col].fillna('').astype(str).str.strip()
    intent_mask = d['matchingIntention'].eq(intention)

    blank_chan = d['marketingChannel'].eq('')
    if intention in ['planner', 'spending']:
        chan_mask = blank_chan
    else:
        chan_mask = d['marketingChannel'].isin(mkt_channels) | blank_chan

    blank_rel = d['relatedMetric'].eq('')
    if not related_m:
        rel_mask = pd.Series(True, index=d.index)
    else:
        rel_mask = d['relatedMetric'].isin(related_m) | blank_rel

    mask = intent_mask & chan_mask & rel_mask
    sentences = d.loc[mask, 'genomePrinciple'].tolist()
    if intention == 'planner' and not reduce_budget:
        sentences = sentences[1:]
    seen = set()
    unique_sentences = []
    for s in sentences:
        if s not in seen:
            seen.add(s)
            unique_sentences.append(s)
    return "\n".join(unique_sentences)


def _bi_spend_range(client_code, model_group_id, principle, genome_principle, data, dim_cols):
    df = load_total_data(client_code, model_group_id)
    df = df[(df['metric'] == 'share of spend') & (df['period_type'].str.contains('year', case=False, na=False))]
    df["year_int"] = df["time"].str.extract(r"(\d{4})").astype(float)
    max_year = df["year_int"].max()
    df = df[df["year_int"] == max_year]
    channel_col = 'media_channel' if 'media_channel' in data.columns else 'marketing_channel'
    channels = data[channel_col].unique().tolist()
    df = df[df['driver'].isin(channels)]
    if dim_cols:
        level_map = {}
        for col in dim_cols:
            level_map[col] = data[col].unique().tolist()
        for col, allowed_values in level_map.items():
            if col in df.columns:
                df = df[df[col].isin(allowed_values)]
    else:
        for col in dim_cols:
            df = df[df[col] == 'overall']
    if not principle.empty and not df.empty:
        summaries = []
        df = pd.merge(df, principle, how='inner', left_on='driver', right_on='marketingChannel')
        scenario_summaries = []
        for _, row in df.iterrows():
            channel = row['marketingChannel']
            fore_share = row['value']
            median = row['medianPoint']
            left = row['maintainSpendShareRangeLeft']
            right = row['maintainSpendShareRangeRight']
            # Forecast interpretation
            fore_comp = "higher than" if fore_share > median else "lower than" if fore_share < median else "equal to"
            fore_range = "within" if left <= fore_share <= right else "out of"
            if genome_principle.empty:
                genome_str = ''
            else:
                matched = genome_principle[
                    (genome_principle['matchingIntention'] == 'spending') &
                    (genome_principle['marketingChannel'] == channel)]
                genome_str = matched['genomePrinciple'].iloc[0] if not matched.empty else ''
            summary = (
                f"- **{channel.title()}**: The spend share is {fore_share:.2f}%, which is {fore_comp} the median point ({median:.2f}%), "
                f"and {fore_range} the spend share range ({left:.2f}%–{right:.2f}%). "
                f"{genome_str}")
            scenario_summaries.append(summary)
        scenario_block = "\n".join(scenario_summaries)
        summaries.append(scenario_block)
        return "\n\n".join(summaries)
    else:
        return ''


def dedup_post(detail_view, agg_view, df_dict, detail_level, agg_level, client_code, main_metric, type='bi'):
    if client_code in ['HILSP', 'LINKEDIN']:
        if detail_view.empty:
            return detail_view, agg_view

        # value_col_detail = detail_view.columns[detail_view.columns.get_loc('Tactics') + 1]
        # value_col_agg = agg_view.columns[agg_view.columns.get_loc('Tactics') + 1] if not agg_view.empty else value_col_detail
        d_start = detail_view.columns.get_loc('Tactics') + 1
        detail_value_cols = list(detail_view.columns[d_start:])
        if agg_view is not None and not agg_view.empty and 'Tactics' in agg_view.columns:
            a_start = agg_view.columns.get_loc('Tactics') + 1
            agg_value_cols = list(agg_view.columns[a_start:])
        else:
            agg_value_cols = detail_value_cols

        common = set(detail_value_cols) & set(agg_value_cols)
        if main_metric:
            main_metric = ["source of change" if m == "sourceofchange" else m for m in main_metric]
            value_cols = [c for c in detail_value_cols if c in common
                   and all(m in c.lower() for m in main_metric) and not any(g in c.lower() for g in ['yoy', 'qoq', 'hoh'])]
        else:
            value_cols = [c for c in detail_value_cols if c in common and not any(g in c.lower() for g in ['yoy', 'qoq', 'hoh'])]
        # if value_col_detail != value_col_agg:
        #     return detail_view, agg_view
        if not value_cols:
            return detail_view, agg_view
            # value_col = value_col_detail
        NA_SENTINEL = "__NA__"
        def make_key_index(df, cols):
            tmp = df[cols].copy()
            tmp = tmp.where(~tmp.isna(), NA_SENTINEL)
            return pd.MultiIndex.from_frame(tmp, names=cols)
        def remove_from_agg(detail_df, agg_df):
            if detail_df.empty or agg_df.empty:
                return agg_df
            remove_keys = make_key_index(detail_df, value_cols)
            agg_keys = make_key_index(agg_df, value_cols)
            return agg_df.loc[~agg_keys.isin(remove_keys)].copy()

        if type == 'bi':
            if agg_level == 'mg':
                # agg_view = agg_view.drop_duplicates(subset=['Tactics', value_cols[0]], keep='first')
                agg_view = agg_view.drop_duplicates(subset=['Tactics']+value_cols, keep='first')

            if detail_level == 'mg':
                # detail_view = detail_view.drop_duplicates(subset=['Tactics', value_cols[0]], keep='first')
                detail_view = detail_view.drop_duplicates(subset=['Tactics']+value_cols, keep='first')

            if agg_level == 'ag' and detail_level == 'm':
                ag_value_df = df_dict['m_df'][['measure_origin', 'activity_group_origin']].drop_duplicates()
                detail_view = pd.merge(detail_view, ag_value_df,
                                       how='left', left_on='Tactics', right_on='measure_origin')
                # pairs_remove = set(zip(detail_view['activity_group_origin'], detail_view[value_col]))
                # agg_view = agg_view[~agg_view.apply(lambda x: (x['Tactics'], x[value_col]) in pairs_remove, axis=1)]
                agg_view = remove_from_agg(detail_view, agg_view)
                detail_view = detail_view.drop([c for c in detail_view.columns if '_origin' in c], axis=1)
            elif (detail_level == 'ag' and agg_level == 'mg') or (detail_level == 'mg' and agg_level == 'ag'):
                # pairs_remove = set(zip(detail_view['Tactics'], detail_view[value_col]))
                # agg_view = agg_view[~agg_view.apply(lambda x: (x['Tactics'], x[value_col]) in pairs_remove, axis=1)]
                agg_view = remove_from_agg(detail_view, agg_view)
            elif detail_level == 'm' and agg_level == 'mg':
                ag_value_df_detail = df_dict['m_df'][['measure_origin', 'activity_group_origin']].drop_duplicates()
                ag_value_df_agg = df_dict['mg_df'][['measure_group_origin', 'activity_group_origin']].drop_duplicates()
                detail_view = pd.merge(detail_view, ag_value_df_detail,
                                       how='left', left_on='Tactics', right_on='measure_origin')
                if 'Marketing Focus' in agg_view.columns:
                    agg_merge_col = ['Marketing Focus', 'Tactics']
                    agg_merge_col_r = ['activity_group_origin', 'measure_group_origin']
                else:
                    agg_merge_col = 'Tactics'
                    agg_merge_col_r = 'measure_group_origin'
                agg_view = pd.merge(agg_view, ag_value_df_agg,
                                       how='left', left_on=agg_merge_col, right_on=agg_merge_col_r)
                agg_view = remove_from_agg(detail_view, agg_view)
                detail_view = detail_view.drop(columns=[c for c in detail_view.columns if '_origin' in c])
                agg_view = agg_view.drop(columns=[c for c in agg_view.columns if '_origin' in c])
        else:
            if detail_level == 'm' and agg_level == 'mg':
                agg_view = remove_from_agg(detail_view, agg_view)

    return detail_view, agg_view

def select_scenarios_by_kpi(planner_filter: pd.DataFrame, kpis: list[str]) -> pd.DataFrame:
    """
    Select scenarios AND KPI rows based on substring matching.

    Priority:
    1) Scenario contains ALL kpis somewhere across its KPI rows
    2) If none, scenario contains ANY kpi

    Returns:
    - Only matched Scenario IDs
    - Only KPI rows that actually matched
    """

    df = planner_filter.copy()

    # Normalize
    df["_kpi_norm"] = df["KPI"].astype(str).str.lower()
    kpis_norm = [k.lower() for k in kpis]

    # Row-level ANY match mask (reused later)
    row_match_any = False
    for k in kpis_norm:
        row_match_any |= df["_kpi_norm"].str.contains(k, na=False)

    # ---- Step 1: scenario contains ALL kpis (across rows) ----
    scenario_has_all = (
        df.groupby("Scenario ID")["_kpi_norm"]
          .apply(lambda s: all(s.str.contains(k).any() for k in kpis_norm))
    )

    scenario_ids = scenario_has_all[scenario_has_all].index

    if len(scenario_ids) > 0:
        # keep only matched scenarios AND matched KPI rows
        result = df[
            df["Scenario ID"].isin(scenario_ids) & row_match_any
        ]

    else:
        # ---- Step 2: fallback ANY ----
        scenario_has_any = (
            df.groupby("Scenario ID")["_kpi_norm"]
              .apply(lambda s: any(s.str.contains(k).any() for k in kpis_norm))
        )

        scenario_ids = scenario_has_any[scenario_has_any].index

        result = df[
            df["Scenario ID"].isin(scenario_ids) & row_match_any
        ]

    return result.drop(columns="_kpi_norm")

def planner_kpi_filter(planner_filter_raw, planner_df, ner_res_dict, ner_filter):
    kpi = ner_res_dict.get('kpi', 'irrelevant')
    planner_filter = planner_filter_raw.copy()
    if 'KPI' in planner_filter.columns and kpi == 'specific':
        kpi = ner_filter.get('kpi', ['overall'])
        detailed_kpis = ner_filter.get('detailed_kpi', ['overall'])
        if 'overall' not in detailed_kpis:
            kpis = detailed_kpis
        else:
            kpis = kpi
        # kpis = kpi + detailed_kpis

        if 'overall' not in kpis:
            planner_filter = select_scenarios_by_kpi(planner_filter, kpis)
            if planner_filter.empty:
                planner_filter = select_scenarios_by_kpi(planner_filter_raw, kpi)
        else:
            planner_filter = planner_filter[planner_filter['KPI'] == planner_filter['default_kpi']]
        planner_df = pd.merge(planner_df, planner_filter[['Scenario ID', 'KPI']], how='inner',
                              on=['Scenario ID', 'KPI'])
    elif 'KPI' in planner_filter.columns and kpi == 'irrelevant':
        planner_filter = planner_filter[planner_filter['KPI'] == planner_filter['default_kpi']]
        planner_df = pd.merge(planner_df, planner_filter[['Scenario ID', 'KPI']], how='inner',
                              on=['Scenario ID', 'KPI'])
    return planner_df, planner_filter

def apply_level_mapping_display(client_code, model_group_id, df):
    support_df = load_level_rename_display(client_code, model_group_id)
    if not support_df.empty and not df.empty:
        cols_to_drop = [c for c in df.columns
            if "Source Of Change" in c and any(x in c for x in ["QOQ", "YOY", "MOM", "HOH"])]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
    if support_df.empty or df.empty:
        return df

    df = df.copy()

    # ---- column rename (only matched ones) ----
    # support originals are lowercase by assumption
    support_originals = support_df[['original', 'mapped']].drop_duplicates()

    # build mapping using lowercase comparison, but keep original df column names
    col_map = {}
    cols_to_drop_fix = []
    for col in df.columns:
        col_l = col.lower()
        if col_l in support_originals['original'].values:
            mapped = support_originals.loc[support_originals['original'] == col_l, 'mapped'].iloc[0]
            if pd.isna(mapped):
                cols_to_drop_fix.append(col)
            else:
                col_map[col] = mapped
    df = df.drop(columns=cols_to_drop_fix).rename(columns=col_map)
    # ---- value mapping (only for mapped columns) ----
    for orig_lower, mapped_col in support_originals.set_index('original')['mapped'].items():
        if mapped_col not in df.columns:
            continue
        value_map = (support_df[support_df['original'] == orig_lower]
            .set_index('original_value')['mapped_value'].to_dict())
        s = df[mapped_col]
        df[mapped_col] =(s.where(s.notna(), None).astype(str).str.lower().map(value_map).fillna(s))
    # ---- sorting: follow 'original' order from support_df, sort values alphabetically ----
    # e.g. business_unit_for_kpi, kpi, detailed_kpi, country (in that order)
    sort_originals = support_df['original'].drop_duplicates().tolist()
    # map originals -> mapped column names (after renaming)
    original_to_mapped = support_originals.set_index('original')['mapped'].to_dict()
    sort_cols = [original_to_mapped[o] for o in sort_originals if o in original_to_mapped]
    # keep only those that actually exist in df
    sort_cols = [c for c in sort_cols if c in df.columns]
    if sort_cols:
        # stable sort, case-insensitive alphabetical on each key
        df = df.sort_values(by=sort_cols, kind="mergesort", key=lambda s: s.astype(str).str.lower())
    # ---- extra sorting: global split by default values (A then B across groups) ----
    # Desired final order example:
    #   group1 = A + B, group2 = A + B  ==>  group1A + group2A + group1B + group2B
    default_kpi_map = (support_df.loc[support_df["default_value"].notna(), ["original", "default_value"]]
        .set_index("original")["default_value"].to_dict())
    if default_kpi_map and sort_cols:
        # preserve original row order after prior sorting
        df["__row_order__"] = range(len(df))
        # establish group ordering based on current (already sorted) appearance
        # group is defined as all sort cols except the "default column" we are splitting on
        # We'll apply the split for any original that has a default_value and exists in df.
        for orig_lower, default_val in default_kpi_map.items():
            mapped_col = original_to_mapped.get(orig_lower)
            if not mapped_col or mapped_col not in df.columns or default_val is None:
                continue
            group_cols = [c for c in sort_cols if c != mapped_col and c in df.columns]
            if not group_cols:
                # no grouping keys; just bring defaults to top of entire table
                group_cols = []
            default_str = str(default_val).strip().lower()

            is_default = (df[mapped_col].where(df[mapped_col].notna(), None)
                .astype(str).str.lower().eq(default_str)).astype(int)
            df["__is_default__"] = is_default
            if group_cols:
                # group order by first appearance
                grp_key = df[group_cols].astype(str).agg("\x1f".join, axis=1)
                first_seen = {k: i for i, k in enumerate(grp_key.drop_duplicates().tolist())}
                df["__group_order__"] = grp_key.map(first_seen)

                df = df.sort_values(by=["__is_default__", "__group_order__", "__row_order__"],
                    ascending=[False, True, True],kind="mergesort",).drop(columns=["__group_order__"])
            else:
                df = df.sort_values(by=["__is_default__", "__row_order__"],
                    ascending=[False, True],kind="mergesort",)
            df = df.drop(columns=["__is_default__"])
        df = df.drop(columns=["__row_order__"])
    # ---- reorder columns based on support_df original order ----
    sort_originals = support_df['original'].drop_duplicates().tolist()
    # original -> mapped (after rename)
    original_to_mapped = support_originals.set_index('original')['mapped'].to_dict()
    ordered_cols = [original_to_mapped[o] for o in sort_originals if original_to_mapped.get(o) in df.columns]
    # keep other columns in their existing order
    remaining_cols = [c for c in df.columns if c not in ordered_cols]
    df = df[ordered_cols + remaining_cols]

    return df

def level_filter_default(support_df, data, ner_res_dict):
    if data.empty:
        return data
    default_kpi_map = (support_df.loc[support_df["default_value"].notna(), ["original", "default_value"]]
                       .set_index("original")["default_value"].to_dict())
    # mapped_level = (support_df[["original", "mapped"]].drop_duplicates()
    #                    .set_index("original")["mapped"].to_dict())
    # col_lookup = {c: c for c in data.columns}
    mask = False
    if default_kpi_map:
        for key, value in default_kpi_map.items():
            if ner_res_dict.get(key, 'irrelevant') == 'irrelevant':
                # mapped_key = mapped_level[key]
                # col = col_lookup.get(mapped_key)
                # col = mapped_key
                if key is not None:
                    mask = mask | (data[key].astype(str).str.lower() == value)
    if isinstance(mask, bool):
        return data
    if not mask.any():
        return data
    filtered_df = data[mask]
    df = filtered_df if not filtered_df.empty else data
    return df
# def readout_adjust(client_code, model_group_id, data, agg_data, ner_res_dict, threshold = 4):
#     support_df = load_level_rename_display(client_code, model_group_id)
#     intention = ner_res_dict.get('intention', ['roi'])[0]
#     if support_df.empty:
#         return None, None
#     df = level_filter_default(support_df, data, ner_res_dict)
#     if df.empty:
#         return None, None
#     agg_df = level_filter_default(support_df, agg_data, ner_res_dict)
#     level_cols = [c for c in df.columns if c in support_df['mapped'].unique().tolist()]
#     combination_num = len(df[level_cols].drop_duplicates())
#     agg_table_str, detail_table_str, _, _ = table_to_text(agg_df, df, take_head=False)
#     if combination_num>threshold:
#         agg_analysis = 'analysis'
#         detail_analysis = 'analysis'
#         agg_table_str+=agg_analysis
#         detail_table_str+=detail_analysis
#     if intention=='contribution':
#         group_cols = support_df[support_df['contribution_agg']=='Y']['mapped'].unique().tolist()
#         sum_cols = [c for c in df.columns if "contribution" in c.lower() and "% change" not in c.lower()]
#         df_conv = df.copy()
#         for c in sum_cols:
#             df_conv[c] = pd.to_numeric(df_conv[c].astype(str).str.replace("%", "", regex=False),errors="coerce")
#         group_df = df_conv.groupby(list(group_cols), as_index=False)[sum_cols].sum(min_count=1)
#         for c in sum_cols:
#             group_df[c] = group_df[c].round(2).astype(str) + "%"
#         _, detail_agg_str, _, _ = table_to_text(pd.DataFrame(), group_df, take_head=False)
#         detail_table_str = "Aggregated Contribution: \n" + detail_agg_str + "\n\nDetailed Contribution: \n" + detail_table_str
#         agg_table_str = ''
#
#     return agg_table_str, detail_table_str

def compare_country_by_group_mwu(df, dim_cols, main_metric, country_col="country", tactics_col="tactics", time_col="time",
    value_col="value", min_n=3, alpha=0.05,):
    data = df.copy()
    data = data[data['metric']==main_metric]
    if data.empty:
        return pd.DataFrame(), ""
    countries = data[country_col].dropna().unique()
    if len(countries) < 2:
        return pd.DataFrame(), ""

    c1, c2 = countries[:2]
    # group keys = rest dims (exclude country & tactics) + time
    group_keys = [c for c in dim_cols if c not in [country_col, tactics_col] and c in data.columns]
    if time_col in data.columns and time_col not in group_keys:
        group_keys.append(time_col)
    rows = []
    for keys, g in data.groupby(group_keys, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        key_dict = dict(zip(group_keys, keys))
        # if duplicates per (country, tactics), average first
        if tactics_col in g.columns:
            g2 = g.groupby([country_col, tactics_col], dropna=False)[value_col].mean().reset_index()
        else:
            g2 = g[[country_col, value_col]].copy()

        a = g2.loc[g2[country_col] == c1, value_col].to_numpy(dtype=float)
        b = g2.loc[g2[country_col] == c2, value_col].to_numpy(dtype=float)
        a = a[~np.isnan(a)]
        b = b[~np.isnan(b)]
        # if any group insufficient → return empty immediately
        if len(a) < min_n or len(b) < min_n:
            continue
        # Mann–Whitney U (two-sided)
        U, p = mannwhitneyu(a, b, alternative="two-sided")
        # dominance ~ P(a > b)
        dominance = float(U / (len(a) * len(b)))
        # if p < alpha:
        if dominance > 0.5 and dominance<=0.6:
            concl = f"{c1} slightly outperform {c2}"
        elif dominance > 0.6:
            concl = f"{c1} outperform {c2}"
        elif dominance < 0.5 and dominance >= 0.4:
            concl = f"{c2} slightly outperform {c1}"
        elif dominance < 0.4:
            concl = f"{c2} outperform {c1}"
        else:
            concl = ''
        # else:
        #     concl = ''

        rows.append({**key_dict, "conclusion": concl, "dominance_score": dominance, "p_value": float(p),
            f"n_{c1}": int(len(a)), f"n_{c2}": int(len(b)),})

    result_df = pd.DataFrame(rows)

    # nice ordering
    ordered = group_keys + ["conclusion", "dominance_score", "p_value", f"n_{c1}", f"n_{c2}"]
    if not result_df.empty:
        result_df = result_df[ordered]
        lines=[]
        for _, row in result_df.iloc[::-1].iterrows():
            # join left column values with "-"
            left_part = "-".join(str(row[col]) for col in group_keys if col!=time_col)
            sentence = f"For {left_part}, {row['conclusion']} in {row[time_col]}."
            lines.append(sentence)

        return result_df, " ".join(lines) + "\n\n"
    else:
        return result_df, ""

def readout_adjust(client_code, model_group_id, data, str_q, ner_res_dict, ner_dict, how_many, driver_high, ner_filter, sort_by_col, intention, growth_col, trend_check, rank_check, sort_metric, metric_df, threshold = 2):
    support_df = load_level_rename_display(client_code, model_group_id)
    # intention = ner_res_dict.get('intention', ['roi'])[0]
    if support_df.empty:
        return str_q, data, ner_dict
    df = level_filter_default(support_df, data, ner_res_dict)
    if df.empty:
        return str_q, data, ner_dict
    # ---- sorting: follow 'original' order from support_df, sort values alphabetically ----
    # e.g. business_unit_for_kpi, kpi, detailed_kpi, country (in that order)
    sort_cols = support_df['original'].drop_duplicates().tolist()
    # keep only those that actually exist in df
    sort_cols = [c for c in sort_cols if c in df.columns]
    if sort_cols:
        # stable sort, case-insensitive alphabetical on each key
        df = df.sort_values(by=sort_cols, kind="mergesort", key=lambda s: s.astype(str).str.lower())
    # ---- reorder columns based on support_df original order ----
    ordered_cols = [o for o in sort_cols if o in df.columns]
    ordered_dims = [o for o in sort_cols if o in ner_dict['dimension']]
    ner_dict['dimension'] = ordered_dims
    # keep other columns in their existing order
    remaining_cols = [c for c in df.columns if c not in ordered_cols]
    df = df[ordered_cols + remaining_cols]

    top_data = _top_data(df, ner_dict, how_many)
    top_data = _filter_top_media_channel(driver_high, how_many, ner_filter, ner_dict, sort_by_col, top_data)
    str_q = _get_readout(intention, growth_col, trend_check, rank_check, sort_metric, how_many,
                                       ner_dict, ner_filter, df, top_data, metric_df)
    dimensions = ner_dict.get("dimension", [])
    formatted_dims = [d.replace("_", " ") for d in dimensions]
    str_q = f"The following groups of data are structured around {', '.join(formatted_dims)}. \n" + str_q if formatted_dims else str_q
    level_cols = [c for c in df.columns if c in support_df['original'].unique().tolist()]
    combination_num = len(df[level_cols].drop_duplicates())
    if combination_num>threshold:
        main_metric = ner_filter.get('main_metric', ['roi'])[0]
        tactics_col = 'tactics'
        if ner_filter.get('data', 'bi')=='br':
            target_cols = ['business_driver', 'business_driver_detail', 'tactics']
            existing = [c for c in target_cols if c in df.columns]
            for col in reversed(existing):
                if not df[col].isna().all():
                    tactics_col = col
        _, detail_analysis = compare_country_by_group_mwu(df, ordered_dims, main_metric, tactics_col = tactics_col)
        str_q = 'LINKEDIN no table readout. ' + str_q
        str_q = str_q + "\n\n" + detail_analysis
    # if intention=='contribution':
    #     group_cols = support_df[support_df['contribution_agg']=='Y']['mapped'].unique().tolist()
    #     sum_cols = [c for c in df.columns if "contribution" in c.lower() and "% change" not in c.lower()]
    #     df_conv = df.copy()
    #     for c in sum_cols:
    #         df_conv[c] = pd.to_numeric(df_conv[c].astype(str).str.replace("%", "", regex=False),errors="coerce")
    #     group_df = df_conv.groupby(list(group_cols), as_index=False)[sum_cols].sum(min_count=1)
    #     for c in sum_cols:
    #         group_df[c] = group_df[c].round(2).astype(str) + "%"
    #     _, detail_agg_str, _, _ = table_to_text(pd.DataFrame(), group_df, take_head=False)
    #     detail_table_str = "Aggregated Contribution: \n" + detail_agg_str + "\n\nDetailed Contribution: \n" + detail_table_str
    #     agg_table_str = ''
    halo_level = support_df.loc[support_df['mapped'].isna(), 'original'].iloc[0]
    if halo_level in data.columns:
        data = data.drop(columns = [halo_level])
        ner_dict['dimension'] = [o for o in ordered_dims if o!=halo_level]

    return str_q, data, ner_dict


def load_genome_principle_client(client_code, model_group_id):
    file_path = get_data_path('client_principle', 'client_principle.csv')
    if os.path.exists(file_path):
        df = pd.read_csv(file_path, low_memory=False)
        return df
    else:
        return pd.DataFrame()

def load_genome_principle_industry(client_code, model_group_id):
    file_path = get_data_path('industry_principle', 'industry_principle.csv')
    if os.path.exists(file_path):
        df = pd.read_csv(file_path, low_memory=False)
        return df
    else:
        return pd.DataFrame()

def load_genome_principle_mapping(client_code, model_group_id):
    file_path = get_data_path('genome_principle_mapping', 'genome_principle_mapping.csv')
    if os.path.exists(file_path):
        df = pd.read_csv(file_path, low_memory=False)
        return df
    else:
        return pd.DataFrame()


# def principle_df_to_text(
#     general_df: pd.DataFrame,
#     deep_df: pd.DataFrame,
#     merge_dim_cols: list[str],
#     target_cols: list[str],
#     group_col: str = "marketing_group",
#     channel_col: str = "marketing_channel",
#     buy_col: str = "marketing_buy",) -> str:
#     """
#     Convert principle dataframes to a readable string.
#
#     - general_df: aggregated at marketing_group level (may include merge_dim_cols)
#     - deep_df: detailed rows (marketing_group + marketing_channel [+ marketing_buy])
#     - merge_dim_cols: e.g. ["country"] or ["country","region"]
#     - target_cols: e.g. ["marketing_revenue_roi", "marketing_revenue_roi_index_brand"]
#                   or ["marketing_share_spend_brand"]
#     """
#
#     def _fmt_val(col: str, v):
#         if pd.isna(v):
#             return "NA"
#         try:
#             v = float(v)
#         except Exception:
#             return str(v)
#         # if it's spend share, scale to percent
#         if "share_spend" in col.lower():
#             v *= 100.0
#         return f"{v:.2f}"
#
#     def _dims_prefix(dim_vals: dict) -> str:
#         # "For USA" or "For USA, North America"
#         if not dim_vals:
#             return ""
#         parts = [str(dim_vals[c]) for c in merge_dim_cols if c in dim_vals]
#         return f"For {', '.join(parts)}, "
#
#     def _one_section(df: pd.DataFrame, header_prefix: str, is_deep: bool) -> str:
#         lines = []
#         if df.empty:
#             return ""
#
#         if merge_dim_cols:
#             grouped = df.groupby(merge_dim_cols, dropna=False, sort=False)
#         else:
#             grouped = [(None, df)]
#
#         for key, g in grouped:
#             # build prefix
#             dim_vals = {}
#             if merge_dim_cols:
#                 if len(merge_dim_cols) == 1:
#                     dim_vals[merge_dim_cols[0]] = key
#                 else:
#                     for c, kv in zip(merge_dim_cols, key):
#                         dim_vals[c] = kv
#
#             prefix = _dims_prefix(dim_vals)
#
#             if not is_deep:
#                 # General: one line per marketing_group
#                 lines.append(f"{header_prefix}{prefix}principle summary by marketing group:")
#                 for _, r in g.iterrows():
#                     mg = r.get(group_col, "Unknown")
#                     kvs = []
#                     for c in target_cols:
#                         if c in g.columns:
#                             kvs.append(f"{c}={_fmt_val(c, r.get(c))}")
#                     if kvs:
#                         lines.append(f"- {mg}: " + ", ".join(kvs))
#                 lines.append("")  # spacer
#             else:
#                 # Deep: list detailed rows; include buy if present/non-null
#                 lines.append(f"{header_prefix}{prefix}deep dive details:")
#                 # optional: keep a stable order
#                 sort_cols = [c for c in [group_col, channel_col, buy_col] if c in g.columns]
#                 if sort_cols:
#                     g2 = g.sort_values(sort_cols, kind="stable")
#                 else:
#                     g2 = g
#
#                 for _, r in g2.iterrows():
#                     mg = r.get(group_col, "Unknown")
#                     ch = r.get(channel_col, "Unknown")
#                     buy = r.get(buy_col, None) if buy_col in g2.columns else None
#
#                     kvs = []
#                     for c in target_cols:
#                         if c in g2.columns:
#                             kvs.append(f"{c}={_fmt_val(c, r.get(c))}")
#
#                     if buy is not None and not pd.isna(buy) and str(buy).strip():
#                         lines.append(f"- {mg} / {ch} / {buy}: " + ", ".join(kvs))
#                     else:
#                         lines.append(f"- {mg} / {ch}: " + ", ".join(kvs))
#                 lines.append("")  # spacer
#
#         return "\n".join(lines).strip()
#
#     parts = []
#
#     # General chunk
#     general_txt = _one_section(general_df, header_prefix="", is_deep=False)
#     if general_txt:
#         parts.append(general_txt)
#
#     # Deep chunk (separate section)
#     deep_txt = _one_section(deep_df, header_prefix="More detail — ", is_deep=True)
#     if deep_txt:
#         parts.append(deep_txt)
#
#     return "\n\n".join([p for p in parts if p]).strip()


def build_ind_filter(raw_clt: pd.DataFrame, raw_ind: pd.DataFrame, min_rows: int = 10, min_clients: int = 2,):
    """
    Build an industry peer filter with progressive widening.

    Tries in order:
      1) category + country
      2) industry + country
      3) industry + region

    Returns:
      ind_filter: filtered raw_ind
      compare: whether comparison is meaningful
      filter_cols: columns used for the chosen filter (helpful downstream)
    """

    # Precompute client-side values once
    clt_vals = {
        "category": raw_clt["category"].dropna().unique().tolist() if "category" in raw_clt else [],
        "country": raw_clt["country"].dropna().unique().tolist() if "country" in raw_clt else [],
        "industry": raw_clt["industry"].dropna().unique().tolist() if "industry" in raw_clt else [],
        "region": raw_clt["region"].dropna().unique().tolist() if "region" in raw_clt else [],
    }

    # Each strategy: (filter_cols, [(raw_ind_col, raw_clt_key_for_values), ...])
    strategies = [(["category", "country"], [("category", "category"), ("country", "country")]),
        (["industry", "country"], [("industry", "industry"), ("country", "country")]),
        (["industry", "region"], [("industry", "industry"), ("region", "region")]),]

    def apply_strategy(mappings):
        mask = pd.Series(True, index=raw_ind.index)
        for ind_col, clt_key in mappings:
            # If required col missing or no client values, this strategy yields empty
            if ind_col not in raw_ind.columns:
                return pd.Series(False, index=raw_ind.index)
            wanted = clt_vals.get(clt_key, [])
            if not wanted:
                return pd.Series(False, index=raw_ind.index)
            mask &= raw_ind[ind_col].isin(wanted)
        return mask

    last_filter = raw_ind.iloc[0:0]
    last_cols = strategies[-1][0]

    for filter_cols, mappings in strategies:
        mask = apply_strategy(mappings)
        ind_filter = raw_ind.loc[mask].copy()
        ind_clients = ind_filter["clientcode"].dropna().unique().tolist() if "clientcode" in ind_filter else []

        last_filter, last_cols = ind_filter, filter_cols

        # same logic as your original: compare=False if too few rows OR only 1 client
        if len(ind_filter) > min_rows and len(ind_clients) >= min_clients:
            return ind_filter, True, filter_cols

    # If all strategies fail
    return last_filter, False, last_cols

def _clean_roi(values, require_positive = True):
    """Return finite ROI values; optionally require >0 (needed for lognormal)."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if require_positive:
        v = v[v > 0]
    return v


def empirical_percentile(vals, x, threshold=10):
    vals = np.asarray(vals)
    if len(vals) < threshold:
        return np.nan
    # proportion <= x
    sorted_vals = np.sort(vals)
    pct = np.searchsorted(sorted_vals, x, side="right") / len(sorted_vals)
    return round(pct * 100, 2)


def principle_table_to_text(clt_roi, clt_roi_channel, merge_dim_cols, target_cols, ind_roi_values= None, group_to_values = None, filter_cols = None):
    """
    Generate clean, instructive narrative text for LLM prompts.

    Output style (example):
      ## Client overall marketing ROI vs industry
      - For country=US, LinkedIn's overall marketing ROI is 2.24, and it falls within/below/above (P50.00 Q2) the typical marketing revenue ROI range of [Q1 value] to [Q3 value] for the [industry name] industry.

      ## Marketing group ROI index vs industry
      - For country=US, LinkedIn's ROI index in Digital Audio is 9.54, which is less efficient/of average efficiency/more efficient than (P6.67 Q1) industry peers with ROI index ranging from [Q1 value] to [Q3 value].

    Notes:
      - Skips any NULL values (won't print "N/A").
      - Assumes pct_vs_industry is already 0–100 (percentage).
      - Uses merge_dim_cols to build the "For ..." context; if empty, omits it.
      - If overall ROI <Q1, then below, if >=Q1 <= Q3 then between, otherwise above
      - If channel ROI index <90 then less efficient, if >=90 and <=110 then of average efficiency, otherwise more efficient
    """

    # -------------------------
    # Helpers
    # -------------------------
    def fmt_num(v, decimals=2):
        if v is None or pd.isna(v):
            return None
        if isinstance(v, (float, np.floating, int, np.integer)):
            return round(float(v), decimals)
        return None

    def fmt_spend_pct(v, decimals=2):
        """
        Spend share stored as 0–1 -> display as 0–100%.
        """
        v = fmt_num(v, 10)
        if v is None:
            return None
        v = round(v * 100.0, decimals)
        return f"{v:.{decimals}f}%"

    def dim_phrase(row, cols, df_cols):
        parts = []
        for c in cols or []:
            if c in df_cols and pd.notna(row.get(c)):
                parts.append(f"{c}={row.get(c)}")
        return ", ".join(parts)

    def q1_q3_from(values):
        if values is None:
            return (None, None)
        arr = np.asarray(values)
        arr = arr[~pd.isna(arr)]
        if arr.size == 0:
            return (None, None)
        return (float(np.percentile(arr, 25)), float(np.percentile(arr, 75)))

    def percentile_phrase(pct):
        pct = fmt_num(pct, 2)
        if pct is None:
            return None
        pct = max(0.0, min(100.0, float(pct)))
        return f"ahead of {int(round(pct))}% of industry peers"

    def pick_metric(candidates, df_cols, target_cols_):
        for c in candidates:
            if c in target_cols_ and c in df_cols:
                return c
        return None

    def metric_label(col):
        mapping = {
            "marketing_revenue_roi_brand": "overall marketing ROI",
            "marketing_revenue_roi_index_brand": "ROI index",
            "marketing_share_spend_brand": "marketing spend share",
            "marketing_share_spend_overall": "overall marketing spend share",
        }
        return mapping.get(col, col.replace("_", " "))

    def metric_formatter(col):
        if col in {"marketing_share_spend_brand", "marketing_share_spend_overall"}:
            return fmt_spend_pct
        return lambda v: (f"{fmt_num(v, 2):.2f}" if fmt_num(v, 2) is not None else None)

    def append_spend_suffix(row, df_cols, target_cols_, main_metric):
        """
        Append spend context only if spend cols are requested AND
        the narrated metric is NOT already a spend metric.
        Spend values formatted as (value * 100)%.
        """
        if main_metric in {"marketing_share_spend_brand", "marketing_share_spend_overall"}:
            return ""

        parts = []

        if "marketing_share_spend_brand" in target_cols_ and "marketing_share_spend_brand" in df_cols:
            v = fmt_spend_pct(row.get("marketing_share_spend_brand"), 2)
            if v is not None:
                parts.append(f"brand spend share {v}")

        if "marketing_share_spend_overall" in target_cols_ and "marketing_share_spend_overall" in df_cols:
            v = fmt_spend_pct(row.get("marketing_share_spend_overall"), 2)
            if v is not None:
                parts.append(f"overall spend share {v}")

        if not parts:
            return ""

        return ", with " + " and ".join(parts)

    def _safe_filter_key(row, df_cols):
        """
        Build the grouping key for benchmark lookup.
        Returns None if filter_cols not provided or missing.
        """
        if not filter_cols:
            return None
        for c in filter_cols:
            if c not in df_cols:
                return None
        return tuple(row.get(c) for c in filter_cols)

    def _lookup_overall_values(row, df_cols):
        """
        Resolve benchmark values for overall section.
        Supports:
          - ind_roi_values as list/array
          - ind_roi_values as dict keyed by filter tuple
        """
        if ind_roi_values is None:
            return None

        # New grouped behavior
        if isinstance(ind_roi_values, dict):
            key = _safe_filter_key(row, df_cols)
            if key is None:
                return None
            return ind_roi_values.get(key)

        # Old behavior: list/array
        return ind_roi_values

    def _lookup_group_values(row, df_cols, grp_str):
        """
        Resolve benchmark values for group section.
        Supports:
          - group_to_values as {grp: values}
          - group_to_values as {(filter_tuple + (grp,)): values}
        """
        if group_to_values is None:
            return None

        # New grouped behavior
        if isinstance(group_to_values, dict) and filter_cols:
            key = _safe_filter_key(row, df_cols)
            if key is not None:
                return group_to_values.get(tuple(key) + (grp_str,))

        # Old behavior: group only
        if isinstance(group_to_values, dict):
            return group_to_values.get(grp_str)

        return None

    # -------------------------
    # Decide which metrics to narrate
    # -------------------------
    overall_metric = None
    if clt_roi is not None and not clt_roi.empty:
        overall_metric = pick_metric(
            candidates=[
                "marketing_revenue_roi_brand",
                "marketing_share_spend_overall",
                "marketing_share_spend_brand",
            ],
            df_cols=clt_roi.columns,
            target_cols_=target_cols,
        )

    group_metric = None
    if clt_roi_channel is not None and not clt_roi_channel.empty:
        group_metric = pick_metric(
            candidates=[
                "marketing_revenue_roi_index_brand",
                "marketing_revenue_roi_brand",
                "marketing_share_spend_brand",
                "marketing_share_spend_overall",
            ],
            df_cols=clt_roi_channel.columns,
            target_cols_=target_cols,
        )

    lines = []

    # -------------------------
    # Overall narrative
    # -------------------------
    if clt_roi is not None and not clt_roi.empty and overall_metric is not None:
        lines.append(f"## Client {metric_label(overall_metric)} vs industry")
        fmt_overall = metric_formatter(overall_metric)

        for _, r in clt_roi.iterrows():
            dims = dim_phrase(r, merge_dim_cols, clt_roi.columns)
            company = r.get("company")
            company_str = str(company) if pd.notna(company) else "The client"

            val_str = fmt_overall(r.get(overall_metric))
            if val_str is None:
                continue

            pct_text = (
                percentile_phrase(r.get("pct_vs_industry"))
                if "pct_vs_industry" in target_cols and "pct_vs_industry" in clt_roi.columns
                else None
            )
            pct_part = f", ranking {pct_text}" if pct_text else ""

            # Benchmarks (Q1–Q3) - now filter-aware
            values = _lookup_overall_values(r, clt_roi.columns)
            ind_q1, ind_q3 = q1_q3_from(values)

            range_part = ""
            if ind_q1 is not None and ind_q3 is not None:
                v_num = fmt_num(r.get(overall_metric), 10)
                if v_num is not None:
                    if v_num < ind_q1:
                        position = "below"
                    elif v_num <= ind_q3:
                        position = "within"
                    else:
                        position = "above"

                    if overall_metric in {"marketing_share_spend_brand", "marketing_share_spend_overall"}:
                        q1_str, q3_str = fmt_spend_pct(ind_q1, 2), fmt_spend_pct(ind_q3, 2)
                    else:
                        q1_str, q3_str = f"{fmt_num(ind_q1, 2):.2f}", f"{fmt_num(ind_q3, 2):.2f}"

                    range_part = (
                        f", and it is {position} the typical industry range of {q1_str} to {q3_str} (Q1–Q3)"
                    )

            spend_suffix = append_spend_suffix(r, clt_roi.columns, target_cols, overall_metric)
            lead = f"For {dims}, " if dims else ""

            lines.append(
                f"- {lead}{company_str}'s {metric_label(overall_metric)} is {val_str}"
                f"{pct_part}{range_part}{spend_suffix}."
            )

    # -------------------------
    # Group narrative
    # -------------------------
    if clt_roi_channel is not None and not clt_roi_channel.empty and group_metric is not None:
        if lines:
            lines.append("")
        lines.append(f"## Marketing group {metric_label(group_metric)} vs industry")

        fmt_group = metric_formatter(group_metric)

        for _, r in clt_roi_channel.iterrows():
            if "marketing_group" not in target_cols or "marketing_group" not in clt_roi_channel.columns:
                continue

            grp = r.get("marketing_group")
            if pd.isna(grp):
                continue
            grp_str = str(grp)

            dims = dim_phrase(r, merge_dim_cols, clt_roi_channel.columns)
            company = r.get("company")
            company_str = str(company) if pd.notna(company) else "The client"

            val_str = fmt_group(r.get(group_metric))
            if val_str is None:
                continue

            pct_text = (
                percentile_phrase(r.get("pct_vs_industry"))
                if "pct_vs_industry" in target_cols and "pct_vs_industry" in clt_roi_channel.columns
                else None
            )
            pct_part = f", ranking {pct_text}" if pct_text else ""

            # Benchmarks (Q1–Q3) - now filter-aware (filter tuple + marketing_group)
            values = _lookup_group_values(r, clt_roi_channel.columns, grp_str)
            ch_q1, ch_q3 = q1_q3_from(values)

            range_part = ""
            if ch_q1 is not None and ch_q3 is not None:
                v_num = fmt_num(r.get(group_metric), 10)
                if v_num is not None:
                    if v_num < ch_q1:
                        position = "below"
                    elif v_num <= ch_q3:
                        position = "within"
                    else:
                        position = "above"

                    if group_metric in {"marketing_share_spend_brand", "marketing_share_spend_overall"}:
                        q1_str, q3_str = fmt_spend_pct(ch_q1, 2), fmt_spend_pct(ch_q3, 2)
                    else:
                        q1_str, q3_str = f"{fmt_num(ch_q1, 2):.2f}", f"{fmt_num(ch_q3, 2):.2f}"

                    range_part = (
                        f", and it is {position} the typical industry range of {q1_str} to {q3_str} (Q1–Q3)"
                    )

            spend_suffix = append_spend_suffix(r, clt_roi_channel.columns, target_cols, group_metric)
            lead = f"For {dims}, " if dims else ""

            lines.append(
                f"- {lead}{company_str}'s {metric_label(group_metric)} in {grp_str} is {val_str}"
                f"{pct_part}{range_part}{spend_suffix}."
            )

    return "\n".join(lines).strip()



def genome_principle_test_new(client_code, model_group_id, data, dim_cols, intention, dive_grp = ['Paid Search', 'Paid Social', 'TV-Like Video Streaming']):
    data = data.rename(columns={'marketing_channel': 'marketing_group_key', 'media_channel': 'marketing_group_key'})
    raw_clt = load_genome_principle_client(client_code, model_group_id)
    raw_ind = load_genome_principle_industry(client_code, model_group_id)
    mapping = load_genome_principle_mapping(client_code, model_group_id)
    if raw_clt.empty or raw_ind.empty:
        return ''
    start, end = (data['start']//100).min(), (data['end']//100).max()
    years = list(range(start, end + 1))
    cols = ['marketing_group', 'marketing_channel', 'marketing_buy']  # columns you want to change
    raw_clt[cols] = raw_clt[cols].apply(lambda s: s.str.replace('_', ' ', regex=False))
    raw_ind[cols] = raw_ind[cols].apply(lambda s: s.str.replace('_', ' ', regex=False))
    ind_df, compare, filter_cols = build_ind_filter(raw_clt, raw_ind)
    keep_cols = ['clientcode', 'company', 'region', 'country', 'master_brand', 'industry', 'category',
                 'annual_year', 'marketing_group', 'marketing_channel', 'marketing_buy',
                 'marketing_spend_usd', 'marketing_revenue_usd', 'marketing_revenue_roi',
                 'marketing_spend_usd_brand', 'marketing_revenue_roi_brand',
                 'marketing_revenue_roi_index_brand', 'marketing_spend_usd_group',
                 'marketing_revenue_roi_group', 'marketing_revenue_roi_index_group',
                 'marketing_share_spend_brand']
    clt_df = raw_clt[keep_cols].drop_duplicates()
    ind_df = ind_df[keep_cols].drop_duplicates()
    if intention in ['roi', 'margin roi', 'performance']:
        clt_value_col = ['marketing_revenue_roi_brand', 'marketing_share_spend_overall']
        channel_value_col = ['marketing_revenue_roi_index_brand', 'marketing_share_spend_brand']
    elif intention == 'spending':
        clt_value_col = ['marketing_share_spend_overall', 'marketing_share_spend_brand']
        channel_value_col = clt_value_col
    else:
        return ''
    group_cols = ['country', 'master_brand', 'company', 'industry', 'category', 'annual_year']
    spend_map = (ind_df[(ind_df['marketing_channel'].isnull()) &(ind_df['marketing_buy'].isnull())]
        .groupby(group_cols)['marketing_share_spend_brand'].sum())
    ind_df["marketing_share_spend_overall"] = ind_df.set_index(group_cols).index.map(spend_map)
    clt_df["marketing_share_spend_overall"] = clt_df.set_index(group_cols).index.map(spend_map)
    # -------------------------
    # Build client tables first (raw)
    # -------------------------
    clt_roi = (
        clt_df[clt_df["annual_year"].isin(years)][group_cols + clt_value_col]
        .drop_duplicates(subset=group_cols)
        .copy()
    )

    clt_roi_channel = (
        clt_df[(clt_df["marketing_channel"].isnull()) & (clt_df["marketing_buy"].isnull())][
            clt_df["annual_year"].isin(years)][
            group_cols + ["marketing_group"] + channel_value_col
            ]
        .drop_duplicates(subset=group_cols + ["marketing_group"])
        .copy()
    )

    # -------------------------
    # Prepare mapping dicts (so we can align keys)
    # -------------------------
    mapping_dict = {}
    mapping_dict_bi = {}

    if not mapping.empty:
        mapping_dict = {
            col: dict(zip(g["principle_value"].str.lower(), g["display_value"]))
            for col, g in mapping.groupby("column")
        }
        mapping_dict_bi = {
            col: dict(zip(g["bibr_value"].str.lower(), g["display_value"]))
            for col, g in mapping.groupby("column")
        }

    def _apply_map_to_cols(df_, cols_, mp_dict):
        """
        Apply principle mapping to selected columns only.
        Matches what you do later on principle_tmp_* so dict keys will align.
        """
        if df_ is None or df_.empty or not mp_dict:
            return df_
        for c in cols_:
            if c in df_.columns and c in mp_dict:
                # mimic: .str.lower().replace(mp)
                df_[c] = df_[c].astype(str).str.lower().replace(mp_dict[c])
        return df_

    # Align benchmark keys with mapped table values:
    # Apply the SAME mapping to the KEY columns used for grouping / lookup
    key_cols_for_map = list(set(filter_cols + ["marketing_group"]))
    ind_df_for_map = ind_df.copy()
    clt_roi_for_map = clt_roi.copy()
    clt_roi_channel_for_map = clt_roi_channel.copy()

    ind_df_for_map = _apply_map_to_cols(ind_df_for_map, key_cols_for_map, mapping_dict)
    clt_roi_for_map = _apply_map_to_cols(clt_roi_for_map, key_cols_for_map, mapping_dict)
    clt_roi_channel_for_map = _apply_map_to_cols(clt_roi_channel_for_map, key_cols_for_map, mapping_dict)

    # -------------------------
    # Build benchmark distributions (US compares to US, etc.) using MAPPED keys
    # -------------------------
    def _filter_key(row, fcols):
        return tuple(row[c] for c in fcols)

    if compare:
        # IMPORTANT: include filter_cols in frame before groupby
        ind_roi_df = ind_df_for_map[group_cols + [clt_value_col[0]]].drop_duplicates(subset=group_cols)

        ind_roi_values_map = (
            ind_roi_df.groupby(filter_cols)[clt_value_col[0]]
            .apply(lambda s: np.sort(s.dropna().values))
            .to_dict()
        )

        def _pct_overall(row):
            key = _filter_key(row, filter_cols)
            x = row[clt_value_col[0]]
            values = ind_roi_values_map.get(key)

            if values is None or len(values) < 5 or pd.isna(x):
                return np.nan

            return empirical_percentile(values, x)

        # compute pct using mapped-key version of client table to match dict keys
        clt_roi["pct_vs_industry"] = clt_roi_for_map.apply(_pct_overall, axis=1).values
    else:
        clt_roi["pct_vs_industry"] = None
        ind_roi_values_map = None

    if compare:
        ind_roi_channel_df = ind_df_for_map[
            (ind_df_for_map["marketing_channel"].isnull()) & (ind_df_for_map["marketing_buy"].isnull())
            ][group_cols + ["marketing_group", channel_value_col[0]]].drop_duplicates(
            subset=group_cols + ["marketing_group"]
        )

        group_to_values_map = (
            ind_roi_channel_df.groupby(filter_cols + ["marketing_group"])[channel_value_col[0]]
            .apply(lambda s: np.sort(s.dropna().values))
            .to_dict()
        )

        def _pct_channel(row):
            key = tuple(row[c] for c in filter_cols) + (row["marketing_group"],)
            x = row[channel_value_col[0]]
            values = group_to_values_map.get(key)

            if values is None or len(values) < 5 or pd.isna(x):
                return np.nan

            return empirical_percentile(values, x)

        clt_roi_channel["pct_vs_industry"] = clt_roi_channel_for_map.apply(_pct_channel, axis=1).values
    else:
        clt_roi_channel["pct_vs_industry"] = None
        group_to_values_map = None

    # -------------------------
    # Build merge keys & mapped tables for text (your original behavior)
    # -------------------------
    merge_dim_cols = [d for d in dim_cols if d in clt_roi_channel.columns]
    merge_col = merge_dim_cols + ["marketing_group_key"]
    df = data[merge_col].drop_duplicates()

    principle_tmp_channel = clt_roi_channel.copy()
    principle_tmp_channel["marketing_group_key"] = principle_tmp_channel["marketing_group"].str.lower()
    principle_tmp_overall = clt_roi.copy()

    if mapping_dict:
        for col, mp in mapping_dict.items():
            if col in principle_tmp_channel.columns:
                principle_tmp_channel[col] = principle_tmp_channel[col].astype(str).str.lower().replace(mp)
            if col in principle_tmp_overall.columns:
                principle_tmp_overall[col] = principle_tmp_overall[col].astype(str).str.lower().replace(mp)

    if mapping_dict_bi:
        for col, mp in mapping_dict_bi.items():
            if col in df.columns:
                df[col] = df[col].astype(str).str.lower().replace(mp)

    if "country" in merge_col and "country" in df.columns and (df["country"] == "overall").all():
        merge_col = [c for c in merge_col if c != "country"]
        merge_dim_col = [c for c in merge_dim_cols if c != "country"]
    else:
        merge_dim_col = merge_dim_cols

    principle_df_channel = principle_tmp_channel.merge(df[merge_col].drop_duplicates(), how="inner", on=merge_col)
    principle_df_overall = (
        principle_tmp_overall.merge(df[merge_dim_col].drop_duplicates(), how="inner", on=merge_dim_col)
        if merge_dim_col
        else principle_tmp_overall
    )

    if intention in ["roi", "margin roi", "performance"]:
        target_cols = [
            "marketing_group",
            "marketing_revenue_roi_index_brand",
            "marketing_revenue_roi_brand",
            "pct_vs_industry",
            "marketing_share_spend_brand",
        ]
        principle_df_overall = principle_df_overall.rename(
            columns={"marketing_share_spend_overall": "marketing_share_spend_brand"})
    elif intention == "spending":
        target_cols = ["marketing_group", "marketing_share_spend_brand", "marketing_share_spend_overall",
                       "pct_vs_industry"]
        principle_df_overall = principle_df_overall.drop(columns=["marketing_share_spend_brand"])
    else:
        return ""

    group_cols_filtered = [
        col
        for col in group_cols
        if col in principle_df_channel.columns and principle_df_channel[col].nunique(dropna=True) > 1
    ]
    merge_dim_cols = list(dict.fromkeys(group_cols_filtered + merge_dim_cols))

    text = principle_table_to_text(
        principle_df_overall,
        principle_df_channel,
        merge_dim_cols,
        target_cols,
        ind_roi_values_map,
        group_to_values_map,
        filter_cols,
    )
    return text


def build_metric_lookup(cfg: pd.DataFrame) -> dict:
    """
    Build lookup: {search_key: row} where search_key can be either metric name or alias.
    Both metric and alias are registered as separate entries pointing to the same row.
    Longer keys take priority during matching (handled in find_metric_config).
    """
    lookup = {}
    for _, row in cfg.iterrows():
        metric_key = str(row["metric"]).strip()
        lookup[metric_key] = row
        if pd.notna(row.get("alias")) and str(row["alias"]).strip():
            alias_key = str(row["alias"]).strip()
            lookup[alias_key] = row
    return lookup


def find_metric_config(col: str, lookup: dict):
    """
    Match column name against metric names and aliases.
    Sort keys by length descending to ensure longest match wins
    (e.g. 'cost per kpi' matched before 'cost per').
    """
    for key in sorted(lookup.keys(), key=len, reverse=True):
        if key in col.lower():
            return key, lookup[key]
    return None, None


def format_value(x, unit_sign: str, keep_digit: int, use_comma: bool) -> str:
    if pd.isna(x):
        return x
    try:
        val = float(x)
    except (ValueError, TypeError):
        return x

    fmt = f":,.{keep_digit}f" if use_comma else f":.{keep_digit}f"
    formatted = f"{val:{fmt.lstrip(':')}}"

    if unit_sign == "$":
        return f"${formatted}"
    elif unit_sign == "%":
        return f"{formatted}%"
    else:
        return formatted


def format_pivot_by_config(df: pd.DataFrame, period_suffixes, to_mm_metrics, metric_col_idx, lookup: dict) -> pd.DataFrame:
    df = df.copy()
    metric_cols = list(df.columns[metric_col_idx:])

    rename_map = {}

    for col in metric_cols:
        is_period_col = any(suffix in col for suffix in period_suffixes)
        metric_key, cfg_row = find_metric_config(col, lookup)

        if cfg_row is None:
            continue

        unit_sign = str(cfg_row["unit_sign"]).strip() if pd.notna(cfg_row["unit_sign"]) else ""
        keep_digit = int(cfg_row["keep_digit"])
        metric_name = str(cfg_row["metric"]).strip()

        if is_period_col:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].apply(lambda x: f"{x:.{keep_digit}f}%" if pd.notna(x) else x)
            continue

        if metric_name in to_mm_metrics:
            df[col] = pd.to_numeric(df[col], errors="coerce") / 1_000_000
            new_col = col + " (MM)"
            rename_map[col] = new_col
            col = new_col
            df.rename(columns={list(rename_map.keys())[-1]: col}, inplace=True)

        use_comma = unit_sign != "%"
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].apply(lambda x, u=unit_sign, k=keep_digit, c=use_comma: format_value(x, u, k, c))

    return df


def add_unit_sign(data, metric_col_idx = 0, to_mm_metrics = None, period_suffixes = ["YOY", "QOQ", "HOH"], client_code=None, model_group_id=None):
    if to_mm_metrics is None:
        to_mm_metrics = []
    if period_suffixes is None:
        period_suffixes = ["YOY", "QOQ", "HOH"]
    metric_config = load_unit_sign_config()
    metric_lookup = build_metric_lookup(metric_config)
    data = format_pivot_by_config(data, period_suffixes, to_mm_metrics, metric_col_idx, metric_lookup)
    data = data.loc[:, ~data.apply(lambda col: col.isna().all() or (col == "").all())]
    return data




