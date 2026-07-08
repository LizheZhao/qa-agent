# Apply the filter to DataFrame, generate 2 dict + 3 df

import os
import re
import json
import pandas as pd
import numpy as np

import logging

logger = logging.getLogger(__name__)


### process ner filter for readout ###


def _get_intention_answer_dict(dimension: str, answer_dict: dict, intention_list: list) -> dict:
    res_dict = {}
    if isinstance(answer_dict, list) and isinstance(answer_dict[0], dict):
        answer_dict = answer_dict[0]
    for ak, av in answer_dict.items():
        if ak == "intention":
            if isinstance(av, str) and av in intention_list:
                res_dict[ak] = [av]
            elif isinstance(av, list) and any(x for x in av if x in intention_list):
                res_dict[ak] = [[x for x in av if x in intention_list][0]]
        # allow multiple intentions
        # else:
        #     res_dict['intention'] = [x for x in intention_list if x in intention_list]
        #     break
    # set default intention to performance
    if "intention" not in res_dict:
        res_dict["intention"] = ["performance"]
    return res_dict


def _get_time_answer_dict(df: pd.DataFrame, fields: list, answer_dict: dict) -> dict:
    # answer_dict = ast.literal_eval(answer)
    res_dict = {}
    t = answer_dict.get('time', 'irrelevant')
    p = answer_dict.get('period_type', 'irrelevant')

    p_priority = ['fiscal year', 'fiscal half', 'fiscal quarter', 'fiscal month']

    # check for mixed time and period type
    if (isinstance(p, list) and len([x for x in p if x in df['period_type'].dropna().unique()]) > 1) or (isinstance(t, list) and len(df[df['time'].isin(t)]['period_type'].dropna().unique()) > 1):
        p_priority = ['fiscal quarter', 'fiscal half', 'fiscal year']
    if isinstance(t, list):
        p_priority = [x for x in p_priority if x in df[df['time'].isin(t)]['period_type'].dropna().unique()]
        if len(p_priority) == 0:
            p_priority = ['fiscal year', 'fiscal half', 'fiscal quarter', 'fiscal month']

    # type check
    if isinstance(p, str) and p not in df['period_type'].unique():
        p = 'irrelevant'
    elif isinstance(p, str) and p in df['period_type'].unique():
        p = [p]
    elif isinstance(p, list) and np.sum([True if x in df['period_type'].unique() else False for x in p]) >= 1:
        p = [[x for x in p_priority if x in p][0]]
    elif isinstance(p, list) and np.sum([True if x in df['period_type'].unique() else False for x in p]) < 1:
        p = 'irrelevant'
    else:
        p = 'irrelevant'

    if isinstance(t, str):
        t = [x for x in [t] if x in df['time'].unique()]
    elif isinstance(t, list):
        t = [x for x in t if x in df['time'].unique()]
    if len(t) < 1:
        t = 'irrelevant'

    if t == 'irrelevant' and p == 'irrelevant':
        p = [x for x in p_priority if x in df['period_type'].unique()][0]
        t = sorted(list(df[df['period_type'].isin([p])]['time'].unique()))[-2:]
    elif t == 'irrelevant':
        if p != ['fiscal year']:
            t = sorted(list(df[df['period_type'].isin(p)]['time'].unique()))
        else:
            # return last two fiscal year
            t = sorted(list(df[df['period_type'].isin(p)]['time'].unique()))[-2:]
    elif p == 'irrelevant':
        p = df[df['time'].isin(t)]['period_type'].value_counts().index[0]

    if not isinstance(p, list):
        p = [p]
    if not isinstance(t, list):
        t = [t]

    def parse_fiscal_month(fiscal_str):
        parts = fiscal_str.split()
        month = int(parts[2])
        year = int(parts[3])
        return (year, month)

    # return last half and yag or last 8 quarters or last 12 months
    time_list = sorted(list(df[df['period_type'].isin(p)]['time'].unique()))
    if p == ['fiscal half']:
        time_list = sorted(list(df[df['period_type'].isin(p)]['time'].unique()))
        t = sorted([x for x in time_list if x in t], key=parse_fiscal_month)[-1: -4: -2]
    if p == ['fiscal quarter']:
        time_list = sorted(list(df[df['period_type'] == 'fiscal quarter']['time'].unique()))
        t = sorted([x for x in time_list if x in t], key=parse_fiscal_month)[-8:]
    if p == ['fiscal month']:
        t = sorted([x for x in time_list if x in t], key=parse_fiscal_month)[-12:]
    res_dict['time'] = t if isinstance(t, list) else [t]
    res_dict['period_type'] = p if isinstance(p, list) else [p]
    return res_dict


def _get_answer_dict(df: pd.DataFrame, dimension: str, fields: list, answer_dict: dict, data: str = 'bi') -> dict:
    # answer_dict = ast.literal_eval(answer)
    res_dict = {}
    for ak, av in answer_dict.items():
        # ensure list
        if not isinstance(av, list):
            av = [av]
        if ak in df.columns and ak in fields:
            if 'irrelevant' in av:
                if 'overall' in df[ak].values:
                    res_dict[ak] = ['overall'] # hardcoded for product
            elif 'all' in av:
                valid_list = list([x for x in df[ak].dropna().unique() if x not in ['overall', "portfolio"]])  # hard coded for product\bu
                if len(valid_list) >= 1:
                    res_dict[ak] = valid_list
            elif 'overall' in av and len(av) == 1:
                res_dict[ak] = ['overall'] # hardcoded for product
            elif 'relevant' in av:
                res_dict[ak] = ['yes']
            elif ak == "business_driver":
                bd_list = av
                if "marketing" in av:
                    bd_list.extend(["marketing", "media", "non-media", "promotions"])
                elif "media" in av:
                    bd_list.extend(["media"])
                elif "base" in av:
                    bd_list.extend(["base", "other"])
                elif "specific" in av:
                    pass
                    # bd_list = set(df['activity_group'].dropna().unique()) | set(df['business_driver_detail'].dropna().unique())
                    # # get original terms
                    # spacy_bd = set()
                    # for term, filters in ner_filter.get("core_dimension", {}).items():
                    #     spacy_bd.update({x.get('org_term', term) for x in filters})
                    # # spacy_bd = ner_filter.get("core_dimension", {}).keys()
                    # valid_list = [x for x in spacy_bd if x in bd_list]
                    # bd_filter = {"business_driver": valid_list} if len(valid_list) > 0 else {}
                    # ignore_fields += ['business_driver']
                else:
                    pass
                valid_list = [x for x in set(bd_list) if x in df['business_driver'].unique()]
                if len(valid_list):
                    res_dict[ak] = valid_list
            else:
                valid_list = [v for v in av if v in df[ak].dropna().unique()]
                if len(valid_list) >= 1:
                    res_dict[ak] = valid_list
        elif ak == 'trend':
            l = [x for x in av if x in ['yes', 'no', 'na']]
            res_dict[ak] = l[0] if len(l) > 0 else 'na'
        elif ak == 'rank':
            l = [x for x in av if x in ['top', 'bottom', 'na']]
            res_dict[ak] = l[0] if len(l) > 0 else 'na'
        elif ak == 'how_many':
            try:
                res_dict[ak] = int(av[0]) if 'na' not in av else 'na'
            except:
                res_dict[ak] = 'na'
        else:
            logger.debug(f'{ak} is not a valid answer dict key with values {av}')
    return res_dict


def _apply_answer_dict(df: pd.DataFrame, answer_dict: dict, res_dict=None, ignore_fields=None) -> pd.DataFrame:
    if ignore_fields is None:
        ignore_fields = []
    if res_dict is None:
        res_dict = {}
    tmp = df.copy()
    for k, v in answer_dict.items():
        if not isinstance(v, list):
            continue
        if k not in df.columns:
            continue
        if all(value not in tmp[k].unique() for value in v):
            continue
        if k not in ignore_fields:
            if res_dict.get(k, '') == 'all':
                tmp = tmp[tmp[k].isin(v) | tmp[k].isna()]
            else:
                tmp = tmp[tmp[k].isin(v)]
    logger.debug(tmp.shape, 'after applying filter')
    return tmp.reset_index(drop=True)


def _get_valid_dict(answer_dict: dict, df: pd.DataFrame, fields: list):
    """
    process valid level 1 ner result dictionary
    """
    # answer_dict = ast.literal_eval(answer)
    ans = {}
    for k, v in answer_dict.items():
        if v in ['irrelevant', 'relevant', 'all'] or k in ['intention']:
            ans[k] = v
        elif isinstance(v, str) and k in fields:
            vl = [x for x in [v] if x in df[k].unique()]
            if len(vl) >= 1:
                ans[k] = vl
            else:
                ans[k] = 'irrelevant'
        elif isinstance(v, list) and k in fields:
            vl = [x for x in v if x in df[k].unique()]
            if len(vl) >= 1:
                ans[k] = vl
            else:
                ans[k] = 'irrelevant'
    return ans


def _postprocess_filter(ner_filter: dict, ner_dict: dict) -> tuple[dict, dict]:
    """
    postprocess ner_filter based on ner_dict and business logic,
    ner_filter: actual list of filter values
    ner_dict: 'relevant' or 'irrelevant'
    1. if media_channel is 'all' in ner_dict, 'business_driver' should be 'media' in ner_filter
    2. if retailer is 'relevant' in ner dict, 'media_channel' should be 'irrelevant'
    3. copy intention in ner_filter to ner_dict
    4. if trend is yes, include at least 2 years of data or quarter from previous year
    5. if intention is sovc, trend is no
    6. if intention is performance/roi and period_type is month, change to fiscal quarter
    7. (not applicable for other categories) if product_halo is irrelevant, map product to product halo
    """
    # postprocess logic 1
    if ner_dict.get('media_channel', '') == 'all':
        ner_filter['business_driver'] = ['media']
    # postprocess logic 2
    if ner_dict.get('retailer', '') == 'relevant':
        ner_dict['media_channel'] = 'irrelevant'
        if 'media_channel' in ner_filter.keys():
            del ner_filter['media_channel']
    # postprocess logic 3
    ner_dict['intention'] = ner_filter['intention']
    # postprocess logic 4
    p, t = ner_filter.get('period_type', []), ner_filter.get('time', [])
    if ner_filter.get('trend', 'na') == 'yes' or ner_dict.get('intention', []) == ['source of change']:
        if p == ['fiscal year'] and len(t) == 1 and re.search(r'\d{4}', t[0]):
            year = int(re.search(r'\d{4}', t[0]).group())
            ner_filter['time'].append('fiscal year ' + str(year-1))
        if p == ['fiscal quarter'] or p == ['fiscal half']:
            years = set(re.search(r'\d{4}', time_str).group() for time_str in t if re.search(r'\d{4}', time_str))
            if len(years) == 1:
                year = years.pop()
                previous_year_time_str = [time_str.replace(year, str(int(year) - 1)) for time_str in t]
                ner_filter['time'].extend(previous_year_time_str)
    # accommodate for product spend in tp
    # if ner_dict.get('intention', ['']) == ['spending'] and ner_dict.get('media_channel', '') == 'irrelevant' and ner_dict.get('custom_aggregated', ['']) == ['yes']:
    #     ner_filter['activity_group'] = ['pillars']
    # postprocess logic 5
    if ner_dict.get('intention', '') == ['source of change']:
        ner_dict['trend'] = 'no'
        ner_filter['trend'] = 'no'
    # postprocess logic 6
    if ner_dict.get('intention', '') in [['performance'], ['margin roi']] and ner_filter.get('period_type', '') == ['fiscal month']:
        ner_dict['period_type'] = ['fiscal quarter']
        ner_filter['period_type'] = ['fiscal quarter']
        # get year from time
        t = ner_filter.get('time', [])
        year = set(int(re.search(r'\d{4}', _t).group()) for _t in t if re.search(r'\d{4}', _t))
        quarter_list = []
        for y in year:
            quarter_list.extend([f'fiscal quarter 1 {y}', f'fiscal quarter 2 {y}', f'fiscal quarter 3 {y}', f'fiscal quarter 4 {y}'])
        ner_filter['time'] = quarter_list
    # postprocess logic 7
    # if ner_filter.get('data', 'br') == 'bi' and ner_filter.get('product_halo') == None:
    #     product_media_list = ['equity', 'optic white', 'total tp', 'ahw']
    #     if ner_dict.get('product', 'irrelevant') == 'all':
    #         ner_filter['product_halo'] = [x for x in product_media_list if x not in ['equity', 'ahw']]
    #         ner_filter['product'] = ['overall']
    #         ner_dict['product_halo'] = ner_filter['product_halo']
    #     elif isinstance(ner_dict.get('product'), list) and ('overall' not in ner_dict.get('product', [])):
    #         mapping = {'optic white tp': 'optic white'}
    #         ner_filter['product_halo'] = [mapping.get(x, x) for x in ner_dict.get('product') if x in product_media_list or x == 'optic white tp']
    #         ner_filter['product'] = ['overall']
    #         ner_dict['product_halo'] = ner_filter['product_halo']
    #         if len(ner_filter['product_halo']) < 1:
    #             del ner_filter['product_halo']
    return dict(ner_filter), dict(ner_dict)


def _get_level(df, level=None):
    if level is None:
        level = ['activity_group', 'measure_group', 'measure']
    l1, l2, l3 = level
    levels = []
    for i in range(len(df)):
        ag = df[l1].values[i]
        mg = df[l2].values[i]
        m = df[l3].values[i]
        if (pd.isna(m) or m == 'invalid') and (pd.isna(mg) or mg == 'invalid') and not pd.isna(ag):
            levels.append(l1)
        elif (pd.isna(m) or m == 'invalid') and (not pd.isna(mg) or mg == 'invalid') and not pd.isna(ag):
            levels.append(l2)
        elif not (pd.isna(m) or m == 'invalid') and not (pd.isna(mg) or m == 'invalid') and not pd.isna(ag):
            levels.append(l3)
        else:
            levels.append('invalid')
    df['record_level'] = levels
    return df


def _get_level_answer(df, level=None):
    if level is None:
        level = ['activity_group', 'measure_group', 'measure']
    l1, l2, l3 = level
    df_with_level = _get_level(df, level=level)
    logger.debug('record level information:')
    logger.debug(df_with_level['record_level'].value_counts())
    ag_res = df_with_level[df_with_level['record_level']==l1].reset_index(drop=True).copy()
    mg_res = df_with_level[df_with_level['record_level']==l2].reset_index(drop=True).copy()
    # mg_res = mg_res[~mg_res[l1].isin(ag_res[l1])].reset_index(drop=True).copy()
    m_res = df_with_level[df_with_level['record_level']==l3].reset_index(drop=True).copy()
    # m_res = m_res[~m_res[l1].isin(ag_res[l1])].reset_index(drop=True).copy()
    # m_res = m_res[~m_res[l1].isin(mg_res[l1])].reset_index(drop=True).copy()
    return ag_res, mg_res, m_res


def map_keyword_in_query(modelgroupId: int, df: pd.DataFrame, query: str, ner_filter: dict, ignore_fields: list) -> tuple[pd.DataFrame, dict, list]:
    res_df = df.copy()
    res_filter = dict(ner_filter)
    ag_list = res_df.activity_group.dropna().unique()
    mg_list = res_df.measure_group.dropna().unique()
    m_list = res_df.measure.dropna().unique()
    ag_filter_list, mg_filter_list, m_filter_list = [], [], []
    keyword_map = {'amazon': ['amazon', 'amz'], 'meta': ['meta', 'facebook', 'fb'],
                   'facebook': ['meta', 'facebook', 'fb'],
                   'fb': ['meta', 'facebook', 'fb'], 'tiktok': ['tiktok'], 'ttd': ['ttd', 'trade desk'],
                   'youtube': ['youtube', 'trueview'], 'google': ['google'], 'pinterest': ['pinterest'],
                   'twitter': ['twitter']}
    for key, val in keyword_map.items():
        if key in query.lower():
            for v in val:
                ag_filter_list.extend([x for x in ag_list if v in x])
                mg_filter_list.extend([x for x in mg_list if v in x])
                m_filter_list.extend([x for x in m_list if v in x])
    if len(mg_filter_list) or len(m_filter_list) or len(ag_filter_list):
        ag_filter_list += res_filter.get('activity_group', [])
        mg_filter_list += res_filter.get('measure_group', [])
        m_filter_list += res_filter.get('measure', [])
        # res_df = res_df[(res_df['measure_group'].isin(mg_filter_list)) | (res_df['measure'].isin(m_filter_list)) | (res_df['activity_group'].isin(ag_filter_list))]
        # if keyword in list match, overwrite media_channel
        # res_filter['media_channel'] = [sorted(res_df['media_channel'].dropna().unique())[0]]
        # ignore_fields += ['media_channel', 'activity_group', 'measure_group', 'measure']
    if modelgroupId == 3045:
        # softsoap not apply retailer matching
        pass
    elif ('retailer' in query.lower() or 'retail' in query.lower()) and 'retailer' in df.columns:
        # res_df = res_df[res_df['retailer']=='yes']
        res_filter['retailer'] = 'yes'
    ignore_fields += ['platform', 'retailer']
    return res_df, res_filter, ignore_fields