# Apply the filter to DataFrame, generate 2 dict + 3 df

import os
import re
import json
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from datetime import datetime
from dateutil.relativedelta import relativedelta
from itertools import combinations

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


def _get_time_answer_dict_fiscal(df: pd.DataFrame, fields: list, answer_dict: dict) -> dict:
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
    if isinstance(p, str):
        p_list = [x.strip() for x in p.split(',') if x.strip() in df['period_type'].unique()]
        p = p_list if len(p_list) else 'irrelevant'
    elif isinstance(p, list):
        valid_p = [x for x in p if x in df['period_type'].unique()]
        p = [next((x for x in p_priority if x in valid_p), 'irrelevant')] if valid_p else 'irrelevant'
    else:
        p = 'irrelevant'

    if isinstance(t, str):
        t_list = [x.strip() for x in t.split(',') if x.strip() in df['time'].unique()]
        t = t_list if len(t_list) else 'irrelevant'
    elif isinstance(t, list):
        t = [x for x in t if x in df['time'].unique()]
    else:
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


def _get_time_answer_dict(df: pd.DataFrame, fields: list, answer_dict: dict) -> dict:
    # answer_dict = ast.literal_eval(answer)
    res_dict = {}
    t = answer_dict.get('time', 'irrelevant')
    p = answer_dict.get('period_type', 'irrelevant')

    p_priority = ['year', 'half', 'quarter', 'month']

    # check for mixed time and period type
    if (isinstance(p, list) and len([x for x in p if x in df['period_type'].dropna().unique()]) > 1) or (isinstance(t, list) and len(df[df['time'].isin(t)]['period_type'].dropna().unique()) > 1):
        p_priority = ['month', 'quarter', 'half', 'year']
    if isinstance(t, list):
        p_priority = [x for x in p_priority if x in df[df['time'].isin(t)]['period_type'].dropna().unique()]
        if len(p_priority) == 0:
            p_priority = ['year', 'half', 'quarter', 'month']

    # type check
    if isinstance(p, str):
        p_list = [x.strip() for x in p.split(',') if x.strip() in df['period_type'].unique()]
        p = p_list if len(p_list) else 'irrelevant'
    elif isinstance(p, list):
        valid_p = [x for x in p if x in df['period_type'].unique()]
        p = [next((x for x in p_priority if x in valid_p), 'irrelevant')] if valid_p else 'irrelevant'
    else:
        p = 'irrelevant'

    if isinstance(t, str):
        t_list = [x.strip() for x in t.split(',') if x.strip() in df['time'].unique()]
        t = t_list if len(t_list) else 'irrelevant'
    elif isinstance(t, list):
        valid_t = [x for x in t if x in df['time'].unique()]
        t = valid_t if len(t) else 'irrelevant'
    else:
        t = 'irrelevant'

    if t == 'irrelevant' and p == 'irrelevant':
        p = [x for x in p_priority if x in df['period_type'].unique()][0]
        t = sorted(list(df[df['period_type'].isin([p])]['time'].unique()))[-2:]
    elif t == 'irrelevant':
        if p != ['year']:
            t = sorted(list(df[df['period_type'].isin(p)]['time'].unique()))
        else:
            # return last two year
            t = sorted(list(df[df['period_type'].isin(p)]['time'].unique()))[-2:]
    elif p == 'irrelevant':
        p = df[df['time'].isin(t)]['period_type'].value_counts().index[0]

    if not isinstance(p, list):
        p = [p]
    if not isinstance(t, list):
        t = [t]

    def parse_month(fiscal_str):
        parts = fiscal_str.split()
        month = int(parts[1])
        year = int(parts[2])
        return (year, month)

    # return last half and yag or last 8 quarters or last 12 months
    time_list = sorted(list(df[df['period_type'].isin(p)]['time'].unique()))
    if p == ['half']:
        time_list = sorted(list(df[df['period_type'].isin(p)]['time'].unique()))
        t = sorted([x for x in time_list if x in t], key=parse_month)[-1: -4: -2]
    if p == ['quarter']:
        time_list = sorted(list(df[df['period_type'] == 'quarter']['time'].unique()))
        t = sorted([x for x in time_list if x in t], key=parse_month)[-8:]
    if p == ['month']:
        t = sorted([x for x in time_list if x in t], key=parse_month)[-12:]
    res_dict['time'] = t if isinstance(t, list) else [t]
    res_dict['period_type'] = p if isinstance(p, list) else [p]
    return res_dict


def _get_time_period_answer_dict(df: pd.DataFrame, ner_results: dict, ner_postprocess_indicator: dict, intention: list) -> dict:
    res_dict = {}

    # ---- resolve period_type ----
    trend = ner_results.get('trend', 'no')
    p = ner_results.get('period_type', 'irrelevant')
    if isinstance(p, list):
        p = p[0]
    period_type_irrelevant = (p == 'irrelevant')

    # load indicators
    calendar_type = ner_postprocess_indicator.get('calendar_type', 'fiscal')
    period_priority = ner_postprocess_indicator.get('period_priority')
    all_period_types = [x for x in period_priority if x in df['period_type'].unique()]

    # trend with no requested period type -> largest available period type
    if trend == 'yes' and p == 'irrelevant':
        p_answer = [all_period_types[0]]
    else:
        if any(item in ['performance', 'margin roi'] for item in intention):
            performance_period_granularity = ner_postprocess_indicator.get('performance_period_granularity', all_period_types[-1])
            report_index = period_priority.index(performance_period_granularity)
            pp = period_priority[:report_index+1][::-1]
        else:
            pp = period_priority[::-1]

        if calendar_type == 'custom':
            p = p.replace("fiscal", "").strip()

        if isinstance(p, str):
            p = [x.strip() for x in p.split(',') if x.strip() in pp]

        valid_p = [next((x for x in p if x in pp), all_period_types[0])]
        p_answer = [next((x for x in pp[:pp.index(valid_p[0]) + 1][::-1] if x in all_period_types),
                         all_period_types[-1])]

    # ---- resolve start/end given resolved period_type ----
    start = ner_results.get('start', 'irrelevant')
    end = ner_results.get('end', 'irrelevant')
    df['start'] = df['start'].astype(int)
    df['end'] = df['end'].astype(int)

    time_length = int(ner_postprocess_indicator.get('trend_time_length') or '2')
    filtered_df = df[df['period_type'].isin(p_answer)].copy()

    time_report_length = ner_postprocess_indicator.get("non_trend_report_length")
    report_yoy = ner_postprocess_indicator.get("report_yoy")

    if start == "irrelevant" or end == "irrelevant":
        if trend == 'yes':
            # trend with no requested start/end -> entire data history for the resolved period type
            start_res = sorted(filtered_df["start"].unique().tolist())
            end_res = sorted(filtered_df["end"].unique().tolist())
            n_req = len(start_res)
            # single-period history with a trend request is degenerate -> snapshot
            if n_req <= 1:
                time_tag = "snapshot"
            elif n_req <= 3:
                time_tag = "period_over_period_change"
            else:
                time_tag = "trend"
        else:
            max_start = max(df[df["period_type"].isin(p_answer)]["start"].unique())
            max_end = max(df[df["period_type"].isin(p_answer)]["end"].unique())
            # period = "year" if report_yoy else p_answer[0]
            period = p_answer[0]
            start_res = subtract_periods(max_start, time_report_length.get(period, time_length)-1, period)
            end_res = subtract_periods(max_end, time_report_length.get(period, time_length)-1, period)
            time_tag = "snapshot" if period_type_irrelevant else "multiple_periods"

    elif int(min(start)) > max(filtered_df['end'].unique()) or int(max(end)) < min(filtered_df['start'].unique()):
        mask = pd.Series(False, index=df.index)
        for s_ref, e_ref in zip(start, end):
            mask |= (df["start"] <= int(e_ref)) & (df["end"] >= int(s_ref))
        tmp = df[mask]

        if tmp.empty:
            max_start = max(df[df["period_type"].isin(p_answer)]["start"].unique())
            max_end = max(df[df["period_type"].isin(p_answer)]["end"].unique())
            period = "year" if report_yoy else p_answer[0]
            start_res = subtract_periods(max_start, time_report_length.get(period, time_length)-1, period)
            end_res = subtract_periods(max_end, time_report_length.get(period, time_length)-1, period)
        else:
            if isinstance(p_answer, list) and p_answer[0] in period_priority:
                p_ans_index = period_priority.index(p_answer[0])
                if p_ans_index > max([period_priority.index(p) for p in tmp['period_type'].unique()]):
                    order = -1
                else:
                    period_priority = period_priority[p_ans_index:]
                    order = 0

            tmp_period_types = [x for x in period_priority if x in tmp['period_type'].unique()]

            valid_p = [next((x for x in p_answer if x in period_priority),
                            [x for x in period_priority if x in tmp_period_types][0])]
            p_answer = [next(
                (x for x in period_priority[:period_priority.index(valid_p[0]) + 1][::-1] if x in tmp_period_types),
                tmp_period_types[order])]

            max_start = max(tmp[tmp["period_type"].isin(p_answer)]["start"].unique())
            max_end = max(tmp[tmp["period_type"].isin(p_answer)]["end"].unique())
            period = p_answer[0]
            start_res = subtract_periods(max_start, time_report_length.get(period, time_length)-1, period)
            end_res = subtract_periods(max_end, time_report_length.get(period, time_length)-1, period)

        # dates given but out of range -> classified by the fallback period count
        if trend == 'yes':
            n_req = len(start_res)
            if n_req <= 1:
                time_tag = "snapshot"
            elif n_req <= 3:
                time_tag = "period_over_period_change"
            else:
                time_tag = "trend"
        else:
            time_tag = "snapshot" if len(start_res) <= 1 else "multiple_periods"

    else:
        start_res = []
        end_res = []

        for start_str, end_str in zip(start, end):
            s = start_str[:-2] + find_closer_time_index(start_str[-2:], p_answer[0], 'start')
            e = end_str[:-2] + find_closer_time_index(end_str[-2:], p_answer[0], 'end')
            start_strs, end_strs = expand_forward_periods(s, e, p_answer[0])
            start_res.extend(start_strs)
            end_res.extend(end_strs)

        len_res = len(start_res)
        if len_res == 1:
            start_res.append(str(int(str(min(start_res))[:4])-1) + str(min(start_res))[-2:])
            end_res.append(str(int(str(min(end_res))[:4])-1) + str(min(end_res))[-2:])

        # len_res is the requested period count before any previous-period padding
        if trend == 'yes':
            time_tag = "period_over_period_change" if len_res <= 3 else "trend"
        else:
            time_tag = "snapshot" if len_res == 1 else "multiple_periods"

    res_dict["start"] = sorted([int(x) for x in set(start_res)])
    res_dict["end"] = sorted([int(x) for x in set(end_res)])
    res_dict["period_type"] = p_answer
    res_dict["time_tag"] = time_tag

    return res_dict


def get_time_answer_dict_given_indicator(df: pd.DataFrame, answer_dict: dict, ner_postprocess_indicator: dict) -> dict:
    res_dict = {}
    t = answer_dict.get('time', 'irrelevant')
    p = answer_dict.get('period_type', 'irrelevant')

    # load indicators
    calendar_type = ner_postprocess_indicator.get('calendar_type', 'fiscal')
    time_length = int(ner_postprocess_indicator.get('trend_time_length') or '2')
    period_priority = ner_postprocess_indicator.get('period_priority')
    # period_priority = [x for x in period_priority if x in df['period_type'].unique()]
    all_period_types = [x for x in period_priority if x in df['period_type'].unique()]

    # type check
    if isinstance(p, str):
        p_list = [x.strip() for x in p.split(',') if x.strip() in period_priority]
        p = p_list if len(p_list) else [all_period_types[0]]
    elif isinstance(p, list):
        valid_p = [next((x for x in p if x in period_priority), all_period_types[0])]
        p = [next((x for x in period_priority[:period_priority.index(valid_p[0]) + 1][::-1] if x in all_period_types),
                  all_period_types[0])]
    else:
        p = [all_period_types[0]]

    if isinstance(t, str):
        t_list = [x.strip() for x in t.split(',') if x.strip() in df[df['period_type'].isin(p)]['time'].unique()]
        t = t_list if len(t_list) else 'irrelevant'
    elif isinstance(t, list):
        valid_t = [x for x in t if x in df[df['period_type'].isin(p)]['time'].unique()]
        t = valid_t if len(valid_t) else 'irrelevant'
    else:
        t = 'irrelevant'

    if t == 'irrelevant':
        years = _find_year(df[df['period_type'].isin(p)]['time'].unique().tolist(), calendar_type, time_length)
        t = _generate_time(p, years, df)

    res_dict['time'] = t if isinstance(t, list) else [t]
    # res_dict['period_type'] = p if isinstance(p, list) else [p]
    return res_dict


def find_closer_time_index(month_str: str, period: str, position: str) -> str:
    month = int(month_str)

    if "year" in period and "half" not in period:
        return "01" if position == "start" else "12"

    elif "half" in period:
        if position == "start":
            return "01" if month <= 6 else "07"
        else:  # end
            return "06" if month <= 6 else "12"

    elif "quarter" in period:
        if position == "start":
            if month <= 3:
                return "01"
            elif month <= 6:
                return "04"
            elif month <= 9:
                return "07"
            else:
                return "10"
        else:  # end
            if month <= 3:
                return "03"
            elif month <= 6:
                return "06"
            elif month <= 9:
                return "09"
            else:
                return "12"

    else:
        return month_str


def _get_answer_dict(df: pd.DataFrame, dimension: str, fields: list, answer_dict: dict, data: str = 'bi') -> dict:
    # answer_dict = ast.literal_eval(answer)
    res_dict = {}
    for ak, av in answer_dict.items():
        # ensure list
        if not isinstance(av, list):
            av = [av]
        if ak in df.columns and ak in fields+['measure_group']:
            if 'irrelevant' in av:
                if 'overall' in df[ak].values:
                    res_dict[ak] = ['overall']
                else:
                    pass
            elif 'all' in av or "specific" in av:
                valid_list = list([x for x in df[ak].dropna().unique() if x not in ['overall', 'portfolio']])  # hard coded for product/bu
                if len(valid_list) >= 1:
                    res_dict[ak] = valid_list
            elif 'overall' in av:
                res_dict[ak] = ['overall'] # hard coded for product
            elif 'relevant' in av:
                res_dict[ak] = ['yes']
            elif ak == 'measure_group':
                res_dict[ak] = [item for item in df[ak].dropna().unique() if any(sub.lower() in item.lower() for sub in av)]
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
                valid_list = [v for v in av if v in list(df[ak].dropna().unique())+['equity', 'ahw'] if v not in ['ahw media', 'pillars']]
                if len(valid_list) >= 1:
                    res_dict[ak] = valid_list
                elif ak == 'product':
                    res_dict[ak] = ['overall']
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


def _minimal_conflicting_subset(df, applied, offender_mask):
    """
    Smallest subset S of `applied` (list of (key, mask)) such that
    S + offender empties the data. `applied` masks are jointly non-empty
    by the invariant maintained in _apply_answer_dict.
    Returns list of keys.
    """
    keys = list(range(len(applied)))
    for r in range(1, len(applied) + 1):
        for combo in combinations(keys, r):
            m = offender_mask.copy()
            for i in combo:
                m &= applied[i][1]
            if not m.any():
                return [applied[i][0] for i in combo]
    return []


def _apply_answer_dict_with_log(df: pd.DataFrame, answer_dict: dict, res_dict=None, ignore_fields=None,
                       prioritized_indicator=None):
    if ignore_fields is None:
        ignore_fields = []
    if res_dict is None:
        res_dict = {}
    if prioritized_indicator is None:
        prioritized_indicator = []

    log = {
        "skipped_no_value": {},   # key -> values that match nothing in the (already-narrowed) data
        "conflicts": [],          # filters that emptied the running conjunction
    }

    tmp = df.copy()
    applied = []                  # list of (key, full-index mask) actually applied, in order
    indicators = [x for x in prioritized_indicator + list(answer_dict.keys()) if x not in ignore_fields]
    seen = set()

    for k in indicators:
        if k in seen:             # prioritized_indicator may duplicate answer_dict keys
            continue
        seen.add(k)

        v = answer_dict.get(k)
        if not isinstance(v, list):
            continue
        if k not in df.columns:
            continue

        if k == "start" or k == "end":
            col_mask = pd.Series(False, index=df.index)
            for s, e in zip(answer_dict["start"], answer_dict["end"]):
                col_mask |= (df["start"].astype(int) <= int(e)) & (df["end"].astype(int) >= int(s))
        elif all(value not in df[k].unique() for value in v):
            log["skipped_no_value"][k] = v
            continue
        elif k not in ignore_fields:
            if res_dict.get(k, '') == 'all':
                col_mask = df[k].isin(v) | df[k].isna()
            else:
                col_mask = df[k].isin(v)
        else:
            continue

        # running conjunction restricted to current tmp
        new_idx = tmp.index.intersection(df.index[col_mask])
        if len(new_idx) == 0:
            minimal = _minimal_conflicting_subset(df, applied, col_mask)
            log["conflicts"].append({
                "filter": k,
                "values": v,
                "conflicts_with": minimal,
                "all_applied_before": [key for key, _ in applied],
            })
            continue  # skip this filter, keep collecting further conflicts

        tmp = tmp.loc[new_idx]
        applied.append((k, col_mask))

    logger.debug(tmp.shape, 'after applying filter')
    return tmp.reset_index(drop=True), log


def _apply_answer_dict(df: pd.DataFrame, answer_dict: dict, res_dict=None, ignore_fields=None,
                       prioritized_indicator=None) -> pd.DataFrame:
    # TODO: log not applied filters
    if ignore_fields is None:
        ignore_fields = []
    if res_dict is None:
        res_dict = {}
    if prioritized_indicator is None:
        prioritized_indicator = []
    tmp = df.copy()
    indicators = prioritized_indicator + list(answer_dict.keys())
    for k in indicators:
        v = answer_dict.get(k)
        if not isinstance(v, list):
            continue
        if k not in df.columns:
            continue
        if k == "start" or k == "end":
            mask = pd.Series(False, index=tmp.index)
            # tmp["start"] = tmp["start"].astype(int)
            # tmp["end"] = tmp["end"].astype(int)
            # valid_min_start = min([x for x in answer_dict['start'] if x in tmp["start"]])
            # valid_max_end = max([x for x in answer_dict['end'] if x in tmp["end"]])
            # tmp = tmp[(tmp['start'] >= int(valid_min_start)) & (tmp['end'] <= int(valid_max_end))]
            for s, e in zip(answer_dict["start"], answer_dict["end"]):
                mask |= (tmp["start"].astype(int) <= int(e)) & (tmp["end"].astype(int) >= int(s))
            tmp = tmp[mask]
        elif all(value not in tmp[k].unique() for value in v):
            continue
        elif k not in ignore_fields:
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


def parse_time(time_str, calendar_type):
    numbers = [int(num) for num in re.findall(r'\d+', time_str)]
    if len(numbers) >= 2:
        return numbers[-1], numbers[-2]
    elif len(numbers) == 1:
        return numbers[-1], None
    else:
        return None, None


def _find_year(time_list: list, calendar_type: str, max_length: int=None) -> list:
    """
    given calendar type, find years from time_list, cap by max_length
    """
    year_set = {parse_time(t, calendar_type)[0] for t in time_list}
    if max_length:
        year_list = sorted(year_set)[-max_length:]
    else:
        year_list = sorted(year_set)
    return year_list


def _generate_time(period_type: list, year_list: list, data: pd.DataFrame=None) -> list:
    """
    given period_type and year_list, generate valid time strings that cover the years
    """
    time_list = []
    for period in period_type:
        if 'month' in period:
            time_format = '{p} {ti} {y}'
            time_list.extend([time_format.format(p=period, ti=ti, y=y) for y in year_list for ti in range(1, 13)])
        if 'quarter' in period:
            time_format = '{p} {ti} {y}'
            time_list.extend([time_format.format(p=period, ti=ti, y=y) for y in year_list for ti in range(1, 5)])
        if 'half' in period:
            time_format = '{p} {ti} {y}'
            time_list.extend([time_format.format(p=period, ti=ti, y=y) for y in year_list for ti in range(1, 3)])
        if 'year' in period:
            time_format = '{p} {y}'
            time_list.extend([time_format.format(p=period, y=y) for y in year_list])
    time_list = [x for x in time_list if x in data['time'].unique()]
    if any('month' in period for period in period_type):
        time_list = time_list[-12:]
    if any('quarter' in period for period in period_type):
        time_list = time_list[-8:]
    if any('half' in period for period in period_type):
        time_list = time_list[-2:]
    return time_list


def _halo_level_to_tagging(halo_level: str, halo_tagging: str, data, ner_filter: dict, ner_dict: dict) -> [dict, dict]:
    if (halo_level and halo_tagging and
            ner_filter.get('data', 'br') == 'bi' and ner_filter.get(halo_tagging) == None):
        product_media_list = list(data[halo_tagging].dropna().unique())
        if ner_dict.get(halo_level, 'irrelevant') == 'all':
            ner_filter[halo_tagging] = [x for x in product_media_list if x not in ['equity', 'ahw']]
            ner_filter[halo_level] = ['overall']
            ner_dict[halo_tagging] = ner_filter[halo_tagging]
        elif (isinstance(ner_dict.get(halo_level), list)
            and ('overall' not in ner_filter.get(halo_level, []))
            and ('overall' in data[halo_level].unique())):
            mapping = {'optic white tp': 'optic white'}
            ner_filter[halo_tagging] = [mapping.get(x, x) for x in ner_dict.get(halo_level) if
                                                x in set(product_media_list).union(mapping.keys())]
            ner_filter[halo_level] = ['overall']
            ner_dict[halo_tagging] = ner_filter[halo_tagging]
            if len(ner_filter[halo_tagging]) < 1:
                del ner_filter[halo_tagging]
    return dict(ner_filter), dict(ner_dict)


def _postprocess_filter(ner_filter: dict, ner_dict: dict, data: pd.DataFrame) -> tuple[dict, dict]:
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
    7. if product_halo is irrelevant, map product to product halo
    8. for campaign, if ner_dict is all, drop blank rows
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
        if (p == ['fiscal year'] or p == ['year']) and len(t) == 1 and re.search(r'\d{4}', t[0]):
            year = int(re.search(r'\d{4}', t[0]).group())
            trend_time = 'fiscal year ' + str(year-1) if 'fiscal year' in data['period_type'] else 'year ' + str(year-1)
            ner_filter['time'].append(trend_time)
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
    product_media_column = next((x for x in ["product_halo", "brand_focused_marketing"] if x in data.columns), None)
    product_column = next((x for x in ["product", "brand"] if x in data.columns), None)
    if (product_column and product_media_column and
            ner_filter.get('data', 'br') == 'bi' and ner_filter.get(product_media_column) == None):
        product_media_list = list(data[product_media_column].dropna().unique())
        if ner_dict.get(product_column, 'irrelevant') == 'all':
            ner_filter[product_media_column] = [x for x in product_media_list if x not in ['equity', 'ahw']]
            ner_filter[product_column] = ['overall']
            ner_dict[product_media_column] = ner_filter[product_media_column]
        elif isinstance(ner_dict.get(product_column), list) and ('overall' not in ner_dict.get(product_column, [])):
            mapping = {'optic white tp': 'optic white'}
            ner_filter[product_media_column] = [mapping.get(x, x) for x in ner_dict.get(product_column) if
                                                x in set(product_media_list).union(mapping.keys())]
            ner_filter[product_column] = ['overall']
            ner_dict[product_media_column] = ner_filter[product_media_column]
            if len(ner_filter[product_media_column]) < 1:
                del ner_filter[product_media_column]
    # postprocess logic 8
    target_cols = [x for x in ["campaign", "retailer_media", "product_halo", "brand_focused_marketing"] if x in data.columns]
    for target_col in target_cols:
        if ner_dict.get(target_col, 'irrelevant') == 'all':
            ner_dict[target_col] = list(data[target_col].dropna().unique())
            ner_filter[target_col] = list(data[target_col].dropna().unique())
    return dict(ner_filter), dict(ner_dict)


def _postprocess_filter_given_indicator(ner_filter: dict, ner_dict: dict,
                                        data: pd.DataFrame, ner_postprocess_indicator: dict,
                                        level_type: dict) -> tuple[dict, dict]:
    """
    postprocess ner_filter based on ner results and postprocess indicators,
    ner_filter: actual list of filter values
    ner_dict: 'relevant' or 'irrelevant'
    1. if intention is source of change, trend is no
    2. if intention is source of change or trend is yes, change time based on period_type and trend_time_length
    3. if intention is performance or roi, change time and period_type based on performance_time_length and
    performance_period_granularity (moved to processing func in sequential ner)
    4. halo_level to halo_tagging mapping based on level_type
    5. general_tagging drop blank when all
    """
    # load indicators
    calendar_type = ner_postprocess_indicator.get('calendar_type', 'fiscal')
    trend_time_length = int(ner_postprocess_indicator.get('trend_time_length') or '2')
    performance_period_granularity = ner_postprocess_indicator.get('performance_period_granularity', 'fiscal quarter')
    period_priority = ner_postprocess_indicator.get('period_priority')
    halo_level = (level_type.get('halo_level') or [None])[0]
    halo_tagging = (level_type.get('halo_tagging') or [None])[0]
    general_tagging = level_type.get('general_tagging', [])

    ner_dict['intention'] = ner_filter['intention']

    # rule 1
    if any(item in ['source of change'] for item in ner_filter.get('intention', [])):
        ner_dict['trend'] = 'no'
        ner_filter['trend'] = 'no'
    # rule 2
    if ('source of change' in ner_filter.get('intention', [])) or (ner_filter.get('trend') == 'yes'):
        p = ner_filter.get('period_type', [])
        start = ner_filter.get('start', [])
        end = ner_filter.get('end', [])
        if len(start) >= 2:
            pass
        else:
            if 'year' in p or 'fiscal year' in p:  # return all yearly data if trend is yes
                valid_start = list(data[data['period_type'].isin(p)]['start'].unique())
                valid_end = list(data[data['period_type'].isin(p)]['end'].unique())
            else:
                period_type_to_max = ner_postprocess_indicator.get("trend_report_length")
                max_length = next((v for k, v in period_type_to_max.items() if k in p), 99)
                valid_start = [d for x in start for d in subtract_periods(x, max_length - len(start), p[0])]
                valid_end = [d for x in end for d in subtract_periods(x, max_length - len(end), p[0])]
            # else:
            #     valid_start = []
            #     valid_end = []
            ner_filter['start'] = [int(x) for x in sorted(set(valid_start + start))]
            ner_filter['end'] = [int(x) for x in sorted(set(valid_end + end))]
    # rule 3
    # if any(item in ['performance', 'margin roi'] for item in ner_filter.get('intention', [])):
    #     p = ner_filter.get('period_type')[0]
    #     t = ner_filter.get('time')
    #     ans_index = period_priority.index(p)
    #     report_index = period_priority.index(performance_period_granularity)
    #     if ans_index > report_index:
    #         report_p = [performance_period_granularity]
    #         years = _find_year(t, calendar_type)
    #         report_t = _generate_time(report_p, years, data=data)
    #         ner_filter['period_type'] = report_p
    #         ner_filter['time'] = report_t
    # rule 4, specific rule, moved to reconciliation
    # if halo_level and halo_tagging:
    #     ner_filter, ner_dict = _halo_level_to_tagging(halo_level, halo_tagging, data,
    #                                                   ner_filter=ner_filter, ner_dict=ner_dict)
    # rule 5
    for tagging_col in general_tagging:
        if ner_dict.get(tagging_col, 'irrelevant') == 'all':
            ner_dict[tagging_col] = list(data[tagging_col].dropna().unique())
            ner_filter[tagging_col] = list(data[tagging_col].dropna().unique())
    return dict(ner_filter), dict(ner_dict)

def _get_level(df, level=None):
    if not level:
        level = ['business_driver', 'business_driver_detail', 'activity_group', 'measure_group', 'measure']
    df['record_level'] = 'invalid'
    for l in level:
        df.loc[(~pd.isna(df[l]) & (df[l] != 'invalid')), 'record_level'] = l
    return df


def get_report_levels_to_target(possible_levels: list, valid: list, target:int = 3) -> list:
    if len(valid) >= target:
        return valid[-target:]

    # Index of first and last valid values
    first_idx = possible_levels.index(valid[0])
    last_idx = possible_levels.index(valid[-1])

    needed = target - len(valid)

    # Take from the front (before first valid)
    front_available = possible_levels[:first_idx]
    take_front = min(len(front_available), needed)
    front = front_available[-take_front:] if take_front > 0 else []

    needed -= take_front

    # Take from the back (after last valid)
    back = possible_levels[last_idx + 1:last_idx + 1 + needed]

    return front + valid + back


def most_granular_level(possible_levels: list, valid: list) -> list:
    """
    Return the most granular (latest in possible_levels) level that appears in cols.
    If none of cols are in possible_levels, return None.
    """
    rank = {lvl: i for i, lvl in enumerate(possible_levels)}
    candidates = [c for c in valid if c in rank]
    return [max(candidates, key=lambda c: rank[c])]


def get_report_levels(df: pd.DataFrame, level:list, ner_filter: dict, is_halo_tagging: bool) -> list:
    """
    get report levels for br data based on the most granular level in core dimensions
    """
    data_source = ner_filter.get("data", "bi")
    core_dimensions = ner_filter.get("core_dimension", {})

    possible_levels = ["business_driver", "business_driver_detail", "activity_group", "measure_group", "measure"]
    tmp_df = _get_level(df)
    levels_in_data = [x for x in possible_levels if x in tmp_df["record_level"].unique()]


    if data_source == "bi":
        return level
    elif is_halo_tagging:
        # if halo tagging
        return get_report_levels_to_target(possible_levels, levels_in_data, target=3)
    elif not core_dimensions:
        return level

    has_bdd = False
    all_filter_levels = set()
    for core_dim, core_dim_filter_list in core_dimensions.items():
        filter_levels = [x for filter in core_dim_filter_list for x in filter.keys() if x in levels_in_data and 'br' in filter.get("source", 'na')]
        all_filter_levels.update(filter_levels)
        if next((x for x in levels_in_data[::-1] if x in filter_levels), level[-1]) == "business_driver_detail":
            has_bdd = True
    if not all_filter_levels:
        all_filter_levels = set(levels_in_data)
    detail_level = most_granular_level(possible_levels, list(all_filter_levels))
    result = get_report_levels_to_target(possible_levels, detail_level, target=3)

    if "business_driver" not in result and has_bdd and "business_driver_detail" not in result:
        result = ["business_driver_detail"] + result[1:]
    return result


def _dedup_df(df, level=None, level_type=None):
    if level is None:
        level = ['activity_group', 'measure_group', 'measure']
    if level_type is None:
        level_type = {}
    hierarchy_levels = level_type.get("general_level", []) + level_type.get("halo_level", [])
    df_with_level = _get_level(df)
    df_with_level['record_level'] = df_with_level['record_level'].map({k: k for k in level}).fillna('invalid')
    priority = {l: i for i, l in enumerate(level)}
    # drop rows with same metric value
    dedup_df = (
        df_with_level.assign(_rank=df_with_level['record_level'].map(priority))
        .sort_values('_rank')
        .drop_duplicates(subset=hierarchy_levels + ['time', 'metric', 'value'], keep='last')
        .drop(columns='_rank')
    )
    return dedup_df


def _get_level_answer(df, level=None):
    if level is None:
        level = ['activity_group', 'measure_group', 'measure']
    l1, l2, l3 = level
    df_with_level = _get_level(df)
    logger.debug('record level information:')
    logger.debug(df_with_level['record_level'].value_counts())
    df_with_level['record_level'] = df_with_level['record_level'].map({k: k for k in level}).fillna('invalid')
    ag_res = df_with_level[df_with_level['record_level']==l1].reset_index(drop=True).copy()
    mg_res = df_with_level[df_with_level['record_level']==l2].reset_index(drop=True).copy()
    # mg_res = mg_res[~mg_res[l1].isin(ag_res[l1])].reset_index(drop=True).copy()
    m_res = df_with_level[df_with_level['record_level']==l3].reset_index(drop=True).copy()
    # m_res = m_res[~m_res[l1].isin(ag_res[l1])].reset_index(drop=True).copy()
    # m_res = m_res[~m_res[l1].isin(mg_res[l1])].reset_index(drop=True).copy()
    return ag_res, mg_res, m_res


def map_keyword_in_query(df: pd.DataFrame, query: str, ner_filter: dict, ignore_fields: list) -> tuple[pd.DataFrame, dict, list]:
    res_df = df.copy()
    res_filter = dict(ner_filter)
    ag_list = res_df.activity_group.dropna().unique()
    mg_list = res_df.measure_group.dropna().unique()
    m_list = res_df.measure.dropna().unique()
    ag_filter_list, mg_filter_list, m_filter_list = [], [], []
    keyword_map = {'amazon': ['amazon', 'amz'], 'meta': ['meta', 'facebook', 'fb'], 'tiktok': ['tiktok'],
                   'ttd(the trade desk)': ['ttd', 'trade desk'], 'youtube': ['youtube', 'trueview'],
                   'google': ['google'], 'pinterest': ['pinterest'], 'twitter': ['twitter'], 'walmart': ['walmart'],
                   'bing': ['bing'], 'target': ['target'], 'iheart': ['iheart'], 'kroger': ['kroger'],
                   'sams club': ['sams club', "sam's club"], 'podcast': ['podcast'], 'instacart': ['instacart'],
                   'ahw': ['ahw'], 'equity': ['masterbrand'], 'masterbrand': ['masterbrand'],
                   'sub-brand': ['optic white', 'total tp']}
    platform_list = []
    for key, val in keyword_map.items():
        if key in query.lower():
            if key not in ['ahw', 'equity', 'masterbrand', 'sub-brand']:
                platform_list.append(key)
            for v in val:
                ag_filter_list.extend([x for x in ag_list if v in x])
                mg_filter_list.extend([x for x in mg_list if v in x])
                m_filter_list.extend([x for x in m_list if v in x])
    if len(mg_filter_list) or len(m_filter_list) or len(ag_filter_list):
        ag_filter_list += res_filter.get('activity_group', [])
        mg_filter_list += res_filter.get('measure_group', [])
        m_filter_list += res_filter.get('measure', [])
        # res_df = res_df[(res_df['measure_group'].isin(mg_filter_list)) | (res_df['measure'].isin(m_filter_list)) | (res_df['activity_group'].isin(ag_filter_list))]
        # if keyword in list match, overwrite media_channe
        # res_filter['media_channel'] = [sorted(res_df['media_channel'].dropna().unique())[0]]
        res_filter['platform'] = platform_list
        # ignore_fields += ['media_channel', 'activity_group', 'measure_group', 'measure']
    if ('retailer' in query.lower() or 'retail' in query.lower()) and 'retailer' in df.columns:
        # res_df = res_df[res_df['retailer'] == 'yes']
        res_filter['retailer'] = 'yes'
    ignore_fields += ['platform', 'retailer']
    return res_df, res_filter, ignore_fields


def _get_specific_values(df: pd.DataFrame, ner_filter: dict, ner_res: dict,
                         level_type: dict, ignore_fields: list) -> tuple[dict, dict, list]:
    """
    If any of general levels or taggings is classified into "specific" by LLM
    check for exact matches in spacy captured term list and update ner filter accordingly
    """
    ner_res_correct = {}
    custom_levels = level_type.get("general_level", [])
    custom_taggings = level_type.get("general_tagging", [])
    for key, val in ner_res.items():
        if key == "business_driver":
            em_cols = ["activity_group", "business_driver_detail", "business_driver"]
            spacy_type = ["core_dimension", "business_driver_captured"]
        elif key in custom_levels:
            em_cols = update_cols = custom_levels
            spacy_type = ["custom_level"]
        elif key in custom_taggings:
            em_cols = update_cols = custom_taggings
            spacy_type = ["custom_tagging"]
        else:
            em_cols = update_cols = []
            spacy_type = []
        if len(em_cols):
            if (isinstance(val, str) and val == "specific") or any(ner_filter.get(st) for st in spacy_type):
                em_dict = {c: set(df[c].dropna()) for c in em_cols}
                if key == "business_driver":
                    em_dict = {key: set(df[em_cols].stack().dropna())}
                # get original terms in em columns
                spacy_set = set()
                for st in spacy_type:
                    if isinstance(ner_filter.get(st), dict):
                        if st == "business_driver_captured":
                            for term, filters in ner_filter.get(st, {}).items():
                                spacy_set.update([t for x in filters for t in x.get("business_driver", [term])])
                        for term, filters in ner_filter.get(st, {}).items():
                            spacy_set.update([t for x in filters for c in em_cols for t in x.get(c, [term])])
                    elif isinstance(ner_filter.get(st), list):
                        spacy_set.update(ner_filter.get(st))
                valid_list = {col: [v for v in spacy_set if v in val_set] for col, val_set in em_dict.items()}
                em_filter = {k: v for k, v in valid_list.items() if len(v) > 0}
                if key in em_filter:
                    ner_filter.update(em_filter)
                    ignore_fields += [key]
                    for key in em_filter:
                        ner_res_correct[key] = "specific"
                else:
                    ner_res_correct[key] = "irrelevant"
                    if "overall" in df[key].unique():
                        ner_filter.update({key: ["overall"]})
    ner_res.update(ner_res_correct)
    return dict(ner_filter), dict(ner_res), ignore_fields


def get_period_delta(period: str):
    if "month" in period:
        return relativedelta(months=1)
    elif "quarter" in period:
        return relativedelta(months=3)
    elif "half" in period:
        return relativedelta(months=6)
    elif "year" in period:
        return relativedelta(years=1)
    else:
        raise ValueError(f"Unsupported period type: {period}")


def subtract_periods(date_str, num_periods, period):
    date = datetime.strptime(str(date_str), "%Y%m")
    delta = get_period_delta(period)
    return [int((date - delta * i).strftime("%Y%m")) for i in range(num_periods + 1)]


def expand_forward_periods(start_str, end_str, period):
    start = datetime.strptime(str(start_str), "%Y%m")
    end = datetime.strptime(str(end_str), "%Y%m")
    delta = get_period_delta(period)

    result_start = []
    result_end = []
    curr = start
    while curr <= end:
        result_start.append(int(curr.strftime("%Y%m")))
        curr += delta
    curr = end
    while curr >= start:
        result_end.append(int(curr.strftime("%Y%m")))
        curr -= delta
    return sorted(result_start), sorted(result_end)


def _apply_filtering_logic(clientCode: str, ner_filter: dict, data: pd.DataFrame):
    if clientCode == "LINKEDIN":
        # exclude owned media for roi/performance/spend intention if not in core dimension
        owned_media = ["lol non lan", "events", "seo", "email", "owned", "owned marketing", "owned media"]
        org_terms = {fl.get('org_term', None) for k, f in ner_filter.get("core_dimension", {}).items() for fl in f}
        if not any(x for x in owned_media if x in org_terms):
            filtered_data = data[~data["activity_group"].isin(owned_media)].reset_index(drop=True)
            return filtered_data
    return data