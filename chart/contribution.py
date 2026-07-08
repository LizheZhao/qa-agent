"""
Contribution insights generation readout and table display prep
"""
import os
import sys
import pandas as pd
import numpy as np
import viz_utils as viz_utils

def build_contribution_tbl_for_insights(client_code, model_group_id, data, agg_data, sorted_times, selected_levels, readout_data):
    """
    build contribution table for insights generation readout by time tag
    """
    time_tag = readout_data.ner_filters["time_tag"]
    select_time = viz_utils.get_time_align_dict(sorted_times[::-1])

    # overall table: Load from total file
    df = viz_utils.filter_bd_data(client_code, model_group_id, selected_levels, ['contribution', 'sales'])
    filtered_df = df[df["time_norm"].isin(sorted_times)]
    keys = ['country', 'kpi', 'detailed_kpi', 'business_unit_for_kpi', 'time', 'period_type', 'time_norm']
    # sales df: the 'overall' total per group with metric == 'sales'
    sales = filtered_df[filtered_df['metric'] == 'sales'][keys + ['value']].rename(columns={'value': 'sales_total'})
    # contribution df: driver-level rows with metric == 'contribution'
    if readout_data:
        # filter by bd in data:
        all_bd = pd.concat([readout_data.df_activity_group, readout_data.df_measure_group, readout_data.df_measure])[
            "business_driver"].dropna().unique()
        filtered_df = filtered_df[filtered_df["driver"].isin(all_bd)]
    contrib = filtered_df[filtered_df['metric'] == 'contribution'].copy()
    merged = contrib.merge(sales, on=keys, how='left')
    merged['contribution_pct'] = merged['value'] / 100  # value is already a % (sums to 100 per group)
    merged['contribution_amount'] = merged['contribution_pct'] * merged['sales_total']
    overall_tbl_config = {"instruction": f"Analysis focus: Analyze total {merged['driver'].unique()} contribution for the current period, and the previous period if available, including all given metric values.",
                          "data": merged.to_json(orient="records")}

    # Contribution vs Spend Share




