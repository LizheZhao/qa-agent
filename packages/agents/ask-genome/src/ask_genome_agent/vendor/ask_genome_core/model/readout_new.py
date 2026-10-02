"""Parallel BI/BR readout implementation with separate text and table routes.
"""

from dataclasses import dataclass

import pandas as pd

import src.model.readout as legacy_readout
import src.model.readout_utils as readout_utils


@dataclass
class PreparedReadoutData:
    """Shared prepared data consumed by both output routes."""

    data: pd.DataFrame
    support_df: pd.DataFrame | None

    @property
    def has_level_config(self) -> bool:
        return self.support_df is not None


def _prepare_readout_data(
        client_code, model_group_id, data, ner_res_dict, ner_dict):
    """Apply the data-only portion of legacy readout_adjust()."""
    support_df = readout_utils.load_level_rename_display(client_code, model_group_id)
    if support_df.empty:
        return PreparedReadoutData(data, None), ner_dict

    data = readout_utils.level_filter_default(support_df, data, ner_res_dict)
    if data.empty:
        return PreparedReadoutData(data, None), ner_dict

    sort_cols = [
        column for column in support_df["original"].drop_duplicates()
        if column in data.columns
    ]
    if sort_cols:
        data = data.sort_values(
            by=sort_cols,
            kind="mergesort",
            key=lambda values: values.astype(str).str.lower(),
        )
    data = data[
        sort_cols + [column for column in data.columns if column not in sort_cols]
    ]

    ner_dict = ner_dict.copy()
    ner_dict["dimension"] = [
        column for column in sort_cols if column in ner_dict["dimension"]
    ]
    halo_level = support_df.loc[support_df["mapped"].isna(), "original"].iloc[0]
    if halo_level in data.columns:
        data = data.drop(columns=[halo_level])
        ner_dict["dimension"] = [
            column for column in ner_dict["dimension"]
            if column != halo_level
        ]

    return PreparedReadoutData(data, support_df), ner_dict


def _build_readout_text(
        prepared_data, *, ner_dict, client_code, model_group_id,
        ner_res_dict, how_many, driver_high, ner_filter, sort_by_col,
        intention, growth_col, trend_check, rank_check, sort_metric,
        metric_df, metric_configs, threshold=2):
    """Build readout text from the same prepared data used by tables."""
    data = prepared_data.data
    top_data = data
    if not data.empty:
        top_data = readout_utils._top_data(data, ner_dict, how_many)
        top_data = readout_utils._filter_top_media_channel(
            driver_high, how_many, ner_filter, ner_dict, sort_by_col, top_data
        )

    readout = readout_utils._get_readout(
        intention,
        growth_col,
        trend_check,
        rank_check,
        sort_metric,
        how_many,
        ner_dict,
        ner_filter,
        data,
        top_data,
        metric_df,
        metric_configs,
    )
    if not prepared_data.has_level_config:
        return readout

    # Reproduce the extra narrative added by the legacy readout adjustment.
    dimensions = ner_dict.get("dimension", [])
    formatted_dims = [dimension.replace("_", " ") for dimension in dimensions]
    if formatted_dims:
        readout = (
            f"The following groups of data are structured around "
            f"{', '.join(formatted_dims)}. \n{readout}"
        )

    level_cols = [
        column for column in data.columns
        if column in prepared_data.support_df["original"].unique().tolist()
    ]
    if len(data[level_cols].drop_duplicates()) > threshold:
        main_metric = ner_filter.get("main_metric", ["roi"])[0]
        tactics_col = "tactics"
        if ner_filter.get("data", "bi") == "br":
            candidates = [
                column for column in
                ["business_driver", "business_driver_detail", "tactics"]
                if column in data.columns
            ]
            for column in reversed(candidates):
                if not data[column].isna().all():
                    tactics_col = column
        _, detail_analysis = readout_utils.compare_country_by_group_mwu(
            data, dimensions, main_metric, tactics_col=tactics_col
        )
        readout = "LINKEDIN no table readout. " + readout + "\n\n" + detail_analysis

    return readout


def _prepare_aggregate_new(
        data, period_type, ner_dict, sort_metric, rank_check, metric_df,
        metric_configs, growth_col, sort_by_col):
    """Prepare aggregate data with its own growth and sort metadata."""

    if data.empty:
        return data
    data = readout_utils.quantize_metric_values(data, metric_configs)
    data = readout_utils._calc_growth(period_type, ner_dict, data)
    data = readout_utils._sort_filt_data(
        data, ner_dict, sort_metric, sort_by_col, rank_check, metric_df
    )
    data = readout_utils._ignore_growth_outlier(growth_col, data)
    return data


def _prepare_diag_new(
        data, period_type, ner_dict, sort_metric, rank_check, metric_df,
        metric_configs, growth_col, sort_by_col):
    """Prepare diagnostic data with its own growth and sort metadata."""

    if data.empty:
        return data
    data = readout_utils.quantize_metric_values(data, metric_configs)
    data = readout_utils._calc_growth(period_type, ner_dict, data)
    data = readout_utils._sort_filt_data(
        data, ner_dict, sort_metric, sort_by_col, rank_check, metric_df
    )
    data = readout_utils._ignore_growth_outlier(growth_col, data)
    return data


def _build_bi_readout_str_new(context):
    """Render BI detail, aggregate, and diagnostic data as text only."""

    prepared_detail = context["prepared_detail"]
    ner_dict = context["ner_dict"]
    readout = _build_readout_text(
        prepared_detail,
        ner_dict=ner_dict,
        client_code=context["client_code"],
        model_group_id=context["model_group_id"],
        ner_res_dict=context["ner_res_dict"],
        how_many=context["how_many"],
        driver_high=context["driver_high"],
        ner_filter=context["ner_filter"],
        sort_by_col=context["sort_by_col"],
        intention=context["intention"],
        growth_col=context["growth_col"],
        trend_check=context["trend_check"],
        rank_check=context["rank_check"],
        sort_metric=context["sort_metric"],
        metric_df=context["metric_df"],
        metric_configs=context["metric_configs"],
    )
    # Aggregate text precedes the detailed readout, matching the legacy output.
    agg_data = context["agg_data"]
    if not agg_data.empty:
        if context["client_code"] == "COLGUS":
            agg_ner_dict = ner_dict.copy()
            agg_ner_dict["driver"] = "tactics_detail"
            agg_readout = (
                f"Overall {context['intention']} of each product media is as follows:"
                + readout_utils._get_granular_level_pivot(
                    agg_data,
                    agg_ner_dict,
                    context["ner_filter"],
                    context["trend_check"],
                    context["metric_df"],
                )
            )
        else:
            agg_readout = readout_utils._get_readout(
                context["intention"],
                context["growth_col"],
                context["trend_check"],
                context["rank_check"],
                context["sort_metric"],
                context["how_many"],
                ner_dict,
                context["ner_filter"],
                agg_data,
                agg_data,
                context["metric_df"],
                context["metric_configs"],
            )
        readout = agg_readout + readout
    readout = readout_utils._readout_reformat(
        readout, context["agg_view_df"]
    )

    readout_diag = ""
    diag_data = context["diag_data"]
    if not diag_data.empty:
        top_data_diag = readout_utils._top_data(
            diag_data, ner_dict, context["how_many"]
        )
        top_data_diag = readout_utils._filter_top_media_channel(
            context["driver_high"],
            context["how_many"],
            context["ner_filter"],
            ner_dict,
            context["sort_by_col"],
            top_data_diag,
        )
        readout_diag = readout_utils._get_readout(
            context["intention"],
            context["growth_col"],
            context["trend_check"],
            context["rank_check"],
            context["sort_metric"],
            context["how_many"],
            ner_dict,
            context["ner_filter"],
            diag_data,
            top_data_diag,
            context["metric_df"],
            context["metric_configs"],
        )
        readout_diag = readout_utils._readout_reformat(
            readout_diag, context["agg_view_df"]
        )
    return readout, readout_diag


def _build_bi_tables_new(context):
    """Render BI detail, aggregate, and diagnostic display tables only."""

    prepared_detail = context["prepared_detail"]
    ner_dict = context["ner_dict"]
    pivot_table, pivot_sort_col = readout_utils._get_granular_table(
        prepared_detail.data,
        ner_dict,
        context["metrics_order"],
        context["ner_filter"],
        context["rank_check"],
        context["sort_by_col"],
        context["metric_df"],
        context["client_code"],
        context["model_group_id"],
        context["metric_configs"],
    )
    pivot_biz_table = pd.DataFrame()
    pivot_biz_sort_col = None
    if not context["agg_data"].empty:
        pivot_biz_table, pivot_biz_sort_col = readout_utils._get_granular_table(
            context["agg_data"],
            ner_dict,
            context["metrics_order"],
            context["ner_filter"],
            context["rank_check"],
            context["sort_by_col"],
            context["metric_df"],
            context["client_code"],
            context["model_group_id"],
            context["metric_configs"],
        )

    # Apply the same display-level top-N and dimension cleanup to both tables.
    dimensions = ner_dict["dimension"]
    pivot_table = readout_utils._dynamic_top(
        dimensions, pivot_sort_col, pivot_table, context["metric_df"]
    )
    pivot_table = readout_utils._drop_single_dim(dimensions, pivot_table)
    pivot_biz_table = readout_utils._dynamic_top(
        dimensions, pivot_biz_sort_col, pivot_biz_table, context["metric_df"]
    )
    pivot_biz_table = readout_utils._drop_single_dim(dimensions, pivot_biz_table)

    diag_table = pd.DataFrame()
    diag_data = context["diag_data"]
    if not diag_data.empty:
        diag_table, diag_sort_col = readout_utils._get_granular_table(
            diag_data,
            ner_dict,
            context["metrics_order"],
            context["ner_filter"],
            context["rank_check"],
            context["sort_by_col"],
            context["metric_df"],
            context["client_code"],
            context["model_group_id"],
            context["metric_configs"],
        )
        diag_table = readout_utils._dynamic_top(
            dimensions, diag_sort_col, diag_table, context["metric_df"]
        )
        diag_table = readout_utils._drop_single_dim(dimensions, diag_table)

    client_code = context["client_code"]
    model_group_id = context["model_group_id"]
    if client_code == "COLGUS" and model_group_id == 3041:
        pivot_table, pivot_biz_table = readout_utils._agg_table_format(
            pivot_table,
            pivot_biz_table,
            context["agg_view_df"],
            context["views"],
            "",
            "Product",
        )
        pivot_table = readout_utils._rename_unwanted_tactics(pivot_table)
        pivot_biz_table = readout_utils._rename_unwanted_tactics(pivot_biz_table)
        pivot_biz_table = readout_utils._pillar_pivot_table_rename(
            context["core_dimension_composite"],
            context["core_dimension"],
            pivot_biz_table,
        )
    elif client_code == "COLGUS" and model_group_id > 3041:
        pivot_table = readout_utils._rename_product_media_other_category(
            context["ner_filter"], context["dim_cols"][0], pivot_table
        )
    else:
        pivot_table, pivot_biz_table = readout_utils._agg_table_format(
            pivot_table,
            pivot_biz_table,
            context["agg_view_df"],
            context["views"],
            context["concatenate"],
        )
    return pivot_table, pivot_biz_table, diag_table, pivot_sort_col


def _build_bi_fixtext_str_new(
        client_code, model_group_id, data, dim_cols, driver_tags, ner_filter,
        ner_res_dict, metric_df, metric_configs):
    """Build BI fixtext from the shared query-scoped AG/MG/M data."""

    fixtext = ""
    overall_table = pd.DataFrame()
    dim_dict = {
        column: data[column].unique().tolist()
        for column in dim_cols
    }
    uniq_dim_ner = 1
    uniq_dim_data = (
        data[dim_cols].drop_duplicates().shape[0]
        if not data.empty
        else 1
    )
    for dimension in dim_cols:
        values = ner_filter.get(dimension, [])
        if len(values) > 1:
            uniq_dim_ner *= len(values)
    uniq_dim = max(uniq_dim_ner, uniq_dim_data)
    _, tagging = readout_utils._get_level_tagging(ner_filter, client_code)
    overall_context = (
        not ner_filter.get("core_dimension", {})
        and readout_utils._custom_tagging_condition(ner_filter, tagging)
        and not ner_filter.get("core_dimension_composite", {})
        and not ner_filter.get("diag_tagging", {})
    )
    if overall_context:
        fixtext, overall_table = readout_utils._overall_metric_w_driver(
            client_code,
            model_group_id,
            ner_filter,
            ner_res_dict,
            driver_tags,
            metric_df,
            dim_dict,
            metric_configs,
        )
        fixtext = readout_utils.convert_display_format(fixtext)
        if uniq_dim >= 4:
            fixtext = ""


    # if time_message:
    #     fixtext = time_message + "\n\n" + fixtext
    return fixtext, overall_table


def _build_bi_principle_pretext_new(
        client_code, model_group_id, data, dim_cols, intention):
    """Build BI principle text from the shared query-scoped AG/MG/M data."""

    principle_pretext = readout_utils.genome_principle_test_new(
        client_code,
        model_group_id,
        data,
        dim_cols,
        intention,
    )
    if intention == "spending":
        planner_principle = readout_utils.load_planner_principle(
            client_code, model_group_id
        )
        genome_principle = readout_utils.load_genome_principle(
            client_code, model_group_id
        )
        summary = readout_utils._bi_spend_range(
            client_code,
            model_group_id,
            planner_principle,
            genome_principle,
            data,
            dim_cols,
        )
        if summary and not principle_pretext:
            principle_pretext = "**Additional Insights from ROI Genome.**\n\n"
        if summary:
            principle_pretext += summary
    return principle_pretext


def _prepare_bi_supplemental_context_new(
        client_code, df_dict, ner_filter):
    """Build one query-scoped BI frame from the raw AG/MG/M partitions.

    The three frames have already passed the query filtering pipeline. This
    scope intentionally does not apply the level presentation rules for halo,
    product-media, or custom-aggregated rows. Supplemental text only needs
    dimensions, channel, and date coverage, so metric values and hierarchy
    driver columns are not carried into the shared frame.
    """

    frames = [
        df_dict[key].copy()
        for key in ("ag_df", "mg_df", "m_df")
    ]
    data = pd.concat(frames, ignore_index=True, sort=False)
    if data.empty:
        raise ValueError("No valid data.")
    dim_cols = readout_utils._get_dim_col_in_pivot_table_unique(
        df_dict, ner_filter, client_code
    )

    level, _ = readout_utils._get_level_tagging(ner_filter, client_code)
    data, dim_cols = readout_utils._level_to_drop(level, dim_cols, data)
    if "core_dimension_term" in data.columns:
        dim_cols = dim_cols + ["core_dimension_term"]

    # Use one stable channel name across clients and source levels.
    if "media_channel" not in data.columns and "marketing_channel" in data.columns:
        data = data.rename(columns={"marketing_channel": "media_channel"})

    scope_cols = dim_cols + ["media_channel", "start", "end"]
    data = data[scope_cols].drop_duplicates(ignore_index=True)
    return data, dim_cols



def _build_bi_benchmark_new(context):
    """Build BI benchmark text and table from the combined BI data."""

    data = context["prepared_detail"].data
    ner_dict = context["ner_dict"]
    benchmark_idx = readout_utils.load_benchmark_data(
        context["client_code"], context["model_group_id"]
    )
    if context["client_code"] != "COLGUS":
        return readout_utils._benchmark_output(
            data,
            ner_dict,
            context["how_many"],
            context["intention"],
            context["core_dimension"],
            benchmark_idx,
        )

    top_data = data
    if not data.empty:
        top_data = readout_utils._top_data(data, ner_dict, context["how_many"])
        top_data = readout_utils._filter_top_media_channel(
            context["driver_high"],
            context["how_many"],
            context["ner_filter"],
            ner_dict,
            context["sort_by_col"],
            top_data,
        )
    return readout_utils._merge_benchmark(
        context["intention"],
        context["media_channel_check"],
        context["core_dimension"],
        context["core_dimension_composite"],
        ner_dict,
        data,
        top_data,
        benchmark_idx,
    )


def _process_bi_detail_share_new(
        client_code, model_group_id, df_dict, driver_tags, ner_filter,
        ner_res_dict, map_dict, metric_df, agg_view_df, tactics_level,
        metric_configs):
    """BI detail/share processing with independent text and table routes."""
    if ner_filter["intention"][0] == "planner":
        return legacy_readout._process_bi_detail_share(
            client_code,
            model_group_id,
            df_dict,
            driver_tags,
            ner_filter,
            ner_res_dict,
            map_dict,
            metric_df,
            agg_view_df,
            tactics_level,
            metric_configs,
        )

    file_type, data_type, driver_high, driver_detail = driver_tags
    intention = ner_filter["intention"][0]
    trend_check = ner_res_dict.get("trend", "no")
    rank_check = ner_res_dict.get("rank", "na")
    how_many = ner_filter.get("how_many", "na")
    sort_metric = map_dict[intention]["sortMetric"]
    metrics_order = ner_filter["metric"]
    period_type = (ner_filter.get("period_type") or ["year"])[0]
    core_dimension = ner_filter.get("core_dimension", {})
    core_dimension_composite = ner_filter.get("core_dimension_composite", {})
    diag_tagging = ner_filter.get("diag_tagging", {})
    product_halo_check = ner_filter.get("product_halo", "irrelevant")
    media_channel_check = ner_res_dict.get("media_channel", "irrelevant")
    level, tagging = readout_utils._get_level_tagging(ner_filter, client_code)
    custom_tagging_condition = readout_utils._custom_tagging_condition(
        ner_filter, tagging
    )

    # Stage 1: filter and normalize the selected BI source level.
    curr_df = readout_utils._halo_tagging_filter(
        df_dict[file_type + "_df"], ner_filter, core_dimension_composite
    )
    if curr_df.empty:
        raise ValueError("No valid data.")
    dim_cols = readout_utils._get_dim_col_in_pivot_table_unique(
        df_dict, ner_filter, client_code
    )
    product_media_check = (
        "custom_aggregated" in curr_df.columns
        and (~curr_df["custom_aggregated"].isin(["no", "diagnostic"])).all()
    )
    diagnostic_check = (
        "custom_aggregated" in curr_df.columns
        and (curr_df["custom_aggregated"] == "diagnostic").all()
    )
    if client_code == "COLGUS":
        curr_df = readout_utils._filter_paid_search(intention, curr_df)
        if model_group_id == 3041:
            curr_df = readout_utils._filter_pillar(
                product_halo_check, core_dimension, curr_df
            )
            curr_df = readout_utils._filter_special_media(
                product_halo_check, curr_df
            )

    if not (client_code == "COLGUS" and model_group_id != 3041):
        data1 = readout_utils._read_and_process_csv(curr_df, ner_filter)
    else:
        data1, dim_cols = readout_utils._read_and_process_csv_other_category(
            curr_df, ner_filter, dim_cols
        )
    data1, dim_cols = readout_utils._level_to_drop(level, dim_cols, data1)
    if "core_dimension_term" in data1.columns:
        dim_cols += ["core_dimension_term"]

    # dim_dict = {column: data1[column].unique().tolist() for column in dim_cols}
    # uniq_dim_ner = 1
    # uniq_dim_data = data1[dim_cols].drop_duplicates().shape[0] if not data1.empty else 1
    # for dimension in dim_cols:
    #     values = ner_filter.get(dimension, [])
    #     if len(values) > 1:
    #         uniq_dim_ner *= len(values)
    # uniq_dim = max(uniq_dim_ner, uniq_dim_data)

    # Preserve the existing data-selection decision before output rendering.
    overall_context = (
        not core_dimension
        and custom_tagging_condition
        and not core_dimension_composite
        and not diag_tagging
    )
    overall_ag_only = overall_context
    if overall_context:
        if "custom_aggregated" in data1.columns and not product_media_check:
            data1 = data1[data1["custom_aggregated"].isin(["no", "diagnostic"])]
        if client_code == "COLGUS" and model_group_id == 3041:
            data1 = data1[~data1["media_channel"].isin(["special media", "non-media"])]

    # Resolve the configured tactic representation for this file level.
    concatenate = tactics_level[
        (tactics_level["dataType"] == "bi")
        & (tactics_level["file_type"] == file_type)
    ]["concatenate"].values[0]
    if concatenate == "yes" and not diagnostic_check:
        data = readout_utils._concatenate_ag_mg_bi(data1, driver_detail)
        driver_high = "media_channel" if client_code == "COLGUS" else "tactics_concat"
    else:
        data = data1

    ner_dict = readout_utils._get_ner_key_value(
        data, driver_high, file_type, data_type, driver_detail, dim_cols
    )
    if data.empty:
        raise ValueError("No valid data.")
    # Resolve detail, aggregate, and diagnostic views independently.
    agg_data, data, diag_data, views, ner_dict = readout_utils._get_agg_data(
        data, agg_view_df, ner_dict
    )
    data = readout_utils.quantize_metric_values(data, metric_configs)
    data = readout_utils._calc_growth(period_type, ner_dict, data)
    growth_col, sort_by_col = readout_utils._get_growth_sort_col(
        trend_check, data
    )
    if not data.empty:
        data = readout_utils._sort_filt_data(
            data, ner_dict, sort_metric, sort_by_col, rank_check, metric_df
        )
        data = readout_utils._ignore_growth_outlier(growth_col, data)

    # Stage 2: prepare shared data once, then route it independently below.
    prepared_detail, ner_dict = _prepare_readout_data(
        client_code, model_group_id, data, ner_res_dict, ner_dict
    )
    agg_data = _prepare_aggregate_new(agg_data, period_type, ner_dict, sort_metric, rank_check, metric_df, metric_configs, growth_col, sort_by_col)
    diag_data = _prepare_diag_new(diag_data, period_type, ner_dict, sort_metric, rank_check, metric_df, metric_configs, growth_col, sort_by_col)
    # combined = pd.concat([prepared_detail.data, agg_data, diag_data], ignore_index=True)
    context = {
        "client_code": client_code,
        "model_group_id": model_group_id,
        "ner_filter": ner_filter,
        "ner_res_dict": ner_res_dict,
        "metric_df": metric_df,
        "metric_configs": metric_configs,
        "agg_view_df": agg_view_df,
        "prepared_detail": prepared_detail,
        "ner_dict": ner_dict,
        "agg_data": agg_data,
        "diag_data": diag_data,
        # "combined": combined,
        "intention": intention,
        "period_type": period_type,
        "trend_check": trend_check,
        "rank_check": rank_check,
        "how_many": how_many,
        "sort_metric": sort_metric,
        "sort_by_col": sort_by_col,
        "growth_col": growth_col,
        "driver_high": driver_high,
        "metrics_order": metrics_order,
        "views": views,
        "concatenate": concatenate,
        "core_dimension": core_dimension,
        "core_dimension_composite": core_dimension_composite,
        "media_channel_check": media_channel_check,
        "driver_tags": driver_tags,
        "overall_context": overall_context,
    }
    # Stage 3: render text and tables without either route controlling the other.
    readout, readout_diag = _build_bi_readout_str_new(context)
    benchmark_str, benchmark_data = _build_bi_benchmark_new(context)
    (
        pivot_table,
        pivot_biz_table,
        diag_table,
        pivot_sort_col,
    ) = _build_bi_tables_new(context)
    return (
        readout,
        pivot_table,
        benchmark_data,
        benchmark_str,
        pd.DataFrame(),
        pd.DataFrame(),
        "",
        pivot_biz_table,
        pivot_sort_col,
        product_media_check,
        overall_ag_only,
        pd.DataFrame(),
        "",
        diag_table,
        readout_diag,
    )


def _build_br_readout_str_new(context):
    """Render BR detail and aggregate data as readout text only."""

    readout = ""
    if not context["prepared_detail"].data.empty:
        readout = _build_readout_text(
            context["prepared_detail"],
            ner_dict=context["ner_dict"],
            client_code=context["client_code"],
            model_group_id=context["model_group_id"],
            ner_res_dict=context["ner_res_dict"],
            how_many=context["how_many"],
            driver_high=context["driver_high"],
            ner_filter=context["ner_filter"],
            sort_by_col=context["sort_by_col"],
            intention=context["intention"],
            growth_col=context["growth_col"],
            trend_check=context["trend_check"],
            rank_check=context["rank_check"],
            sort_metric=context["sort_metric"],
            metric_df=context["metric_df"],
            metric_configs=context["metric_configs"],
        )

    prepared_aggregate = context["prepared_aggregate"]
    if prepared_aggregate is not None:
        agg_readout = _build_readout_text(
            prepared_aggregate,
            ner_dict=context["agg_ner_dict"],
            client_code=context["client_code"],
            model_group_id=context["model_group_id"],
            ner_res_dict=context["ner_res_dict"],
            how_many=context["how_many"],
            driver_high=context["driver_high"],
            ner_filter=context["ner_filter"],
            sort_by_col=context["sort_by_col"],
            intention=context["intention"],
            growth_col=context["growth_col"],
            trend_check=context["trend_check"],
            rank_check=context["rank_check"],
            sort_metric=context["sort_metric"],
            metric_df=context["metric_df"],
            metric_configs=context["metric_configs"],
        )
        readout = agg_readout + readout
    # if context["readout_notice_required"]:
    #     readout = "LINKEDIN no table readout. " + readout
    return readout


def _prepare_br_supplemental_context_new(client_code, df_dict, ner_filter):
    """Build one query-scoped BR frame from the raw AG/MG/M partitions."""

    data = pd.concat(
        [df_dict[key].copy() for key in ("ag_df", "mg_df", "m_df")],
        ignore_index=True,
        sort=False,
    )
    if data.empty:
        raise ValueError("No valid data.")
    dim_cols = readout_utils._get_dim_col_in_pivot_table_unique(
        df_dict, ner_filter, client_code
    )
    level, _ = readout_utils._get_level_tagging(ner_filter, client_code)
    data, dim_cols = readout_utils._level_to_drop(level, dim_cols, data)
    if "core_dimension_term" in data.columns:
        dim_cols += ["core_dimension_term"]

    scope_cols = []
    for column in dim_cols + ["business_driver", "start", "end"]:
        if column in data.columns and column not in scope_cols:
            scope_cols.append(column)
    return data[scope_cols].drop_duplicates(ignore_index=True), dim_cols


def _build_br_marketing_driver_notice_new(
        client_code, intention, supplemental_data):
    """Build the query-scoped notice for excluded base/other drivers."""

    notice_intention = intention == "contribution" or (
        client_code == "USB" and intention == "source of change"
    )
    if not notice_intention:
        return ""
    business_drivers = (
        supplemental_data["business_driver"]
        .dropna()
    )
    if business_drivers.isin(["base", "other"]).any():
        return f"We only provide {intention} by marketing drivers. "
    return ""


def _build_br_fixtext_str_new(
        client_code, model_group_id, data, dim_cols, driver_tags, ner_filter,
        ner_res_dict, metric_df, metric_configs):
    """Build BR overall fix text and table from query-scoped AG/MG/M data."""

    fixtext = ""
    overall_table = pd.DataFrame()
    dimensions = [column for column in dim_cols if column in data.columns]
    dim_dict = {
        dimension: data[dimension].unique().tolist()
        for dimension in dimensions
    }
    uniq_dim_ner = 1
    uniq_dim_data = (
        data[dimensions].drop_duplicates().shape[0]
        if not data.empty and dimensions else 1
    )
    for dimension in dimensions:
        values = ner_filter.get(dimension, [])
        if len(values) > 1:
            uniq_dim_ner *= len(values)
    uniq_dim = max(uniq_dim_ner, uniq_dim_data)

    overall_fixtext = ""
    intention = ner_filter["intention"][0]
    _, tagging = readout_utils._get_level_tagging(ner_filter, client_code)
    custom_tagging_condition = readout_utils._custom_tagging_condition(
        ner_filter, tagging
    )
    if intention == "source of change":
        overall_filter = ner_filter.copy()
        overall_filter["main_metric"] = metric_df.loc[
            metric_df["metric"].str.contains(
                "sales", case=False, na=False
            ),
            "metric",
        ].unique().tolist()
        overall_fixtext, overall_table = readout_utils._overall_metric_w_driver(
            client_code,
            model_group_id,
            overall_filter,
            ner_res_dict,
            driver_tags,
            metric_df,
            dim_dict,
            metric_configs,
        )
        overall_fixtext = readout_utils.convert_display_format(overall_fixtext)
    elif (
        intention == "contribution"
        and not ner_filter.get("core_dimension", {})
        and custom_tagging_condition
    ):
        overall_fixtext, overall_table = readout_utils._overall_metric_w_driver(
            client_code,
            model_group_id,
            ner_filter,
            ner_res_dict,
            driver_tags,
            metric_df,
            dim_dict,
            metric_configs,
        )
    if uniq_dim < 4:
        fixtext += overall_fixtext
    return fixtext, overall_table


def _build_br_tables_new(context):
    """Render BR detail and aggregate data as display tables only."""

    prepared_detail = context["prepared_detail"]
    ner_dict = context["ner_dict"]
    pivot_table = pd.DataFrame()
    pivot_biz_table = pd.DataFrame()
    pivot_sort_col = ""

    if context["prepared_aggregate"] is not None:
        aggregate = context["prepared_aggregate"]
        pivot_biz_table, _ = readout_utils._get_granular_table(
            aggregate.data,
            context["agg_ner_dict"],
            context["metrics_order"],
            context["ner_filter"],
            context["rank_check"],
            context["sort_by_col"],
            context["metric_df"],
            context["client_code"],
            context["model_group_id"],
            context["metric_configs"],
        )
    if not prepared_detail.data.empty:
        pivot_table, pivot_sort_col = readout_utils._get_granular_table(
            prepared_detail.data,
            ner_dict,
            context["metrics_order"],
            context["ner_filter"],
            context["rank_check"],
            context["sort_by_col"],
            context["metric_df"],
            context["client_code"],
            context["model_group_id"],
            context["metric_configs"],
        )

    dimensions = ner_dict["dimension"]
    pivot_table = readout_utils._dynamic_top(
        dimensions,
        pivot_sort_col,
        pivot_table,
        context["metric_df"],
        context["rank_check"],
    )
    pivot_table = readout_utils._drop_single_dim(dimensions, pivot_table)
    pivot_biz_table = readout_utils._dynamic_top(
        dimensions,
        pivot_sort_col,
        pivot_biz_table,
        context["metric_df"],
        context["rank_check"],
    )
    pivot_biz_table = readout_utils._drop_single_dim(dimensions, pivot_biz_table)
    return pivot_table, pivot_biz_table, pivot_sort_col


def _process_br_detail_share_new(
        client_code, model_group_id, df_dict, driver_tags, ner_filter,
        ner_res_dict, map_dict, metric_df, metric_configs):
    """BR detail/share processing with independent text and table routes."""
    # Sales is an overall-only workflow and remains on the legacy path.
    if ner_filter["intention"][0] == "sales":
        return legacy_readout._process_br_detail_share(
            client_code,
            model_group_id,
            df_dict,
            driver_tags,
            ner_filter,
            ner_res_dict,
            map_dict,
            metric_df,
            metric_configs,
        )

    file_type, data_type, driver_high, driver_detail = driver_tags
    intention = ner_filter["intention"][0]
    trend_check = ner_res_dict.get("trend", "no")
    if intention == "source of change":
        trend_check = "no"
    rank_check = ner_res_dict.get("rank", "na")
    how_many = ner_res_dict.get("how_many", "na")
    period_type = (ner_filter.get("period_type") or ["year"])[0]
    sort_metric = map_dict[intention]["sortMetric"]
    metrics_order = ner_filter["metric"]
    core_dimension = ner_filter.get("core_dimension", {})
    level, _ = readout_utils._get_level_tagging(ner_filter, client_code)

    # Stage 1: filter and normalize the selected BR source level.
    curr_df = readout_utils._br_custom_agg_filter(
        core_dimension, df_dict[file_type + "_df"], ner_filter
    )
    if curr_df.empty:
        raise ValueError("No valid data.")
    dim_cols = readout_utils._get_dim_col_in_pivot_table_unique(
        df_dict, ner_filter, client_code
    )
    if not (client_code == "COLGUS" and model_group_id != 3041):
        data = readout_utils._read_and_process_csv(curr_df, ner_filter)
    else:
        data, dim_cols = readout_utils._read_and_process_csv_other_category(
            curr_df, ner_filter, dim_cols
        )
    data, dim_cols = readout_utils._level_to_drop(level, dim_cols, data)
    if "core_dimension_term" in data.columns:
        dim_cols += ["core_dimension_term"]
    data, driver_detail = readout_utils._concatenate_br(data, driver_detail)
    ner_dict = readout_utils._get_ner_key_value(
        data, driver_high, file_type, data_type, driver_detail, dim_cols
    )
    ner_dict["driver"] = driver_detail
    if data.empty:
        raise ValueError("No valid data.")

    # Apply BR-specific growth and driver filtering before splitting row types.
    data = readout_utils.quantize_metric_values(data, metric_configs)
    data = readout_utils._calc_growth_br(
        intention, ner_filter, period_type, ner_dict, data
    )
    growth_col, sort_by_col = readout_utils._get_growth_sort_col(
        trend_check, data
    )
    data = readout_utils._ignore_growth_outlier(growth_col, data)
    data = readout_utils._exclude_system(intention, data)
    data, trigger_pretext = readout_utils._filter_br_driver(client_code, intention, data)
    # Keep aggregate and detail rows separate for both downstream routes.
    if "custom_aggregated" in data.columns:
        agg_data = data[data["custom_aggregated"] == "yes"]
        data = data[data["custom_aggregated"] != "yes"]
    else:
        agg_data = pd.DataFrame()

    if not data.empty:
        data = readout_utils._sort_filt_data(
            data, ner_dict, sort_metric, sort_by_col, rank_check, metric_df
        )
    # contribution_notice_required = intention == "contribution" and (
    #     (trigger_pretext and not core_dimension)
    #     or data.empty
    #     or any(
    #         key in ner_filter.get("business_driver_captured", {})
    #         for key in ["base", "other"]
    #     )
    # )
    # readout_notice_required = (
    #     trigger_pretext or contribution_notice_required
    # )

    # Stage 2: prepare independent detail and aggregate views.
    agg_ner_dict = ner_dict.copy()
    prepared_detail, detail_ner_dict = _prepare_readout_data(
        client_code, model_group_id, data, ner_res_dict, ner_dict
    )
    prepared_aggregate = None
    prepared_agg_ner_dict = None
    if not agg_data.empty:
        agg_ner_dict["driver"] = "tactics_detail"
        agg_ner_dict["driver_detail"] = "tactics_detail"
        agg_data = readout_utils._sort_filt_data(
            agg_data,
            agg_ner_dict,
            sort_metric,
            sort_by_col,
            rank_check,
            metric_df,
        )
        prepared_aggregate, prepared_agg_ner_dict = _prepare_readout_data(
            client_code,
            model_group_id,
            agg_data,
            ner_res_dict,
            agg_ner_dict,
        )

    context = {
        "client_code": client_code,
        "model_group_id": model_group_id,
        "ner_filter": ner_filter,
        "ner_res_dict": ner_res_dict,
        "metric_df": metric_df,
        "metric_configs": metric_configs,
        "prepared_detail": prepared_detail,
        "ner_dict": detail_ner_dict,
        "prepared_aggregate": prepared_aggregate,
        "agg_ner_dict": prepared_agg_ner_dict,
        "intention": intention,
        "trend_check": trend_check,
        "rank_check": rank_check,
        "how_many": how_many,
        "sort_metric": sort_metric,
        "sort_by_col": sort_by_col,
        "growth_col": growth_col,
        "driver_high": driver_high,
        "metrics_order": metrics_order,
        # "readout_notice_required": readout_notice_required,
    }
    # Stage 3: render text and tables from the same prepared context.
    readout = _build_br_readout_str_new(context)
    pivot_table, pivot_biz_table, pivot_sort_col = _build_br_tables_new(context)

    return (
        readout,
        pivot_table,
        pd.DataFrame(),
        "",
        pd.DataFrame(),
        "",
        pivot_biz_table,
        pivot_sort_col,
        pd.DataFrame(),
    )


class PostprocessNew(legacy_readout.Postprocess):
    """Legacy share selection with decoupled text and table post-processing."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.function_bi_detail = _process_bi_detail_share_new
        self.function_br_detail = _process_br_detail_share_new

    @staticmethod
    def _serialize_readout_tables(agg_table, detail_table):
        """Serialize table copies without replacing the display tables."""
        agg_text, detail_text, _, _ = readout_utils.table_to_text(
            agg_table.copy(), detail_table.copy(), take_head=False
        )
        return agg_text, detail_text

    def _postprocess_bi_readout_text(
            self, planner_readout, client_code, model_group_id,
            ner_filter_dict, ner_res_dict, metric_df, metric_configs):
        """Apply BI post-processing that only contributes to readout text."""
        self.readout = readout_utils._add_na_term(self.core_dimension, self.readout)
        self.readout = readout_utils._metrics_aka(
            self.readout, self.ner_filter_dict
        )
        prompt_agg = self.overall_view if not self.overall_view.empty else self.agg_view
        agg_table_str, detail_table_str = self._serialize_readout_tables(
            prompt_agg, self.detail_view
        )

        if planner_readout:
            self.readout = planner_readout
        readout_adj = "LINKEDIN no table readout" in self.readout
        if (
            self.BI_readout_dict["ag"]["overall_ag_only"]
            and self.rank_check == "na"
            and not readout_adj
        ):
            instruction = readout_utils.load_prompt_instruction(
                self.client_code, self.model_group_id
            )
            if not instruction.empty:
                extra_inst_series = instruction.loc[
                    instruction["intentionName"] == self.intention, "instruction"
                ]
                extra_instruction = (
                    extra_inst_series.iloc[0] if not extra_inst_series.empty else ""
                )
                custom_inst, outlier = readout_utils._bi_custom_instruction(
                    self.intention,
                    self.time,
                    self.detail_view.copy(),
                    client_code,
                    model_group_id,
                    self.ner_filter_dict,
                    instruction,
                )
                overall_trend_str, self.pretext_table_trend = (
                    readout_utils._get_pretext_overall_trend(
                        client_code,
                        model_group_id,
                        ner_filter_dict,
                        ner_res_dict,
                        self.driver_tags["ag"],
                        metric_df,
                        self.detail_view.copy(),
                        metric_configs,
                    )
                )
                for text in [overall_trend_str, outlier]:
                    if text:
                        self.pretext += f"\n\n{text}"
                parts = []
                if isinstance(extra_instruction, str) and extra_instruction.strip():
                    parts.append(extra_instruction)
                if custom_inst.strip():
                    parts.append(custom_inst)
                parts.append(self.readout)
                self.readout = "\n".join(parts)
        elif not readout_adj:
            self.readout = readout_utils._table_readout_prompt(
                self.readout, agg_table_str, detail_table_str
            )
        else:
            self.readout = self.readout.replace(
                "LINKEDIN no table readout. ", ""
            )
        if self.intention == "spending" and self.readout:
            self.readout = (
                "Use passive tense when describing changes "
                "(e.g., 'was increased' / 'was reduced'). \n\n"
                + self.readout
            )
        return readout_adj

    def _postprocess_bi_output_tables(self):
        """Apply display-only transformations to BI output tables."""
        self.detail_view = readout_utils._table_month_transform(self.detail_view)
        self.agg_view = readout_utils._table_month_transform(self.agg_view)
        self.overall_view = readout_utils._table_month_transform(self.overall_view)
        self.detail_view.columns = [
            readout_utils._rename_change_columns(column)
            for column in self.detail_view.columns
        ]
        self.agg_view.columns = [
            readout_utils._rename_change_columns(column)
            for column in self.agg_view.columns
        ]
        self.overall_view.columns = [
            readout_utils._rename_change_columns(column)
            for column in self.overall_view.columns
        ]
        self.pretext_table_trend = readout_utils._table_month_transform(
            self.pretext_table_trend
        )

    def _process_br_share_new(
            self, client_code, model_group_id, df_dict, driver_tags,
            ner_filter_dict, ner_res_dict, map_dict, metric_df, metric_configs):
        """Run the existing BR share flow with separate final output routes."""
        for level in ["ag", "mg", "m"]:
            driver_tag = self.driver_tags_br[level]
            try:
                (
                    readout,
                    detail_df,
                    benchmark_data,
                    benchmark_str,
                    planner_table,
                    pretext,
                    agg_df,
                    pivot_sort_col,
                    overall_table,
                ) = self.function_br_detail(
                    client_code,
                    model_group_id,
                    df_dict,
                    driver_tag,
                    ner_filter_dict,
                    ner_res_dict,
                    map_dict,
                    metric_df,
                    metric_configs,
                )
                self.BR_readout_dict[level] = {
                    "readout": readout,
                    "detail": detail_df,
                    "benchmark_data": benchmark_data,
                    "benchmark_str": benchmark_str,
                    "planner_table": planner_table,
                    "pretext": pretext,
                    "agg": agg_df,
                    "pivot_sort_col": pivot_sort_col,
                    "overall_table": overall_table,
                }
            except Exception as error:
                legacy_readout.logger.warning(
                    f"Failed to process BR {level} results: {error}"
                )
                self.BR_readout_dict[level] = {
                    "readout": "",
                    "detail": pd.DataFrame(),
                    "benchmark_data": pd.DataFrame(),
                    "benchmark_str": "",
                    "planner_table": pd.DataFrame(),
                    "pretext": "",
                    "agg": pd.DataFrame(),
                    "pivot_sort_col": "",
                    "overall_table": pd.DataFrame(),
                }

        supplemental_data, supplemental_dim_cols = _prepare_br_supplemental_context_new(
            client_code,
            df_dict,
            ner_filter_dict,
        )
        self.time_message = readout_utils._validate_time_range(
            ner_res_dict,
            ner_filter_dict,
            supplemental_data,
        )
        marketing_driver_notice = _build_br_marketing_driver_notice_new(
            client_code,
            ner_filter_dict["intention"][0],
            supplemental_data,
        )
        overall_fixtext, self.pretext_table = _build_br_fixtext_str_new(
            client_code,
            model_group_id,
            supplemental_data,
            supplemental_dim_cols,
            driver_tags,
            ner_filter_dict,
            ner_res_dict,
            metric_df,
            metric_configs,
        )
        self.pretext = marketing_driver_notice + overall_fixtext

        if "agg_mg" in self.df_lookup_br.columns:
            df_ls = [
                self.BR_readout_dict["ag"]["detail"],
                self.BR_readout_dict["mg"]["agg"],
                self.BR_readout_dict["mg"]["detail"],
                self.BR_readout_dict["m"]["detail"],
            ]
            df_states = ["N" if frame.empty else "Y" for frame in df_ls]
            df_states.append(
                "Y"
                if self.BR_readout_dict["mg"].get("all_custom_agg")
                else "N"
            )
            state_count = 4
        else:
            df_ls = [
                self.BR_readout_dict["ag"]["detail"],
                self.BR_readout_dict["mg"]["detail"],
                self.BR_readout_dict["m"]["detail"],
            ]
            df_states = ["N" if frame.empty else "Y" for frame in df_ls]
            state_count = 3

        match = self.df_lookup_br[
            self.df_lookup_br["search_key"]
            == "~".join(df_states[:state_count])
        ]
        if len(match) > 1 and "all_custom_agg" in match:
            match = match[match["all_custom_agg"] == df_states[4]]
        if not match.empty:
            self.result_combo_match = (
                match.reset_index().iloc[0, :].fillna("").to_dict()
            )
            self.readout = readout_utils._get_readout_com(
                self.result_combo_match["readout"],
                self.BR_readout_dict,
                self.result_combo_match,
                self.data_type,
            )
            detail_key = self.result_combo_match["detail_view"].split("_")
            self.detail_view = self.BR_readout_dict.get(detail_key[-1], {}).get(
                detail_key[0], pd.DataFrame()
            )
            self.benchmark_data = pd.DataFrame()
            self.planner_data = pd.DataFrame()
            agg_key = self.result_combo_match["agg_view"].split("_")
            self.agg_view = self.BR_readout_dict.get(agg_key[-1], {}).get(
                agg_key[0], pd.DataFrame()
            )
            self.benchmark_str = ""
            pivot_level = self.result_combo_match["pivot_sort_col"].split("_")[-1]
            self.pivot_sort_col = self.BR_readout_dict.get(pivot_level, {}).get(
                "pivot_sort_col", ""
            )

        self.agg_view_copy = self.agg_view.copy()
        if "Tactics" in self.detail_view.columns:
            detail_level = self.result_combo_match["detail_view"].split("_")[-1]
            agg_level = self.result_combo_match["agg_view"].split("_")[-1]
            self.detail_view, self.agg_view = readout_utils.dedup_post(
                self.detail_view,
                self.agg_view_copy,
                self.df_dict,
                detail_level,
                agg_level,
                self.client_code,
                self.main_metric,
                type="br",
            )
            if (
                not self.agg_view.empty
                and len(self.agg_view) < len(self.agg_view_copy)
            ):
                self.agg_view = self.agg_view_copy
        if "Core_Dimension_Term" in self.detail_view.columns:
            self.dim_cols += ["Core_Dimension_Term"]

        self.detail_view, self.agg_view = readout_utils._br_table_reformat(
            self.detail_view, self.agg_view
        )
        if self.detail_view.equals(self.agg_view):
            self.agg_view = pd.DataFrame()
            detail_level = (
                self.result_combo_match["detail_view"].split("_")[-1]
                if self.result_combo_match.get("detail_view")
                else None
            )
            if detail_level:
                self.readout = readout_utils._get_readout_com(
                    "readout_" + detail_level,
                    self.BR_readout_dict,
                    self.result_combo_match,
                    self.data_type,
                )

        if df_ls[0].empty:
            biz_driver_check = ["base"]
        elif "Business_Driver" in df_ls[0].columns:
            biz_driver_check = df_ls[0]["Business_Driver"].unique().tolist()
        else:
            biz_driver_check = ["base"]

        self.detail_view = readout_utils.apply_level_mapping_display(
            self.client_code, self.model_group_id, self.detail_view
        )
        self.agg_view = readout_utils.apply_level_mapping_display(
            self.client_code, self.model_group_id, self.agg_view
        )
        self.detail_view = readout_utils.format_table_by_metric_config(
            self.detail_view, metric_configs
        )
        self.agg_view = readout_utils.format_table_by_metric_config(
            self.agg_view, metric_configs
        )

        readout_adj = self._postprocess_br_readout_text(biz_driver_check)
        self._postprocess_br_output_tables()
        self.result_combo_match["is_specific"] = bool(
            self.core_dimension
            and self.intention != "sales"
            and self.business_driver != "irrelevant"
        )
        return (
            self.readout,
            self.detail_view,
            self.benchmark_data,
            self.benchmark_str,
            self.planner_data,
            self.spend_share_principle,
            self.pretext,
            self.agg_view,
            self.pretext_table,
            self.pretext_table_trend,
            self.result_combo_match,
            "",
            pd.DataFrame(),
            readout_adj,
        )

    def _process_bi_share_new(
            self, client_code, model_group_id, df_dict, driver_tags,
            ner_filter_dict, ner_res_dict, map_dict, metric_df, metric_configs):
        """Run the existing BI share flow with separate final output routes."""
        for level in ["ag", "mg", "m"]:
            driver_tag = self.driver_tags[level]
            try:
                (
                    readout,
                    detail_df,
                    benchmark_data,
                    benchmark_str,
                    planner_table,
                    spend_share_principle,
                    pretext,
                    agg_df,
                    pivot_sort_col,
                    all_custom_agg,
                    overall_ag_only,
                    overall_table,
                    principle_pretext,
                    diag_df,
                    readout_diag,
                ) = self.function_bi_detail(
                    client_code,
                    model_group_id,
                    df_dict,
                    driver_tag,
                    ner_filter_dict,
                    ner_res_dict,
                    map_dict,
                    metric_df,
                    self.agg_view_df,
                    self.tactics_level,
                    metric_configs,
                )
                self.BI_readout_dict[level] = {
                    "readout": readout,
                    "detail": detail_df,
                    "benchmark_data": benchmark_data,
                    "benchmark_str": benchmark_str,
                    "planner_table": planner_table,
                    "spend_share_principle": spend_share_principle,
                    "pretext": pretext,
                    "agg": agg_df,
                    "pivot_sort_col": pivot_sort_col,
                    "all_custom_agg": all_custom_agg,
                    "overall_ag_only": overall_ag_only,
                    "overall_table": overall_table,
                    "principle_pretext": principle_pretext,
                    "diag": diag_df,
                    "readout_diag": readout_diag,
                }
                planner_readout = (
                    readout if readout and not planner_table.empty else ""
                )
            except Exception as error:
                legacy_readout.logger.warning(
                    f"Failed to process BI {level} results: {error}"
                )
                self.BI_readout_dict[level] = {
                    "readout": "",
                    "detail": pd.DataFrame(),
                    "benchmark_data": pd.DataFrame(),
                    "benchmark_str": "",
                    "planner_table": pd.DataFrame(),
                    "spend_share_principle": pd.DataFrame(),
                    "pretext": "",
                    "agg": pd.DataFrame(),
                    "pivot_sort_col": "",
                    "all_custom_agg": "",
                    "overall_ag_only": False,
                    "overall_table": pd.DataFrame(),
                    "principle_pretext": "",
                    "diag": pd.DataFrame(),
                    "readout_diag": "",
                }
                planner_readout = ""

        diag_level = next(
            (
                level
                for level in ["m", "mg", "ag"]
                if not self.BI_readout_dict[level]["diag"].empty
            ),
            None,
        )
        self.BI_readout_dict["diag"] = {
            "table": (
                self.BI_readout_dict[diag_level]["diag"]
                if diag_level
                else pd.DataFrame()
            ),
            "readout": (
                self.BI_readout_dict[diag_level]["readout_diag"]
                if diag_level
                else ""
            ),
        }

        supplemental_data, supplemental_dim_cols = _prepare_bi_supplemental_context_new(
            client_code,
            df_dict,
            ner_filter_dict,
        )
        self.pretext, self.pretext_table = _build_bi_fixtext_str_new(
            client_code,
            model_group_id,
            supplemental_data,
            supplemental_dim_cols,
            driver_tags,
            ner_filter_dict,
            ner_res_dict,
            metric_df,
            metric_configs,
        )
        self.time_message = readout_utils._validate_time_range(ner_res_dict, ner_filter_dict,supplemental_data)
        self.principle_pretext = _build_bi_principle_pretext_new(
            client_code,
            model_group_id,
            supplemental_data,
            supplemental_dim_cols,
            ner_filter_dict["intention"][0],
        )

        df_ls = [
            self.BI_readout_dict["ag"]["agg"],
            self.BI_readout_dict["ag"]["detail"],
            self.BI_readout_dict["mg"]["agg"],
            self.BI_readout_dict["mg"]["detail"],
            self.BI_readout_dict["m"]["agg"],
            self.BI_readout_dict["m"]["detail"],
        ]
        df_states = ["N" if frame.empty else "Y" for frame in df_ls]
        diag_state = (
            "Y"
            if ner_filter_dict.get("diag_tagging", [])
            and not self.BI_readout_dict["diag"]["table"].empty
            else "N"
        )
        all_custom_agg = (
            "Y" if self.BI_readout_dict["mg"]["all_custom_agg"] else "N"
        )
        lookup_states = df_states + (
            [diag_state] if self.bi_lookup_uses_diag else []
        )
        match = self.df_lookup_bi[
            self.df_lookup_bi["search_key"] == "~".join(lookup_states)
        ]
        if len(match) > 1:
            match = match[match["all_custom_agg"] == all_custom_agg]
        if not match.empty:
            self.result_combo_match = (
                match.reset_index().iloc[0, :].fillna("").to_dict()
            )
            self.readout = readout_utils._get_readout_com(
                self.result_combo_match["readout"],
                self.BI_readout_dict,
                self.result_combo_match,
                self.data_type,
            )
            detail_view = self.result_combo_match["detail_view"]
            if detail_view == "diag_df":
                self.detail_view = self.BI_readout_dict["diag"]["table"]
            else:
                detail_key = detail_view.split("_")
                self.detail_view = self.BI_readout_dict.get(
                    detail_key[-1], {}
                ).get(detail_key[0], pd.DataFrame())
            benchmark_level = self.result_combo_match["benchmark_data"].split("_")[-1]
            self.benchmark_data = self.BI_readout_dict.get(
                benchmark_level, {}
            ).get("benchmark_data", pd.DataFrame())
            planner_level = self.result_combo_match["planner_data"].split("_")[-1]
            self.planner_data = self.BI_readout_dict.get(planner_level, {}).get(
                "planner_table", pd.DataFrame()
            )
            self.spend_share_principle = self.BI_readout_dict.get(
                planner_level, {}
            ).get("spend_share_principle", pd.DataFrame())
            agg_view = self.result_combo_match["agg_view"]
            if agg_view == "diag_df":
                self.agg_view = self.BI_readout_dict["diag"]["table"]
            else:
                agg_key = agg_view.split("_")
                self.agg_view = self.BI_readout_dict.get(agg_key[-1], {}).get(
                    agg_key[0], pd.DataFrame()
                )
            benchmark_str_level = self.result_combo_match["benchmark_str"].split("_")[-1]
            self.benchmark_str = self.BI_readout_dict.get(
                benchmark_str_level, {}
            ).get("benchmark_str", "")
            pivot_level = self.result_combo_match["pivot_sort_col"].split("_")[-1]
            self.pivot_sort_col = self.BI_readout_dict.get(pivot_level, {}).get(
                "pivot_sort_col", ""
            )
            overall_view_value = self.result_combo_match.get("overall_view")
            if overall_view_value:
                overall_key = overall_view_value.split("_")
                self.overall_view = self.BI_readout_dict.get(
                    overall_key[-1], {}
                ).get(overall_key[0], pd.DataFrame())
            else:
                self.overall_view = pd.DataFrame()
            if (
                self.BI_readout_dict["ag"]["overall_ag_only"]
                and self.result_combo_match["readout_ag"] == "Y"
            ):
                self.readout = readout_utils._get_readout_com(
                    "readout_ag",
                    self.BI_readout_dict,
                    self.result_combo_match,
                    self.data_type,
                )
                self.agg_view = pd.DataFrame()
                self.overall_view = pd.DataFrame()
                self.detail_view = self.BI_readout_dict["ag"].get(
                    "detail", pd.DataFrame()
                )
                self.benchmark_data = self.BI_readout_dict["ag"].get(
                    "benchmark_data", pd.DataFrame()
                )
                self.benchmark_str = self.BI_readout_dict["ag"].get(
                    "benchmark_str", ""
                )

        detail_level = self.result_combo_match["detail_view"].split("_")[-1]
        agg_level = self.result_combo_match["agg_view"].split("_")[-1]
        self.detail_view, self.agg_view = readout_utils.dedup_post(
            self.detail_view,
            self.agg_view,
            self.df_dict,
            detail_level,
            agg_level,
            self.client_code,
            self.main_metric,
        )
        if not self.overall_view.empty:
            overall_level = self.result_combo_match["overall_view"].split("_")[-1]
            self.detail_view, self.overall_view = readout_utils.dedup_post(
                self.detail_view,
                self.overall_view,
                self.df_dict,
                detail_level,
                overall_level,
                self.client_code,
                self.main_metric,
            )
            self.agg_view, self.overall_view = readout_utils.dedup_post(
                self.agg_view,
                self.overall_view,
                self.df_dict,
                agg_level,
                overall_level,
                self.client_code,
                self.main_metric,
            )

        if "Core_Dimension_Term" in self.detail_view.columns:
            self.dim_cols += ["Core_Dimension_Term"]
        self.detail_view = readout_utils.apply_level_mapping_display(
            self.client_code, self.model_group_id, self.detail_view
        )
        self.agg_view = readout_utils.apply_level_mapping_display(
            self.client_code, self.model_group_id, self.agg_view
        )
        self.overall_view = readout_utils.apply_level_mapping_display(
            self.client_code, self.model_group_id, self.overall_view
        )
        self.planner_data = readout_utils.apply_level_mapping_display(
            self.client_code, self.model_group_id, self.planner_data
        )
        self.detail_view = readout_utils.format_table_by_metric_config(
            self.detail_view, metric_configs
        )
        self.agg_view = readout_utils.format_table_by_metric_config(
            self.agg_view, metric_configs
        )
        self.overall_view = readout_utils.format_table_by_metric_config(
            self.overall_view, metric_configs
        )

        readout_adj = self._postprocess_bi_readout_text(
            planner_readout,
            client_code,
            model_group_id,
            ner_filter_dict,
            ner_res_dict,
            metric_df,
            metric_configs,
        )
        self._postprocess_bi_output_tables()
        self.result_combo_match["is_specific"] = not self.BI_readout_dict["ag"][
            "overall_ag_only"
        ]
        return (
            self.readout,
            self.detail_view,
            self.benchmark_data,
            self.benchmark_str,
            self.planner_data,
            self.spend_share_principle,
            self.pretext,
            self.agg_view,
            self.pretext_table,
            self.pretext_table_trend,
            self.result_combo_match,
            self.principle_pretext,
            self.overall_view,
            readout_adj,
        )

    def _postprocess_br_readout_text(self, biz_driver_check):
        """Apply BR post-processing that only contributes to readout text."""
        self.readout = readout_utils._add_na_term(self.core_dimension, self.readout)
        self.readout = readout_utils._metrics_aka(
            self.readout, self.ner_filter_dict
        )
        agg_table_str, detail_table_str = self._serialize_readout_tables(
            self.agg_view, self.detail_view
        )
        readout_adj = "LINKEDIN no table readout" in self.readout
        if (
            "base" not in biz_driver_check
            and "other" not in biz_driver_check
            and not self.core_dimension
            and self.rank_check == "na"
            and not readout_adj
        ):
            if self.intention in ["source of change", "sales"]:
                self.readout = readout_utils._table_readout_prompt(
                    self.readout, agg_table_str, detail_table_str
                )
            instruction = readout_utils.load_prompt_instruction(
                self.client_code, self.model_group_id
            )
            if not instruction.empty:
                extra_instruction = instruction.loc[
                    instruction["intentionName"] == self.intention,
                    "instruction",
                ].values[0]
                custom_inst, outlier = readout_utils._br_custom_instruction(
                    self.intention,
                    self.time,
                    self.detail_view.copy(),
                    self.ner_filter_dict,
                )
                if outlier:
                    self.pretext += "\n\n" + outlier
                parts = []
                if isinstance(extra_instruction, str) and extra_instruction.strip():
                    parts.append(extra_instruction)
                if custom_inst.strip():
                    parts.append(custom_inst)
                parts.append(self.readout)
                self.readout = "\n".join(parts)
        elif not readout_adj:
            self.readout = readout_utils._table_readout_prompt(
                self.readout, agg_table_str, detail_table_str
            )
        else:
            self.readout = self.readout.replace(
                "LINKEDIN no table readout. ", ""
            )

        check_driver = any(
            not frame.empty
            and "business_driver" in frame.columns
            and frame["business_driver"].isin(["base", "other"]).any()
            for frame in [
                self.df_dict.get("ag_df"),
                self.df_dict.get("mg_df"),
                self.df_dict.get("m_df"),
            ]
            if frame is not None
        )
        if check_driver and self.intention == "contribution":
            self.readout = (
                "In our business context, 'Base' refers to Base Drivers—factors "
                "such as brand loyalty, pricing, distribution, and seasonality. "
                "These represent the underlying baseline performance of the "
                "business, against which we measure the true incremental impact "
                "of marketing campaigns.\n"
                + self.readout
            )
        return readout_adj

    def _postprocess_br_output_tables(self):
        """Apply display-only transformations to BR output tables."""
        self.detail_view.columns = [
            readout_utils._rename_change_columns(column)
            for column in self.detail_view.columns
        ]
        self.agg_view.columns = [
            readout_utils._rename_change_columns(column)
            for column in self.agg_view.columns
        ]
        self.detail_view = readout_utils._table_month_transform(self.detail_view)
        self.agg_view = readout_utils._table_month_transform(self.agg_view)
        self.pretext_table_trend = readout_utils._table_month_transform(
            self.pretext_table_trend
        )
