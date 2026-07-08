"""
chart package — re-exports all public chart functions so callers can use
either `from chart.roi import ...` or the legacy `import st_utils as utils`.
"""

from chart.common import (
    get_cached_insight,
    generate_chart_insights_parallel_cached,
    render_chart_insight_box,
    get_chart_set_inst,
    get_insights_set_inst,
    _section,
    table_element,
)

from chart.kpi import (
    metric_card,
    most_recent_kpi_card_new,
    get_pretext_trend_new,
)

from chart.roi import (
    build_roi_snapshot_chart_configs,
    build_roi_snapshot_chart_configs_for_insights,
    render_roi_snapshot_charts,
    get_roi_recent_plot_new,
    build_roi_trend_chart_configs,
    build_roi_trend_chart_configs_for_insights,
    render_roi_trend_charts,
    get_roi_trend_plot_new_multi_roi,
    build_roi_response_cost_chart_configs,
    build_roi_response_cost_chart_configs_for_insights,
    render_roi_response_cost_charts,
    get_roi_response_cost_plot_new,
    get_multiple_roi_plot_new,
    get_nivo_grouped_bar_data,
    build_level_stats,
    render_roi_segmentation_summary,
    multiple_roi_stacked_bar,
    fmt_percent,
)

from chart.spending import (
    build_spend_snapshot_chart_configs,
    build_spend_snapshot_chart_configs_for_insights,
    render_spend_snapshot,
    spend_snapshot,
    build_spend_trend_chart_configs,
    build_spend_trend_chart_configs_for_insights,
    render_spend_trend,
    spend_share_trend_by_level,
    build_spend_heatmap_chart_configs,
    render_spend_heatmap,
    pct_change_heatmap,
)

from chart.other_bi import (
    build_other_bi_snapshot_chart_configs,
    render_other_bi_snapshot,
    other_bi_snapshot,
    build_other_bi_trend_chart_configs,
    render_other_bi_trend,
    other_bi_trend_by_level,
)

from chart.breakdown import (
    build_sourceofchange_chart_configs,
    build_sourceofchange_chart_configs_for_insights,
    render_sourceofchange,
    get_plot_for_sourceofchange,
    build_contribution_snapshot_chart_configs,
    render_contribution_snapshot,
    get_contribution_pie,
    build_contribution_trend_chart_configs,
    render_contribution_trend,
    contribution_trend,
    build_contribution_trend_chart_configs_for_insights,
    build_contribution_snapshot_chart_configs_for_insights,
    _sales_trend_data_for_insights,
)

from chart.planner import (
    planner_overall_metric,
    planner_spend_change_all,
    planner_spend_change_all_instruct,
    planner_pct_bar,
    planner_pct_bar_instruct,
    get_spend_share_principle_plot,
    get_planner_spend_level_plot,
    spend_kpi_scatter,
    spend_kpi_scatter_insight,
    plot_diminishing_return_by_spend_level,
    plot_diminishing_return_instruct,
    spend_kpi_scatter_insight_for_insights,
    plot_diminishing_return_instruct_for_insights,
    planner_spend_change_all_instruct_for_insights,
    planner_pct_bar_instruct_for_insights,

)
