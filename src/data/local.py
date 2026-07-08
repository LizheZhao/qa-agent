import os
import pandas as pd
import json
from collections import defaultdict
from typing import Optional, Union
import logging

logger = logging.getLogger(__name__)


def get_client_data_dir_path():
    path = os.path.join(
        os.getenv("DATA_DIR"),
        os.getenv("CLIENT_CODE"),
        os.getenv("MODEL_GROUP_ID")
    )

    return path


def get_data_path(key: str, override_filename: Optional[str] = None) -> str:
    client_data_dir_path = get_client_data_dir_path()

    # TBD: standardize the filenames
    key_to_filename_map = {
        "map_dict": "map_dict.csv",
        "core_dimensions": "spacy_core_dimensions.jsonl",
        "core_dimensions_filter": "spacy_core_dimension_filter.json",
        "kg_bi": "askg_comprehensive_bi_cleaned_withoutmonthroi.csv",
        "total": "total.csv",
        "planner": "planner.csv",
        "benchmark": "benchmark.csv",
        "dim_returns_curve": "planner_response_curve.csv",
        "scenario_summary": "planner_scenario_summary.csv",
        "feasibility_config": "feasibility_config.json",
        "insight_instruction": "insight_instruction.csv",
    }

    if override_filename is None:
        filename = key_to_filename_map[key]
    else:
        filename = override_filename

    path = os.path.join(client_data_dir_path, filename)

    return path


def read_json(path) -> dict:
    with open(path, "r") as file:
        data = json.load(file)

    return data


def get_cached_readout_data(key: str) -> Union[pd.DataFrame, dict]:
    client_data_dir_path = get_client_data_dir_path()
    cache_dir_path = os.path.join(client_data_dir_path, "ReadoutDataCache")

    key_to_filename_map = {
        "ner_filter": ("ner_filter", "json"),
        "ner_res": ("ner_res", "json"),
        "activity_group": ("activity_group", "csv"),
        "measure_group": ("measure_group", "csv"),
        "measure": ("measure", "csv"),
    }

    filename, file_type = key_to_filename_map[key]

    if file_type == "json":
        data = read_json(os.path.join(cache_dir_path, f"{filename}.{file_type}"))
    elif file_type == "csv":
        data = pd.read_csv(os.path.join(cache_dir_path, f"{filename}.{file_type}"))
    else:
        raise NotImplementedError(f"File type {file_type} is not supported.")

    return data


def get_client_data(client_code: str, model_group_id: int):
    data_path = get_client_data_dir_path()
    df_bi = pd.read_csv(os.path.join(data_path, 'bi.csv'), low_memory=False)
    df_br = pd.read_csv(os.path.join(data_path, 'br.csv'), low_memory=False)

    return df_bi, df_br


def get_client_data_config(client_code: str, model_group_id: int) -> tuple[dict, dict, dict]:
    data_path = get_client_data_dir_path()
    df_query_intentions = pd.read_csv(os.path.join(data_path, 'map_dict.csv'), low_memory=False)
    df_data_levels = pd.read_csv(os.path.join(os.getenv('DATA_DIR'), 'data_levels.csv'), low_memory=False)

    metric_info = df_query_intentions.set_index('intentionName').rename(
        columns={"relatedKPIs": "metric", "dataSource": "data"}
    ).T.to_dict()
    for _, v in metric_info.items():
        v['metric'] = json.loads(v['metric'])
        v['mainMetric'] = [x.strip() for x in v['mainMetric'].split(",")]
        v['sortMetric'] = [x.strip() for x in v['sortMetric'].split(",")]

    df_data_levels = df_data_levels[
        (df_data_levels.clientCode == client_code) & (df_data_levels.modelGroupId == model_group_id)
    ]
    if len(df_data_levels) > 0:
        data_levels = df_data_levels[[
            'biLevels', 'brLevels', 'brLevelsToIgnore', 'spacyOverrideColumns']].to_dict('records')[0]
    else:
        data_levels = dict()
    for col in ['biLevels', 'brLevels', 'brLevelsToIgnore', 'spacyOverrideColumns']:
        if col == 'biLevels':
            default_list = '["activity_group", "measure_group", "measure"]'
        elif col == 'brLevels':
            default_list = '["business_driver", "business_driver_detail", "activity_group"]'
        elif col == 'spacyOverrideColumns':
            default_list = '["activity_group", "measure_group", "measure", "business_driver", "business_driver_detail"]'
        else:
            default_list = '[]'
        data_levels[col] = json.loads(data_levels.get(col, str(default_list)))

    acronym_mapping_file = os.path.join(os.getenv('DATA_DIR'), 'acronym_mapping.csv')

    if os.path.exists(acronym_mapping_file):
        acronym_df = pd.read_csv(acronym_mapping_file, low_memory=False)
        acronym_df = acronym_df[acronym_df.client.isin(["common", client_code.lower()])]
        acronym_mapping = acronym_df.set_index("acronym")["definition"].to_dict()
    else:
        logger.info(f"Couldn't find acronym file for {client_code} {model_group_id}")
        acronym_mapping = dict()
    return metric_info, data_levels, acronym_mapping


def get_ner_postprocess_indicator(client_code: str, model_group_id: int) -> dict:
    client_data_dir_path = get_client_data_dir_path()
    file_path = os.path.join(client_data_dir_path, 'ner_postprocess_indicator.csv')
    if os.path.exists(file_path):
        indicator_df = pd.read_csv(file_path, low_memory=False)
        indicator_df = indicator_df[indicator_df['enabled'] == 1]
        ner_process_indicator = indicator_df.set_index('indicator')['indicator_value'].to_dict()
        to_list = ["period_priority", "non_trend_report_length", "trend_report_length"]
        to_bool = ["report_yoy"]
        for indicator_type in to_list:
            if indicator_type in ner_process_indicator:
                ner_process_indicator[indicator_type] = \
                    [x.strip() for x in ner_process_indicator[indicator_type].split(',')]
        for indicator_type in to_bool:
            if indicator_type in ner_process_indicator:
                ner_process_indicator[indicator_type] = bool(ner_process_indicator.get(indicator_type, 0))
    else:
        logger.info(f"Couldn't find ner postprocess indicators for {client_code} {model_group_id}")
        ner_process_indicator = dict()
    return ner_process_indicator


def get_questionnaire(client_code: str, model_group_id: int) -> tuple[dict, dict]:
    """
    get ner questionnaire config
    :param client_code:
    :param model_group_id:
    :return: questionnaire dict
    """
    data_path = get_client_data_dir_path()
    questionnaire_df = pd.read_csv(os.path.join(data_path, 'questionnaire.csv'), low_memory=False)
    questionnaire = questionnaire_df.fillna('').set_index('dimension')[['field', 'context', 'prompt']].T.to_dict()
    for _, v in questionnaire.items():
        v['field'] = [x.strip() for x in v['field'].split(",")]

    if 'level' not in questionnaire_df.columns:
        questionnaire_df['level'] = ''
    else:
        questionnaire_df['level'].fillna('', inplace=True)
    level_type_df = questionnaire_df[questionnaire_df['level'] != ''][['field', 'level']]
    # level_type = level_type_df.groupby('level_type').agg(list)['field'].to_dict()
    level_type = defaultdict(list)
    for i, row in level_type_df.iterrows():
        level = [x.strip() for x in row['level'].split(',')]
        field = [x.strip() for x in row['field'].split(',')]
        for lvl, f in zip(level, field):
            level_type[lvl].append(f)
    return questionnaire, level_type


def get_learning_examples(client_code: str, model_group_id: int) -> pd.DataFrame:
    """
    get in context learning examples
    :param client_code:
    :param model_group_id:
    :return: dataframe
    """
    data_path = get_client_data_dir_path()
    file_path = os.path.join(data_path, 'in_context_examples.csv')
    if os.path.exists(file_path):
        df_examples = pd.read_csv(file_path, low_memory=False)
    else:
        logger.info(f"Couldn't find in-context examples for {client_code} {model_group_id}")
        df_examples = pd.DataFrame()
    return df_examples


def get_spacy_config(client_code: str, model_group_id: int) -> tuple[dict, list[dict]]:
    """
    get spacy files
    :param client_code:
    :param model_group_id:
    :return:
    """
    data_path = get_client_data_dir_path()
    core_dimension_file = os.path.join(data_path, 'core_dimensions.jsonl')
    core_dimension_filter_file = os.path.join(data_path, 'core_dimension_filter.json')
    if os.path.exists(core_dimension_file) and os.path.exists(core_dimension_filter_file):
        with open(core_dimension_filter_file, 'r') as file:
            core_dimension_filter = json.load(file)
        core_dimensions = pd.read_json(path_or_buf=core_dimension_file, lines=True).to_dict('records')
    else:
        logger.info(f"Couldn't find spacy files for {client_code} {model_group_id}")
        core_dimension_filter = dict()
        core_dimensions = []
    return core_dimension_filter, core_dimensions


def get_term_variants(client_code: str, model_group_id: int) -> pd.DataFrame:
    """
    get term variants
    """
    variant_file = os.path.join(os.getenv('DATA_DIR'), 'variants.csv')

    if os.path.exists(variant_file):
        variants = pd.read_csv(variant_file)
        variants = variants[variants.client.isin(["common", client_code.lower()])]
    else:
        logger.info(f"Couldn't find term variant files for {client_code} {model_group_id}")
        variants = pd.DataFrame(columns=["term", "variants", "client"])
    return variants


def get_process_indicator(client_code: str, model_group_id: int) -> dict:
    """
    get process indicators
    :param client_code:
    :param model_group_id:
    :return:
    """
    indicator_df = pd.read_csv(os.path.join(os.getenv('DATA_DIR'), 'categoryFunctions.csv'))
    indicator = indicator_df[
        (indicator_df.clientCode == client_code) & (indicator_df.modelgroupId == model_group_id)
    ].set_index(['clientCode', 'modelgroupId']).astype(bool).to_dict('records')
    if indicator:
        indicator_dict = indicator[0]
    else:
        logger.info(f"Couldn't find process indicators for {client_code} {model_group_id}")
        indicator_dict = {}
    return indicator_dict


def save_spacy_json(spacy_filters: dict) -> None:
    save_path = get_client_data_dir_path()
    file_name = 'core_dimension_filter.json'
    # Writing JSON data
    with open(os.path.join(save_path, file_name), 'w') as file:
        json.dump(spacy_filters, file)
    logger.info(f"core dimension filter saved to {save_path}")


def save_spacy_jsonl(spacy_terms: list) -> None:
    save_path = get_client_data_dir_path()
    file_name = 'core_dimensions.jsonl'
    # Writing JSON data
    with open(os.path.join(save_path, file_name), 'w') as f:
        for record in spacy_terms:
            json_record = json.dumps(record)
            f.write(json_record + '\n')
    logger.info(f"core dimension list saved to {save_path}")


def get_feasibility_config(client_code: str, model_group_id: int) -> dict:
    """Load the offline-built feasibility/coverage config; empty dict if absent."""
    file_path = get_data_path("feasibility_config")

    if not os.path.isfile(file_path):
        logger.info(f"Couldn't find feasibility config for {client_code} {model_group_id}")
        return dict()

    return read_json(file_path)


def get_insights_report_trigger_config(client_code: str, model_group_id: int) -> dict:
    data_path = get_client_data_dir_path()
    file_path = os.path.join(data_path, 'client_deck_intention_keywords.csv')

    if not os.path.isfile(file_path):
        return dict()

    df = pd.read_csv(file_path)
    df['keywords'] = df['keywords'].fillna("").apply(lambda x: [kw.strip() for kw in x.split(',')])
    intention_keywords = dict(zip([None if pd.isna(k) else k for k in df['intention']], df['keywords']))

    return intention_keywords