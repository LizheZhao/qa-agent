import logging
from typing import Dict, Optional
from dataclasses import dataclass, field

import pandas as pd
from pandas import DataFrame

from src.data.local import (get_cached_readout_data, get_client_data, get_client_data_config, get_learning_examples,
                            get_questionnaire, get_spacy_config, get_process_indicator, get_ner_postprocess_indicator,
                            get_term_variants, get_insights_report_trigger_config, get_feasibility_config)


logger = logging.getLogger(__name__)


@dataclass
class ProcessIndicator:
    legacy_preprocessing_enabled: bool = field(default_factory=bool)
    spacy_enabled: bool = field(default_factory=bool)
    hallucination_alerts_enabled: bool = field(default_factory=bool)

    @classmethod
    def from_local(cls, client_code: str, model_group_id: int):
        indicators = get_process_indicator(client_code, model_group_id)
        instance = cls(
            legacy_preprocessing_enabled=indicators.get('legacyPreprocessingEnabled', True),
            spacy_enabled=indicators.get('spacyEnabled', True),
            hallucination_alerts_enabled=indicators.get('hallucinationAlertsEnabled', False),
        )

        return instance

    @classmethod
    def from_database(cls, client_code: str, model_group_id: int):
        ...

    def validate(self):
        ...


@dataclass
class NerData:
    df_bi: DataFrame = field(default_factory=pd.DataFrame)
    df_br: DataFrame = field(default_factory=pd.DataFrame)
    df_examples: DataFrame = field(default_factory=pd.DataFrame)
    questionnaire: Dict[str, any] = field(default_factory=dict)
    level_type: Dict[str, list] = field(default_factory=dict)

    @classmethod
    def from_local(cls, client_code: str, model_group_id: int):
        df_bi, df_br = get_client_data(client_code, model_group_id)
        questionnaire, level_type = get_questionnaire(client_code, model_group_id)
        instance = cls(
            df_bi=df_bi,
            df_br=df_br,
            df_examples=get_learning_examples(client_code, model_group_id),
            questionnaire=questionnaire,
            level_type=level_type,
        )
        instance.validate()

        return instance

    @classmethod
    def from_database(cls) -> None:
        ...

    def validate(self) -> None:
        validate_columns(self.df_examples, ["query", "intention"])
        validate_keys(list(self.questionnaire.keys()), ["classification"])

        logger.debug("NerData Validated")


@dataclass
class NerDataConfig:
    metric_info: Dict[str, any] = field(default_factory=dict)
    data_levels: Dict[str, list] = field(default_factory=dict)
    acronym_dict: Optional[Dict[str, str]] = field(default_factory=dict)
    ner_postprocess_indicators: Optional[Dict[str, any]] = field(default_factory=dict)

    @classmethod
    def from_local(cls, client_code: str, model_group_id: int):
        metric_info, data_levels, acronym_dict = get_client_data_config(client_code, model_group_id)
        postprocess_indicators = get_ner_postprocess_indicator(client_code, model_group_id)
        instance = cls(
            metric_info=metric_info,
            data_levels=data_levels,
            acronym_dict=acronym_dict,
            ner_postprocess_indicators=postprocess_indicators
        )
        instance.validate()

        return instance

    @classmethod
    def from_database(cls, client_code: str, model_group_id: int):
        ...

    def validate(self) -> None:
        for _, v in self.metric_info.items():
            validate_keys(
                list(v.keys()),
                ["metric", "data", "rankMetric", "sortMetric", "mainMetric"]
            )

        validate_keys(
            list(self.data_levels.keys()),
            ["biLevels", "brLevels", "brLevelsToIgnore"]
        )

        logger.debug("NerDataConfig Validated")


@dataclass
class SpacyConfig:
    core_filters: Dict[str, any] = field(default_factory=dict)
    core_dimensions: list[Dict[str, any]] = field(default_factory=list)
    variant_mapping: DataFrame = field(default_factory=pd.DataFrame)

    @classmethod
    def from_local(cls, client_code: str, model_group_id: int):
        core_filters, core_dimensions = get_spacy_config(client_code, model_group_id)
        variant_mapping = get_term_variants(client_code, model_group_id)
        instance = cls(
            core_filters=core_filters,
            core_dimensions=core_dimensions,
            variant_mapping=variant_mapping
        )
        instance.validate()

        return instance

    def validate(self) -> None:
        # validate_keys(list(self.core_filters.keys()), ['intention'])
        # validate_keys(list(self.core_dimensions.keys()), ['intention'])

        logger.debug("SpacyConfig Validated")


@dataclass
class NerFilters:
    query: str = field(default_factory=str)
    ner_filter: Dict[str, any] = field(default_factory=dict)
    ner_res: Dict[str, list] = field(default_factory=dict)

    def validate(self) -> None:
        validate_keys(list(self.ner_filter.keys()), ['intention'])
        validate_keys(list(self.ner_res.keys()), ['intention'])

        logger.debug("NerFilters Validated")


@dataclass
class ReadoutData:
    """
    ner_filters.keys():
    [
        'intention', 'programmatic', 'media_channel', 'trend', 'rank', 'how_many', 'time', 'period_type', 'product',
        'data', 'metric', 'main_metric', 'business_driver', 'core_dimension', 'core_dimension_composite'
    ]
    ner_results:
    {
        'intention': ['performance'],
        'time': 'irrelevant',
        'period_type': 'irrelevant',
        'product': 'irrelevant',
        ...
    }
    """
    ner_filters: Dict = field(default_factory=dict)
    ner_results: Dict = field(default_factory=dict)
    df_activity_group: DataFrame = field(default_factory=pd.DataFrame)
    df_measure_group: DataFrame = field(default_factory=pd.DataFrame)
    df_measure: DataFrame = field(default_factory=pd.DataFrame)

    @classmethod
    def from_local(cls):
        instance = cls(
            ner_filters=get_cached_readout_data("ner_filter"),
            ner_results=get_cached_readout_data("ner_res"),
            df_activity_group=get_cached_readout_data("activity_group"),
            df_measure_group=get_cached_readout_data("measure_group"),
            df_measure=get_cached_readout_data("measure")
        )
        return instance

    def validate(self) -> None:
        ...


@dataclass
class InsightsReportConfig:
    triggers: Dict[str, list] = field(default_factory=dict)

    @classmethod
    def from_local(cls, client_code: str, model_group_id: int):
        triggers = get_insights_report_trigger_config(client_code, model_group_id)
        instance = cls(
            triggers=triggers,
        )
        instance.validate()

        return instance

    @classmethod
    def from_database(cls, client_code: str, model_group_id: int):
        ...

    def validate(self) -> None:
        ...


@dataclass
class FeasibilityConfig:
    """Offline-built coverage/feasibility artifact (per data_source: metrics,
    metric_time_ranges, dimension marginal coverage). See scripts/build_feasibility_config.py."""
    data_sources: Dict[str, any] = field(default_factory=dict)
    built_at: str = field(default_factory=str)

    @classmethod
    def from_local(cls, client_code: str, model_group_id: int):
        config = get_feasibility_config(client_code, model_group_id)
        instance = cls(
            data_sources=config.get("data_sources", {}),
            built_at=config.get("built_at", ""),
        )
        instance.validate()

        return instance

    @classmethod
    def from_database(cls, client_code: str, model_group_id: int):
        ...

    def validate(self) -> None:
        logger.debug("FeasibilityConfig Validated")


def validate_columns(df: DataFrame, expected_columns: list) -> None:
    if not set(expected_columns).issubset(df.columns):
        raise ValueError(f"DataFrame missing expected columns: {expected_columns}")


def validate_keys(keys: list, expected_keys: list) -> None:
    if not set(expected_keys).issubset(keys):
        raise ValueError(f"Dictionary missing expected keys: {','.join([x for x in expected_keys if x not in keys])}")