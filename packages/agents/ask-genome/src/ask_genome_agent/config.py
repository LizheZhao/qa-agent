"""Loads Ask Genome client configuration lazily from:
1. ASK_GENOME_CLIENT_CODE, ASK_GENOME_MODEL_GROUP_ID, ASK_GENOME_DATA_DIR
2. Configuration is read once and cached for the lifetime of the process, so each deployment
   currently supports one client.
3. The ASK_GENOME_ prefix avoids conflict with the application's existing CLIENT_CODE, which is
   used for LLM gateway routing.
Reusing the same variable previously caused incorrect gateway routing.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import pandas as pd

from ask_genome_agent.local import (
    get_client_data,
    get_client_data_config,
    get_feasibility_config,
    get_learning_examples,
    get_metric_configs,
    get_ner_postprocess_indicator,
    get_questionnaire,
    get_spacy_config,
    get_term_variants,
)


def validate_keys(keys: list[str], expected_keys: list[str]) -> None:
    """Ported verbatim from ask-genome-core/src/data/data_interface.py."""

    missing = [key for key in expected_keys if key not in keys]
    if missing:
        raise ValueError(f"Dictionary missing expected keys: {','.join(missing)}")


@dataclass
class NerDataConfig:
    """configuration needed for the intent catalog. Only ``metric_info`` is used in this package.
    The rest (data_levels, acronym_dict, ner_postprocess_indicators) belongs to Question
    Understanding, out of scope for now.
    """

    metric_info: dict[str, Any] = field(default_factory=dict)
    data_levels: dict[str, list[str]] = field(default_factory=dict)
    acronym_dict: dict[str, str] | None = field(default_factory=dict)
    ner_postprocess_indicators: dict[str, Any] | None = field(default_factory=dict)

    @classmethod
    def from_local(cls, client_code: str, model_group_id: int) -> NerDataConfig:
        metric_info, data_levels, acronym_dict = get_client_data_config(client_code, model_group_id)
        postprocess_indicators = get_ner_postprocess_indicator(client_code, model_group_id)
        instance = cls(
            metric_info=metric_info,
            data_levels=data_levels,
            acronym_dict=acronym_dict,
            ner_postprocess_indicators=postprocess_indicators,
        )
        instance.validate()
        return instance

    def validate(self) -> None:
        for _, metric in self.metric_info.items():
            validate_keys(
                list(metric.keys()), ["metric", "data", "rankMetric", "sortMetric", "mainMetric"]
            )
        validate_keys(list(self.data_levels.keys()), ["biLevels", "brLevels", "brLevelsToIgnore"])


@dataclass
class SpacyConfig:
    """Ported verbatim from ask-genome-core/src/data/data_interface.py. configuration used for
    spaCy-based processing.
    """

    core_filters: dict[str, Any] = field(default_factory=dict)
    core_dimensions: list[dict[str, Any]] = field(default_factory=list)
    variant_mapping: pd.DataFrame = field(default_factory=pd.DataFrame)

    @classmethod
    def from_local(cls, client_code: str, model_group_id: int) -> SpacyConfig:
        core_filters, core_dimensions = get_spacy_config(client_code, model_group_id)
        variant_mapping = get_term_variants(client_code, model_group_id)
        instance = cls(
            core_filters=core_filters,
            core_dimensions=core_dimensions,
            variant_mapping=variant_mapping,
        )
        instance.validate()
        return instance

    def validate(self) -> None:
        """The source's own validate is a no-op here too (its key checks are commented out)."""


@dataclass
class FeasibilityConfig:
    """Offline-built coverage/feasibility artifact (per data_source: metrics, metric_time_ranges,
    dimension marginal coverage). Ported verbatim from
    ask-genome-core/src/data/data_interface.py and look at
    ask-genome-core/scripts/build_feasibility_config.py for how data_sources is actually built.
    """

    data_sources: dict[str, Any] = field(default_factory=dict)
    built_at: str = ""

    @classmethod
    def from_local(cls, client_code: str, model_group_id: int) -> FeasibilityConfig:
        config = get_feasibility_config(client_code, model_group_id)
        instance = cls(
            data_sources=config.get("data_sources", {}), built_at=config.get("built_at", "")
        )
        instance.validate()
        return instance

    def validate(self) -> None:
        """The source's own validate is just a debug log; nothing to check here."""


@dataclass
class NerData:
    """Ported from ask-genome-core/src/data/data_interface.py:NerData.

    `metric_configs` is not on the source's NerData; there it is loaded separately per readout
    call. It is carried here so the data-filtering nodes read one cached object rather than
    re-reading CSVs on every subquery.
    """

    df_bi: pd.DataFrame = field(default_factory=pd.DataFrame)
    df_br: pd.DataFrame = field(default_factory=pd.DataFrame)
    df_examples: pd.DataFrame = field(default_factory=pd.DataFrame)
    metric_configs: pd.DataFrame = field(default_factory=pd.DataFrame)
    questionnaire: dict[str, Any] = field(default_factory=dict)
    level_type: dict[str, list[str]] = field(default_factory=dict)

    @classmethod
    def from_local(cls, client_code: str, model_group_id: int) -> NerData:
        questionnaire, level_type = get_questionnaire(client_code, model_group_id)
        df_bi, df_br = get_client_data(client_code, model_group_id)
        instance = cls(
            df_bi=df_bi,
            df_br=df_br,
            df_examples=get_learning_examples(client_code, model_group_id),
            metric_configs=get_metric_configs(client_code, model_group_id),
            questionnaire=questionnaire,
            level_type=level_type,
        )
        instance.validate()
        return instance

    def validate(self) -> None:
        validate_keys(list(self.df_examples.columns), ["query", "intention"])
        validate_keys(list(self.questionnaire.keys()), ["classification"])

    @property
    def custom_cols(self) -> list[str]:
        """The client's custom level/tagging columns.

        Deliberately `general_tagging + general_level` only, matching the source's `custom_cols`
        (filter_generator.py:312), not all five level types. For LINKEDIN that selects
        country/kpi/detailed_kpi and leaves the halo columns on the default answer-format branch.
        """

        return list(self.level_type.get("general_tagging", [])) + list(
            self.level_type.get("general_level", [])
        )


def _client_code() -> str:
    # One deployment serves one client, read from the environment rather than per-request. The
    # ASK_GENOME_ prefix matters: the enterprise LLM gateway already owns bare CLIENT_CODE for
    # something unrelated (see local.py's module docstring).
    return os.environ["ASK_GENOME_CLIENT_CODE"]


def _model_group_id() -> int:
    return int(os.environ["ASK_GENOME_MODEL_GROUP_ID"])


@lru_cache(maxsize=1)
def load_ner_data_config() -> NerDataConfig:
    return NerDataConfig.from_local(_client_code(), _model_group_id())


def load_intent_catalog() -> tuple[str, ...]:
    """The fixed set of coarse intents this client's data supports."""

    return tuple(load_ner_data_config().metric_info.keys())


@lru_cache(maxsize=1)
def load_feasibility_config() -> FeasibilityConfig:
    return FeasibilityConfig.from_local(_client_code(), _model_group_id())


@lru_cache(maxsize=1)
def load_spacy_config() -> SpacyConfig:
    return SpacyConfig.from_local(_client_code(), _model_group_id())


@lru_cache(maxsize=1)
def load_ner_data() -> NerData:
    return NerData.from_local(_client_code(), _model_group_id())


@dataclass(frozen=True)
class NarrowingConfig:
    """The config the data-dependent narrowing steps read, with a hash of exactly those values.

    A stored table recipe replays against this. The hash is over content rather than file times
    so an unchanged config that was merely rewritten still replays, and so a change that reaches
    any of these values invalidates recipes however it arrived.

    `client_code` is here because reconciliation branches on it. It stays out of the fingerprint:
    a recipe already records its client and refuses to replay for another.
    """

    client_code: str
    core_filters: dict[str, Any]
    level_type: dict[str, list[str]]
    sub_cols: tuple[str, ...]
    metric_info: dict[str, Any]
    # Replay picks between these from the recipe's own `data`, so neither is stored per recipe.
    bi_levels: tuple[str, ...]
    br_levels: tuple[str, ...]
    fingerprint: str

    def levels_for(self, source: str) -> list[str]:
        return list(self.bi_levels if source == "bi" else self.br_levels)


@lru_cache(maxsize=1)
def load_narrowing_config() -> NarrowingConfig:
    ner_data_config = load_ner_data_config()
    level_type = load_ner_data().level_type
    core_filters = load_spacy_config().core_filters
    sub_cols = tuple(ner_data_config.data_levels.get("spacyOverrideColumns", []))
    bi_levels = tuple(ner_data_config.data_levels.get("biLevels", []))
    br_levels = tuple(ner_data_config.data_levels.get("brLevels", []))
    # metric_info and the level lists are hashed too: both change which rows survive.
    canonical = json.dumps(
        {
            "core_filters": core_filters,
            "level_type": level_type,
            "sub_cols": sub_cols,
            "metric_info": ner_data_config.metric_info,
            "bi_levels": bi_levels,
            "br_levels": br_levels,
        },
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return NarrowingConfig(
        client_code=_client_code(),
        core_filters=core_filters,
        level_type=level_type,
        sub_cols=sub_cols,
        metric_info=ner_data_config.metric_info,
        bi_levels=bi_levels,
        br_levels=br_levels,
        fingerprint=hashlib.sha256(canonical.encode()).hexdigest()[:16],
    )


@lru_cache(maxsize=1)
def load_time_indicators() -> dict[str, Any]:
    """The postprocess indicators, normalised against the period types this client's data has.

    Cached because the normalisation reads both frames, and the answer is the same for every
    subquery in the process.
    """

    from ask_genome_agent.nodes.question_understanding.support.time_filter import (
        validated_indicators,
    )

    ner_data = load_ner_data()
    indicators = load_ner_data_config().ner_postprocess_indicators or {}
    return validated_indicators(indicators, (ner_data.df_bi, ner_data.df_br))
