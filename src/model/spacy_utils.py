import os
import re
import json
import logging
import numpy as np
import pandas as pd
import spacy
from spacy.lang.en import English
import nltk
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from spacy.tests.lang.tr.test_tokenizer import GENERAL_TESTS

from src.data.local import save_spacy_jsonl, save_spacy_json

nltk.download('punkt')
nltk.download('wordnet')

logger = logging.getLogger(__name__)


def _get_filter_source(bi_list: list, br_list: list) -> str:
    if bi_list and br_list:
        source = 'bibr'
    elif bi_list:
        source = 'bi'
    elif br_list:
        source = 'br'
    else:
        source = 'na'
    return source


def _get_core_dimension_filter(text: str):
    if 'not in' in text:
        negate = True
    else:
        negate = False
    cleaned_text = re.sub(r"^=|^(in \()|^(in\()|^(not in)", "", text).strip()
    cleaned_text = re.sub(", ", ",", cleaned_text)
    cleaned_list = [x.strip() for x in cleaned_text.split(",")]
    return cleaned_list, negate


def _get_core_dimension_filter_by_keyword(text: str, df1: pd.DataFrame, df2: pd.DataFrame, cols=None) -> list:
    if cols is None:
        cols = ['activity_group', 'measure_group', 'measure']
    if text == 'not applicable':
        return []
    keywords = [x.strip() for x in text.split(',')]
    filters = []
    for col in cols:
        bi_valid_list = [x for x in df1[col].dropna().unique() if any([kw for kw in keywords if kw in x.split(' ')])]
        br_valid_list = [x for x in df2[col].dropna().unique() if any([kw for kw in keywords if kw in x.split(' ')])]
        source = _get_filter_source(bi_valid_list, br_valid_list)
        valid_list = sorted(set(bi_valid_list).union(set(br_valid_list)))
        if valid_list:
            filters.append({col: valid_list, 'source': source})
    return filters


# spacy construction function
def _create_spacy_files(core_dimensions: pd.DataFrame,
                        bidata: pd.DataFrame,
                        brdata: pd.DataFrame,
                        level_type: dict = None,
                        include_metric=False) -> tuple[dict, list]:
    """
    Process core dimensions, filter terms, and generate Spacy JSON files.

    :param core_dimensions: DataFrame containing core dimension terms.
    :param bidata: DataFrame
    :param brdata: DataFrame
    """

    # Constants
    FILTER_COLUMNS = ["business_driver", "business_driver_detail", "activity_group", "measure_group", "measure",
                      "custom_aggregated"]
    INDICATOR_COLUMNS = ["composite", "halo_to_ignore", "AKA"]
    GENERAL_EXCLUDE_LIST = {
        'ahw media', 'masterbrand', 'optic white tp', 'ahw',
        'optic white tp comprehensive', 'optic white tp ush',
        'masterbrand ush', 'total tp', 'masterbrand comprehensive',
        'total tp ush', 'total tp comprehensive',  'equity', 'equity ush', 'equity comprehensive',
        'pillars', 'media', 'other', 'promotions', 'system', 'innovation', 'marketing',
        'trends', 'all other', 'overall', 'other media', 'base',
    }
    EXACT_MATCH_COLS = ['business_driver', 'business_driver_detail', 'activity_group', 'measure_group', 'measure']
    # BR_EXACT_MATCH_COLS = {'business_driver', 'business_driver_detail', 'activity_group', 'measure_group', 'measure'}
    SPECIAL_CHARACTERS = {'/', '$'}

    if not level_type:
        level_type = dict()
    level_cols = level_type.get('general_level', [])
    tagging_cols = level_type.get('general_tagging', [])
    CUSTOM_COLUMNS = level_cols + tagging_cols

    if len(core_dimensions) == 0:
        core_dimensions = pd.DataFrame(columns=FILTER_COLUMNS + INDICATOR_COLUMNS + CUSTOM_COLUMNS
                                               + ['media tactics term', 'keyword'])

    # Lowercase all text columns
    core_dimensions = core_dimensions.applymap(lambda x: x.lower() if isinstance(x, str) else x)

    # Drop rows where all filter columns and 'keyword' are NaN
    # core_dimensions.dropna(subset=FILTER_COLUMNS + ['keyword'], how='all', inplace=True)

    # Fill missing filter columns with 'not applicable'
    core_dimensions = core_dimensions.reindex(columns=core_dimensions.columns.union(FILTER_COLUMNS + CUSTOM_COLUMNS), fill_value=np.nan)
    core_dimensions[FILTER_COLUMNS + CUSTOM_COLUMNS + ['keyword']] = core_dimensions[FILTER_COLUMNS + CUSTOM_COLUMNS + ['keyword']].fillna(
        'not applicable')

    # Forward fill 'media tactics term'
    # core_dimensions['media tactics term'] = core_dimensions['media tactics term'].fillna(method='ffill')
    ignore_terms = core_dimensions[core_dimensions['media tactics term'].isna()]
    ignore_dict = {k: [val for val in v if not pd.isna(val)]
                   for k, v in ignore_terms.to_dict('list').items()
                   if any(not pd.isna(val) for val in v)
                   }
    core_dimensions = core_dimensions[~core_dimensions['media tactics term'].isna()]

    # Ensure composite and halo_to_ignore exist and are in string format
    for col in INDICATOR_COLUMNS:
        if col not in core_dimensions.columns:
            core_dimensions[col] = '0'
    core_dimensions[INDICATOR_COLUMNS] = core_dimensions[INDICATOR_COLUMNS].fillna(0).astype(int).astype(str)

    # Composite indicator dictionary
    composite_indicator = core_dimensions.set_index('media tactics term')['composite'].to_dict()

    # Filtering logic
    core_dimension_filters = _generate_filters(core_dimensions, bidata, brdata, FILTER_COLUMNS, INDICATOR_COLUMNS, CUSTOM_COLUMNS)

    # Add product exclusions
    if 'product' in bidata.columns:
        GENERAL_EXCLUDE_LIST.update(bidata['product'].dropna().unique())
    if 'brand' in bidata.columns:
        GENERAL_EXCLUDE_LIST.update(bidata['brand'].dropna().unique())

    # Add exact matches
    core_dimension_filters = _add_exact_matches(core_dimension_filters, bidata, brdata,
                                                EXACT_MATCH_COLS, CUSTOM_COLUMNS, GENERAL_EXCLUDE_LIST,
                                                SPECIAL_CHARACTERS, ignore_dict)
    # core_dimension_filters = _add_exact_matches(core_dimension_filters, brdata, BR_EXACT_MATCH_COLS,
    #                                             GENERAL_EXCLUDE_LIST, SPECIAL_CHARACTERS, ignore_dict,
    #                                             source='br')

    # Apply lemmatization and generate new terms
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()
    core_dimension_filters = _expand_keywords(core_dimension_filters, lemmatizer, stemmer,
                                              composite_indicator, CUSTOM_COLUMNS)

    # Remove invalid filters
    core_dimension_filters = _clean_filters(core_dimension_filters)

    # Generate and save Spacy JSON files
    term_list = _generate_spacy_patterns(core_dimensions, core_dimension_filters,
                                         bidata, brdata, composite_indicator,
                                         lemmatizer, stemmer, include_metric,
                                         level_cols, tagging_cols)

    # save files
    # save_spacy_json(core_dimension_filters)
    # save_spacy_jsonl(term_list)
    return core_dimension_filters, term_list


def _generate_filters(core_dimensions: pd.DataFrame,
                      bidata: pd.DataFrame,
                      brdata: pd.DataFrame,
                      filter_dimensions: list,
                      indicator_dimensions: list,
                      custom_dimensions: list) -> dict:
    """Create filters for core dimensions."""
    core_dimension_filters = {}
    for _, row in core_dimensions.iterrows():
        term = row['media tactics term']
        constructed_filters = core_dimension_filters.get(term, [])
        current_filter = {}

        for filter_dim in filter_dimensions + indicator_dimensions + custom_dimensions:
            text = row[filter_dim]
            if filter_dim not in indicator_dimensions and text != 'not applicable':
                cleaned_list, negate = _get_core_dimension_filter(text)
                bi_available_values = set(bidata[filter_dim].dropna().unique())
                br_available_values = set(brdata[filter_dim].dropna().unique())

                if negate:
                    bi_cleaned_list = sorted([x for x in bi_available_values if x not in cleaned_list])
                    br_cleaned_list = sorted([x for x in br_available_values if x not in cleaned_list])
                else:
                    bi_cleaned_list = sorted([x for x in cleaned_list if x in bi_available_values])
                    br_cleaned_list = sorted([x for x in cleaned_list if x in br_available_values])

                source = _get_filter_source(bi_cleaned_list, br_cleaned_list)
                current_filter['source'] = source

                if bi_cleaned_list or br_cleaned_list:
                    current_filter[filter_dim] = sorted(set(bi_cleaned_list) | set(br_cleaned_list))
                else:
                    # NOTFOUND.append({'term': term, 'level': filter_dim, 'text': text})
                    # print(f'"{term}" at "{filter_dim}" has "{text}" not found in data, source {source}')
                    continue
            elif filter_dim in indicator_dimensions and current_filter:
                current_filter[filter_dim] = text

        # Keyword filter
        if 'keyword' in row:
            keyword_filter = _get_core_dimension_filter_by_keyword(row['keyword'], bidata, brdata)
            constructed_filters.extend(keyword_filter)

        if current_filter:
            constructed_filters.append(current_filter)

        if constructed_filters:
            core_dimension_filters[term] = constructed_filters

    return core_dimension_filters


def _add_exact_matches(core_dimension_filters, bidata, brdata,
                       exact_match_cols, custom_dimensions,
                       exclude_list, special_characters, ignore_dict):
    """Add exact matches to filters."""
    custom_dimension_values = set()
    for col in custom_dimensions:
        custom_dimension_values.update(set(bidata[col].dropna().unique()))
        custom_dimension_values.update(set(brdata[col].dropna().unique()))
    custom_dimension_values -= {'overall'}

    for col in exact_match_cols + custom_dimensions:
        ignore_set = (exclude_list | set(core_dimension_filters.keys()) | set(ignore_dict.get(col, []))) - custom_dimension_values
        bi_valid_em = set(bidata[col].dropna().unique()) - ignore_set
        br_valid_em = set(brdata[col].dropna().unique()) - ignore_set
        for v in bi_valid_em.union(br_valid_em):
            if not any(sc in v for sc in special_characters) and not any(
                    col in f for f in core_dimension_filters.get(v, [])):
                if v in bi_valid_em.intersection(br_valid_em):
                    source = 'bibr'
                elif v in bi_valid_em:
                    source = 'bi'
                elif v in br_valid_em:
                    source = 'br'
                else:
                    source = 'na'
                core_dimension_filters.setdefault(v, []).append({col: [v], 'source': source})
    return core_dimension_filters


def _clean_filters(core_dimension_filters):
    """Remove invalid filters from the dictionary."""
    cleaned_filters = {}

    for keyword, filters in core_dimension_filters.items():
        valid_filters = [f for f in filters if not (len(f) == 1 and 'halo_to_ignore' in f)]
        if valid_filters:
            cleaned_filters[keyword] = list(valid_filters)

    return cleaned_filters


def _expand_keywords(core_dimension_filters, lemmatizer, stemmer, composite_indicator, custom_dimensions):
    """Expand keywords with lemmatization and stemming."""
    new_filters = {}
    for keyword, filters in core_dimension_filters.items():
        all_keys = {x for f in filters for x in f.keys()}
        lemmatized = _lemmatize_words(keyword, lemmatizer)
        stemmed = _stem_words(keyword, stemmer)
        keyword_set = {keyword, lemmatized}

        # if set(custom_dimensions).intersection(all_keys):
        #     keyword_set = {keyword, stemmed, lemmatized}

        raw_tokens = re.split(r'(\W)', keyword)
        tokens = [t for t in raw_tokens if t and not t.isspace()]
        has_punct = any(re.match(r'\W', t) for t in tokens)
        if has_punct:
            words_only_term = ' '.join([t.lower() for t in tokens if re.match(r'\w+', t)])

            lemmatized = _lemmatize_words(words_only_term, lemmatizer)
            stemmed = _stem_words(words_only_term, stemmer)
            keyword_set.update({lemmatized})

        for k in keyword_set:
            k_filter = core_dimension_filters.get(k, [])
            filters.extend(k_filter)
        unique_list = list({json.dumps(d, sort_keys=True) for d in filters})
        unique_list = [json.loads(d) for d in unique_list]
        for d in unique_list:
            d.update({'org_term': keyword})

        for k in keyword_set:
            new_filters[k] = unique_list
            composite_indicator[k] = composite_indicator.get(keyword, '0')

    return new_filters


def _generate_spacy_patterns(core_dimensions, core_dimension_filters, bidata, brdata, composite_indicator,
                             lemmatizer, stemmer, include_metric, level_cols, tagging_cols):
    """Generate Spacy patterns and save them as JSON."""
    terms = set(core_dimension_filters.keys()) | set(core_dimensions['media tactics term'].dropna())
    all_terms = set()
    for term in terms:
        lemmatized = _lemmatize_words(term, lemmatizer)
        # stemmed = _stem_words(term, stemmer)
        all_terms = all_terms | {term, lemmatized}
    all_terms = sorted(all_terms, key=lambda x: len(x.split()), reverse=True)

    dict_list = []
    for term in all_terms:
        # tokens = nltk.word_tokenize(term)
        # tokens = [token.lower() for token in tokens if token.strip()]
        covered_cols = {x for f in core_dimension_filters.get(term, [{}]) for x in f.keys()}
        if core_dimension_filters.get(term) and 'metric' in covered_cols:
            label = "METRIC"
        elif composite_indicator.get(term, '0') == '1':
            label = "CORE_DIMENSION_COMPOSITE"
        elif core_dimension_filters.get(term) and set(level_cols).intersection(covered_cols):
            label = "CUSTOM_LEVEL"
        elif core_dimension_filters.get(term) and set(tagging_cols).intersection(covered_cols):
            label = "CUSTOM_TAGGING"
        else:
            label = "CORE_DIMENSION"

        raw_tokens = re.split(r'(\W)', term)
        tokens = [t for t in raw_tokens if t and not t.isspace()]
        has_punct = any(re.match(r'\W', t) for t in tokens)
        words_only = [t.lower() for t in tokens if re.match(r'\w+', t)]
        pattern_without_punct = [{"LOWER": word} for word in words_only]

        dict_list.append({"label": label, "pattern": pattern_without_punct})

        pattern_with_punct = []
        if has_punct:
            for token in tokens:
                if re.match(r'\W', token):  # punctuation
                    pattern_with_punct.append({"IS_PUNCT": True, "ORTH": token})
                else:
                    pattern_with_punct.append({"LOWER": token.lower()})
            dict_list.append({"label": label, "pattern": pattern_with_punct})

    if include_metric:
        for metric in set(bidata['metric'].dropna().unique()) | set(brdata['metric'].dropna().unique()):
            tokens = metric.lower().split()
            dict_list.append({"label": "METRIC", "pattern": [{"LOWER": x} for x in tokens]})

    return dict_list


def _lemmatize_words(phrase: str, lemmatizer) -> str:
    """Lemmatize words in a phrase."""
    words = nltk.word_tokenize(phrase)
    return ' '.join(lemmatizer.lemmatize(word) for word in words)


def _stem_words(phrase: str, stemmer) -> str:
    """Stem words in a phrase."""
    words = nltk.word_tokenize(phrase)
    new_phrase = [stemmer.stem(word) for word in words]

    return ' '.join(new_phrase)