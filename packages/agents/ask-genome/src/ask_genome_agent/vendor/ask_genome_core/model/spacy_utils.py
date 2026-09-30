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


def _expand_term(term: str, variants: dict) -> str:
    """
    expand longest matched segment in text
    """
    tokens = term.split()
    n = len(tokens)
    i = 0
    result = []

    variant_keys = [(k.split(), k) for k in variants.keys()]

    while i < n:
        best_match = None
        best_len = 0

        # Try all variant keys at position i
        for key_tokens, original_key in variant_keys:
            L = len(key_tokens)
            if i + L <= n and tokens[i:i + L] == key_tokens:
                if L > best_len:
                    best_len = L
                    best_match = original_key

        if best_match:
            # Map using original key → mapped value
            result.append(variants[best_match])
            i += best_len
        else:
            # No match → keep original token
            result.append(tokens[i])
            i += 1

    return " ".join(result)


def normalize_query(query, cleaned_variant_mapping: dict):
    nlp = spacy.load("en_core_web_sm")

    cleaned_query = clean_text(nlp, query)
    normalized_query = _expand_term(cleaned_query, cleaned_variant_mapping)
    return normalized_query


def stable_lemma(token):
    if token.pos_ in {"NOUN", "PROPN"}:
        return token.lemma_
    else:
        return token.text.lower()


def clean_text(nlp, text, lemma=True) -> str:
    """
    remove possessive in text, lemmatize tokens
    :param text:
    :return: cleaned text
    """
    doc = [token for token in nlp(text) if token.tag_ != "POS"]
    # cleaned_text = " ".join([token.lemma_ for token in doc])
    if lemma:
        cleaned_text = " ".join([stable_lemma(token) for token in doc if not token.is_punct])
    else:
        cleaned_text = " ".join([token.text for token in doc if not token.is_punct])
    return cleaned_text


def _clean_variant_mapping(variants_df):
    """
    clean term and variants in variants df to lemmatized terms
    :param nlp:
    :param variants_df:
    :return:
    """
    nlp = spacy.load("en_core_web_sm")
    res = variants_df.copy()
    cleaned_terms = {}
    for term in variants_df["term"].dropna().unique():
        cleaned_terms[term] = clean_text(nlp, term)

    cleaned_variants = {}
    for term in variants_df["variants"].dropna().unique():
        cleaned_variants[term] = clean_text(nlp, term)
    res["cleaned_term"] = res["term"].map(cleaned_terms)
    res["cleaned_variants"] = res["variants"].map(cleaned_variants)
    variant_mapping = res.set_index("cleaned_variants")["cleaned_term"].to_dict()
    return variant_mapping


# spacy construction function
def _create_spacy_files(core_dimensions: pd.DataFrame,
                        bidata: pd.DataFrame,
                        brdata: pd.DataFrame,
                        variant_mapping: pd.DataFrame = pd.DataFrame(),
                        level_type: dict = None,
                        level_group: dict = None,
                        include_diag=False,
                        metric_configs: pd.DataFrame = pd.DataFrame()) -> tuple[dict, list]:
    """
    Process core dimensions, filter terms, and generate Spacy JSON files.

    :param core_dimensions: DataFrame containing core dimension terms.
    :param bidata: DataFrame
    :param brdata: DataFrame
    """
    nlp = spacy.load("en_core_web_sm")

    if not level_type:
        level_type = dict()
    if variant_mapping.empty:
        variant_mapping = pd.DataFrame(columns=["term", "variants"])

    # clean terms and variants
    cleaned_variant_mapping = _clean_variant_mapping(variants_df=variant_mapping)

    level_cols = level_type.get('general_level', [])
    tagging_cols = level_type.get('general_tagging', [])
    halo_cols = level_type.get("halo_level", []) + level_type.get("halo_tagging", [])
    diag_cols = []

    # Constants
    FILTER_COLUMNS = ["business_driver", "business_driver_detail", "activity_group", "measure_group", "measure",
                      "marketing_channel", "media_channel",
                      "custom_aggregated"]
    INDICATOR_COLUMNS = ["composite", "halo_to_ignore", "AKA"]
    GENERAL_EXCLUDE_LIST = {
        'ahw media', 'masterbrand', 'optic white tp', 'ahw',
        'optic white tp comprehensive', 'optic white tp ush',
        'masterbrand ush', 'total tp', 'masterbrand comprehensive',
        'total tp ush', 'total tp comprehensive',  'equity', 'equity ush', 'equity comprehensive',
        'pillars', 'other', 'promotions', 'system', 'innovation',
        'trends', 'all other', 'overall', 'other media',
    }
    EXACT_MATCH_COLS = ['business_driver', 'business_driver_detail',
                        'activity_group', 'measure_group', 'measure']
    if include_diag:
        if "split_type" in bidata.columns or "split_type" in brdata.columns:
            EXACT_MATCH_COLS.append("split_type")
            diag_cols.append("split_type")
        if "split" in bidata.columns or "split" in brdata.columns:
            EXACT_MATCH_COLS.append("split")
            diag_cols.append("split")
    # BR_EXACT_MATCH_COLS = {'business_driver', 'business_driver_detail', 'activity_group', 'measure_group', 'measure'}
    SPECIAL_CHARACTERS = {'/', '$'}
    CUSTOM_COLUMNS = level_cols + tagging_cols + diag_cols + halo_cols

    if len(core_dimensions) == 0:
        core_dimensions = pd.DataFrame(columns=FILTER_COLUMNS + INDICATOR_COLUMNS + CUSTOM_COLUMNS
                                               + ['media tactics term', 'keyword'])

    # Lowercase all text columns
    core_dimensions = core_dimensions.map(lambda x: x.lower() if isinstance(x, str) else x)

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
    core_dimension_filters = _generate_filters(core_dimensions, bidata, brdata,
                                               FILTER_COLUMNS, INDICATOR_COLUMNS, CUSTOM_COLUMNS)

    # Add product exclusions
    if 'product' in bidata.columns:
        GENERAL_EXCLUDE_LIST.update(bidata['product'].dropna().unique())
    if 'brand' in bidata.columns:
        GENERAL_EXCLUDE_LIST.update(bidata['brand'].dropna().unique())

    # Add exact matches
    core_dimension_filters = _add_exact_matches(core_dimension_filters, bidata, brdata,
                                                EXACT_MATCH_COLS, CUSTOM_COLUMNS, GENERAL_EXCLUDE_LIST,
                                                SPECIAL_CHARACTERS, ignore_dict,
                                                tagging_columns=tagging_cols+halo_cols,
                                                variant_mapping=cleaned_variant_mapping,
                                                level_group=level_group, level_type=level_type)
    # core_dimension_filters = _add_exact_matches(core_dimension_filters, brdata, BR_EXACT_MATCH_COLS,
    #                                             GENERAL_EXCLUDE_LIST, SPECIAL_CHARACTERS, ignore_dict,
    #                                             source='br')

    # Add metric names, aliases and metric groups
    if not metric_configs.empty:
        metric_filters, metric_group_filters = _generate_metric_filters(metric_configs, bidata, brdata)
        for metric_lookup in (metric_filters, metric_group_filters):
            for term, filters in metric_lookup.items():
                core_dimension_filters.setdefault(term, []).extend(filters)

    # Apply lemmatization and generate new terms
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()
    # core_dimension_filters = _expand_keywords(core_dimension_filters, lemmatizer, stemmer,
    #                                           composite_indicator, CUSTOM_COLUMNS)

    # normalize terms with acronyms and synonyms
    core_dimension_filters = _expand_variants(nlp,
                                              core_dimension_filters=core_dimension_filters,
                                              variant_mapping=cleaned_variant_mapping)

    # Remove invalid filters
    core_dimension_filters = _clean_filters(core_dimension_filters=core_dimension_filters)

    # Flag filters that only match diagnostic rows
    core_dimension_filters = _flag_diag_only_filters(core_dimension_filters=core_dimension_filters,
                                                     bidata=bidata, brdata=brdata)

    # Generate and save Spacy JSON files
    term_list = _generate_spacy_patterns(core_dimensions=core_dimensions,
                                                core_dimension_filters=core_dimension_filters,
                                                bidata=bidata, brdata=brdata,
                                                composite_indicator=composite_indicator,
                                                level_cols=level_cols, tagging_cols=tagging_cols,
                                                diag_cols=diag_cols, halo_cols=halo_cols, nlp=nlp)
    return core_dimension_filters, term_list


def _flag_diag_only_filters(core_dimension_filters: dict,
                            bidata: pd.DataFrame,
                            brdata: pd.DataFrame) -> dict:
    """Stamp diag_only=True on filters whose conjunction only matches diagnostic rows.

    True means the filter matches no row with custom_aggregated != 'diagnostic' in any frame it
    applies to. A filter that is diagnostic-only in one frame but not the other stays False, so
    the flag never routes a filter away from rows that can answer it. The key is only written
    when True; absent means False, which is the right default for artifacts built before this.
    """
    meta_keys = {"AKA", "org_term", "source", "halo_to_ignore", "detail", "diag_only"}
    normal_frames = []
    for df in (bidata, brdata):
        if 'custom_aggregated' in df.columns:
            normal = df[df['custom_aggregated'] != 'diagnostic']
            # a frame without diagnostic rows cannot make any filter diagnostic-only
            if len(normal) < len(df):
                normal_frames.append(normal)
    if not normal_frames:
        return core_dimension_filters

    for filters in core_dimension_filters.values():
        for f in filters:
            cols = [k for k in f if k not in meta_keys]
            applicable = False
            diag_only = True
            for normal in normal_frames:
                valid_cols = [c for c in cols if c in normal.columns]
                if not valid_cols:
                    continue
                applicable = True
                mask = pd.concat([normal[c].isin(f[c]) for c in valid_cols], axis=1).all(axis=1)
                if mask.any():
                    diag_only = False
                    break
            if applicable and diag_only:
                f['diag_only'] = True
    return core_dimension_filters


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
                       exclude_list, special_characters, ignore_dict,
                       tagging_columns, variant_mapping,
                       level_group, level_type):
    """Add exact matches to filters."""
    custom_dimension_values = set()
    tagging_exclude_set = set(bidata[tagging_columns].stack().unique()) | set(brdata[tagging_columns].stack().unique())
    tagging_exclude_set |= set([normalize_query(x, variant_mapping) for x in tagging_exclude_set])
    bd_exclude_set = set(bidata["business_driver"].dropna().unique()) | set(brdata["business_driver"].dropna().unique())
    predecessor_map = _find_predecessor(level_type, level_group)

    for col in custom_dimensions:
        if col in bidata.columns:
            custom_dimension_values.update(set(bidata[col].dropna().unique()))
        if col in brdata.columns:
            custom_dimension_values.update(set(brdata[col].dropna().unique()))
    custom_dimension_values -= {'overall', 'other'}

    for col in exact_match_cols + custom_dimensions:
        print(col)
        pred_col = predecessor_map.get(col, [])
        # ignore_set = (exclude_list | set(core_dimension_filters.keys()) | set(ignore_dict.get(col, []))) - custom_dimension_values
        ignore_set = set()
        bi_valid_em = set()
        br_valid_em = set()
        if col in bidata.columns and col in ["retailer_focused_marketing"]:
            ignore_set = exclude_list | custom_dimension_values
            bi_valid_em = set(bidata[col].dropna().unique()) - ignore_set
        elif col in bidata.columns:
            ignore_set = exclude_list - custom_dimension_values
            bi_valid_em = set(bidata[col].dropna().unique()) - ignore_set
        if col in brdata.columns:
            br_valid_em = set(brdata[col].dropna().unique()) - ignore_set
        candidates = bi_valid_em.union(br_valid_em)
        # ignore same tagging values in ag/mg/m
        if col in ["activity_group", "measure_group", "measure"]:
            # remove values exact as tagging
            candidates -= tagging_exclude_set
            # remove values that contain tagging
            candidates = {v for v in candidates if not any(ex in v.split() for ex in tagging_exclude_set)}
        # ignore same bd values in all exact match columns
        if col != "business_driver":
            candidates -= bd_exclude_set
        for v in candidates:
            # if not any(sc in v for sc in special_characters) and not any(
            #         col in f for f in core_dimension_filters.get(v, [])):
            if not any(sc in v for sc in special_characters):
                if v in ignore_dict.get(col, []):
                    detail = "ignore when cust_agg"
                else:
                    detail = ""
                if v in bi_valid_em.intersection(br_valid_em):
                    source = 'bibr'
                elif v in bi_valid_em:
                    source = 'bi'
                elif v in br_valid_em:
                    source = 'br'
                else:
                    source = 'na'
                # assume maximum one predecessor level
                if pred_col and pred_col[0] in bidata.columns:
                    pred_v = (
                            set(bidata.loc[bidata[col] == v, pred_col[0]].dropna().unique()) |
                            set(brdata.loc[brdata[col] == v, pred_col[0]].dropna().unique())
                    )
                    print(col, v, pred_col, pred_v)
                    for pv in pred_v:
                        core_dimension_filters.setdefault(v, []).append({pred_col[0]: [pv], col: [v],
                                                                         'source': source, 'detail': detail})
                else:
                    core_dimension_filters.setdefault(v, []).append({col: [v],
                                                                     'source': source, 'detail': detail})

                if col == "business_driver":
                    core_dimension_filters.setdefault("marketing", []).append({col: ["marketing", "media", "non-media", "promotions"],
                                                                               'source': "bibr",
                                                                               'detail': ""})
                    core_dimension_filters.setdefault("base", []).append({col: ["base", "other"],
                                                                          'source': "br",
                                                                          'detail': ""})
                    core_dimension_filters.setdefault("media", []).append({col: ["media"],
                                                                          'source': "bibr",
                                                                          'detail': ""})
                    core_dimension_filters.setdefault("non-media", []).append({col: ["non-media"],
                                                                               'source': "bibr",
                                                                               'detail': ""})
    return core_dimension_filters


def _generate_metric_filters(metric_configs: pd.DataFrame,
                             bidata: pd.DataFrame,
                             brdata: pd.DataFrame) -> tuple[dict, dict]:
    """Create filters for metric names, aliases and metric groups."""
    NAME_COLUMNS = ['kpiName', 'alias', 'MetricName']
    GROUP_COLUMNS = ['metricGroup', 'code']
    bi_metrics = set(bidata['metric'].dropna().unique())
    br_metrics = set(brdata['metric'].dropna().unique())

    configs = metric_configs.copy()
    for col in NAME_COLUMNS + GROUP_COLUMNS:
        configs[col] = configs[col].fillna('').astype(str).str.strip().str.lower()
    configs = configs[configs['kpiName'].isin(bi_metrics | br_metrics)]

    def build_filter(kpis, scope):
        metric_filter = {'metric': sorted(set(kpis)),
                         'source': _get_filter_source(sorted(k for k in kpis if k in bi_metrics),
                                                      sorted(k for k in kpis if k in br_metrics))}
        if scope == 'group':
            metric_filter['metric_scope'] = scope
        return metric_filter

    group_kpis = {}
    for col in GROUP_COLUMNS:
        for group, kpis in configs[configs[col] != ''].groupby(col)['kpiName']:
            group_kpis.setdefault(group, set()).update(kpis)
    metric_group_filters = {group: [build_filter(kpis, 'group')] for group, kpis in group_kpis.items()}

    metric_filters = {}
    for _, row in configs.iterrows():
        for col in NAME_COLUMNS:
            if row[col]:
                metric_filters[row[col]] = [build_filter([row['kpiName']], 'metric')]

    # a group resolving to the same single metric as a specific name of the same text adds nothing
    for term, filters in metric_filters.items():
        group_filters = metric_group_filters.get(term, [])
        if group_filters and group_filters[0]['metric'] == filters[0]['metric']:
            metric_group_filters.pop(term)

    return metric_filters, metric_group_filters


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


def _generate_spacy_patterns(core_dimensions, core_dimension_filters, bidata, brdata,
                             composite_indicator, level_cols, tagging_cols, diag_cols, halo_cols, nlp):
    """Generate Spacy patterns and save them as JSON."""
    terms = set(core_dimension_filters.keys()) | set(core_dimensions['media tactics term'].dropna())
    # all_terms = set()
    # for term in terms:
    #     lemmatized = _lemmatize_words(term, lemmatizer)
    #     # stemmed = _stem_words(term, stemmer)
    #     all_terms = all_terms | {term, lemmatized}
    all_terms = sorted(terms, key=lambda x: len(x.split()), reverse=True)

    dict_list = []

    def get_label(term):
        filters = core_dimension_filters.get(term, [{}])
        covered_cols = {x for f in filters for x in f.keys() if x in bidata.columns}
        metric_scopes = {f.get('metric_scope', 'metric') for f in filters if 'metric' in f}
        if core_dimension_filters.get(term) and metric_scopes == {'group'}:
            return "METRIC_GROUP"
        if core_dimension_filters.get(term) and 'metric' in covered_cols:
            return "METRIC"
        if composite_indicator.get(term, '0') == '1':
            return "CORE_DIMENSION_COMPOSITE"
        if core_dimension_filters.get(term) and set(level_cols).intersection(covered_cols):
            return "CUSTOM_LEVEL"
        if core_dimension_filters.get(term) and set(tagging_cols).intersection(covered_cols):
            return "CUSTOM_TAGGING"
        if core_dimension_filters.get(term) and set(diag_cols).intersection(covered_cols):
            return "DIAG_TAGGING"
        if core_dimension_filters.get(term) and set(halo_cols).intersection(covered_cols):
            return "HALO_TAGGING"
        if core_dimension_filters.get(term) and "business_driver" in covered_cols and len(covered_cols) == 1:
            return "BUSINESS_DRIVER"
        return "CORE_DIMENSION"

    for i, term in enumerate(all_terms):
        # tokens = nltk.word_tokenize(term)
        # tokens = [token.lower() for token in tokens if token.strip()]
        pat_clean = []
        pat_punct = []

        label = get_label(term)
        doc = [token for token in nlp(term) if token.tag_ != "POS"]
        pat_clean = [{"LOWER": token.text.lower()} for token in doc]
        if pat_clean:
            dict_list.append({"label": label, "pattern": pat_clean})

        if any(not t.is_alpha for t in doc):
            pat_punct = []
            for token in doc:
                if token.is_alpha:
                    pat_punct.append({"LOWER": token.text.lower()})
                else:
                    pat_punct.append({"ORTH": token.text})
            if pat_punct:
                dict_list.append({"label": label, "pattern": pat_punct})

    return dict_list


def merge_around_punctuation(nlp, text) -> str:
    doc = [token for token in nlp(text) if token.tag_ != "POS"]
    tokens = []
    i = 0
    while i < len(doc):
        token = doc[i]
        if token.is_punct and i > 0 and i < len(doc) - 1:
            # Merge previous and next token
            merged = doc[i - 1].text + doc[i + 1].text
            tokens.pop()
            tokens.append(merged)
            i += 2
        elif not token.is_punct:
            tokens.append(token.text)
            i += 1
        else:
            i += 1
    return " ".join(tokens)


def _expand_variants(nlp, core_dimension_filters, variant_mapping):
    # ordered_variants = sorted(variant_mapping.keys(),
    #                           key=lambda x: len(x.split()),
    #                           reverse=True) # sort descending
    result_filters = dict()
    for term, filters in core_dimension_filters.items():
        for f in filters:
            f.update({"org_term": f.get("org_term", term)})

        cleaned_term = clean_text(nlp, text=term) # lemmatized token
        res = _expand_term(cleaned_term, variant_mapping)
        cleaned_res = res
        # cleaned_res = clean_text(nlp, text=res)

        merged_term = merge_around_punctuation(nlp, text=term)
        merged_res = clean_text(nlp, text=merged_term)
        res = _expand_term(merged_res, variant_mapping)
        merged_res = res

        variant_filters = result_filters.get(cleaned_res, []) + result_filters.get(merged_res, [])
        variant_filters.extend(filters)
        result_filters[cleaned_res] = variant_filters
        result_filters[merged_res] = variant_filters
        result_filters[term] = variant_filters

    return result_filters


def _find_predecessor(level_type, level_group):
    # Reverse map: column -> group name
    col_to_group = {}
    for group, cols in level_group.items():
        for col in cols:
            col_to_group[col] = group

    # For each column in level_type, find same-group predecessors within the same level_type list
    result = {}

    for level_key, columns in level_type.items():
        for i, col in enumerate(columns):
            col_group = col_to_group.get(col)
            predecessors = []
            if col_group:
                for prev_col in columns[:i]:  # only ranked before
                    if col_to_group.get(prev_col) == col_group:
                        predecessors.append(prev_col)
            if predecessors:
                result[col] = predecessors
    return result


def _lemmatize_words(phrase: str, lemmatizer) -> str:
    """Lemmatize words in a phrase."""
    words = nltk.word_tokenize(phrase)
    return ' '.join(lemmatizer.lemmatize(word) for word in words)


def _stem_words(phrase: str, stemmer) -> str:
    """Stem words in a phrase."""
    words = nltk.word_tokenize(phrase)
    new_phrase = [stemmer.stem(word) for word in words]

    return ' '.join(new_phrase)