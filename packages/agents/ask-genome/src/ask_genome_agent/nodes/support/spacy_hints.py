"""Deterministic: compute spacy hints for a subquery's own text.

1. Runs a deterministic spaCy-based entity extraction pass over a subquery's self-contained text.
2. Produces entity hints used later for feasibility annotation. Not a standalone graph node --
   called directly from annotate_feasibility.py, once per analytical subquery.
3. Mirrors the source spacy_hints() flow:
    Cleans the configured term-variant mappings.
    Normalizes the query.
    Removes apostrophes.
    Runs spaCy-based entity matching.
4. Ports only the text-processing and SpacyNER functionality required by get_spacy_filter.
5. Functions used only by other Ask Genome flows, such as term replacement via
   swap_spacy_term_in_query, are intentionally excluded.
"""

from typing import Any, cast

import spacy
from nltk.stem import PorterStemmer, WordNetLemmatizer
from spacy.pipeline import EntityRuler

from ask_genome_agent.config import SpacyConfig


class _SpacyNER:
    """Ported subset of ask-genome-core's SpacyNER, only what get_spacy_filter exercises."""

    def __init__(self) -> None:
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()

    def lemmatize_words(self, phrase: str, nlp: Any) -> str:
        return " ".join(token.lemma_ for token in nlp(phrase))

    def stem_words(self, phrase: str, nlp: Any) -> str:
        words = [token.text for token in nlp.tokenizer(phrase)]
        return " ".join(self.stemmer.stem(word) for word in words)

    def find_entities_and_index_from_doc(
        self, query: str, doc: Any
    ) -> dict[tuple[int, int], dict[str, Any]]:
        entities: dict[tuple[int, int], dict[str, Any]] = {}
        for ent in doc.ents:
            key = (ent.start, ent.end - 1)
            entities[key] = {"text": ent.text, "label": ent.label_}
        return entities

    def find_longest_segments(
        self, positions: dict[tuple[int, int], dict[str, Any]]
    ) -> list[tuple[int, int]]:
        valid_ents = {v["text"] for v in positions.values()}
        sorted_positions = sorted(positions, key=lambda x: (x[0], x[1]))

        longest_segments: set[tuple[int, int]] = set()
        current_start, current_end = sorted_positions[0]
        for start, end in sorted_positions[1:]:
            if start > current_end:
                longest_segments.add((current_start, current_end))
                current_start, current_end = start, end
            else:
                merged_segment = positions.get((current_start, end), {"text": ""})["text"]
                if merged_segment in valid_ents:
                    current_end = max(current_end, end)
                else:
                    longest_segments.add((current_start, current_end))
                    if start >= current_end:
                        current_start, current_end = start, end
        longest_segments.add((current_start, current_end))
        return list(longest_segments)

    def _run(self, query: str, nlp: Any) -> dict[str, Any]:
        query = query.lower()
        lem_q = self.lemmatize_words(query, nlp)
        stem_q = self.stem_words(query, nlp)

        docs = list(nlp.pipe([query, lem_q, stem_q]))

        entities = self.find_entities_and_index_from_doc(query, docs[0])
        entities.update(self.find_entities_and_index_from_doc(lem_q, docs[1]))
        entities.update(self.find_entities_and_index_from_doc(stem_q, docs[2]))

        identified_metrics: set[str] = set()
        identified_core_dim: set[str] = set()
        identified_core_dim_composite: set[str] = set()
        identified_custom_level: set[str] = set()
        identified_custom_tagging: set[str] = set()
        identified_diag_tagging: set[str] = set()
        identified_halo_tagging: set[str] = set()
        identified_bd: set[str] = set()
        if entities:
            for start, end in self.find_longest_segments(entities):
                v = entities[(start, end)]
                label, text = v["label"], v["text"]
                if label == "METRIC":
                    identified_metrics.add(text)
                elif label == "CORE_DIMENSION_COMPOSITE":
                    identified_core_dim_composite.add(text)
                elif label == "CORE_DIMENSION":
                    identified_core_dim.add(text)
                elif label == "CUSTOM_LEVEL":
                    identified_custom_level.add(text)
                elif label == "CUSTOM_TAGGING":
                    identified_custom_tagging.add(text)
                elif label == "DIAG_TAGGING":
                    identified_diag_tagging.add(text)
                elif label == "HALO_TAGGING":
                    identified_halo_tagging.add(text)
                elif label == "BUSINESS_DRIVER":
                    identified_bd.add(text)
        return {
            "metric": sorted(identified_metrics),
            "core_dimension": sorted(identified_core_dim),
            "core_dimension_composite": sorted(identified_core_dim_composite),
            "custom_level": sorted(identified_custom_level),
            "custom_tagging": sorted(identified_custom_tagging),
            "diag_tagging": sorted(identified_diag_tagging),
            "halo_tagging": sorted(identified_halo_tagging),
            "business_driver": sorted(identified_bd),
        }


def get_spacy_filter(query: str, spacy_config: SpacyConfig) -> dict[str, Any]:
    """Ported verbatim from ask-genome-core/src/model/filter_generator.py."""

    nlp = spacy.load("en_core_web_sm")
    entity_ruler = cast(
        EntityRuler,
        nlp.add_pipe("entity_ruler", name="entity_ruler", config={"overwrite_ents": True}),
    )
    entity_ruler.add_patterns(spacy_config.core_dimensions)
    return _SpacyNER()._run(query, nlp)


def stable_lemma(token: Any) -> str:
    """Ported verbatim from ask-genome-core/src/model/filter_generator.py."""

    if token.pos_ in {"NOUN", "PROPN"}:
        return str(token.lemma_)
    return str(token.text.lower())


def clean_text(nlp: Any, text: str, lemma: bool = True) -> str:
    """Ported verbatim from ask-genome-core/src/model/filter_generator.py.

    Removes possessives, lemmatizes tokens (nouns/proper nouns only -- stable_lemma).
    """

    doc = [token for token in nlp(text) if token.tag_ != "POS"]
    if lemma:
        return " ".join(stable_lemma(token) for token in doc if not token.is_punct)
    return " ".join(token.text for token in doc if not token.is_punct)


def _expand_term(term: str, variants: dict[str, str]) -> str:
    """Ported verbatim from ask-genome-core/src/model/filter_generator.py.

    Expands the longest matched segment in text against the cleaned variant mapping.
    """

    tokens = term.split()
    n = len(tokens)
    i = 0
    result: list[str] = []
    variant_keys = [(k.split(), k) for k in variants]

    while i < n:
        best_match = None
        best_len = 0
        for key_tokens, original_key in variant_keys:
            length = len(key_tokens)
            if i + length <= n and tokens[i : i + length] == key_tokens and length > best_len:
                best_len = length
                best_match = original_key
        if best_match:
            result.append(variants[best_match])
            i += best_len
        else:
            result.append(tokens[i])
            i += 1
    return " ".join(result)


def _clean_variant_mapping(variants_df: Any) -> dict[str, str]:
    """Ported verbatim from ask-genome-core/src/model/filter_generator.py.

    Cleans term and variants in the variants dataframe to lemmatized terms.
    """

    nlp = spacy.load("en_core_web_sm")
    res = variants_df.copy()
    cleaned_terms = {term: clean_text(nlp, term) for term in variants_df["term"].dropna().unique()}
    cleaned_variants = {
        term: clean_text(nlp, term) for term in variants_df["variants"].dropna().unique()
    }
    res["cleaned_term"] = res["term"].map(cleaned_terms)
    res["cleaned_variants"] = res["variants"].map(cleaned_variants)
    mapping: dict[str, str] = res.set_index("cleaned_variants")["cleaned_term"].to_dict()
    return mapping


def normalize_query(query: str, cleaned_variant_mapping: dict[str, str]) -> str:
    """Ported verbatim from ask-genome-core/src/model/filter_generator.py."""

    nlp = spacy.load("en_core_web_sm")
    cleaned_query = clean_text(nlp, query)
    return _expand_term(cleaned_query, cleaned_variant_mapping)


def compute_spacy_hints(spacy_config: SpacyConfig, query: str) -> dict[str, Any]:
    """Ported verbatim from ask-genome-core/src/model/planner.py:spacy_hints.

    Called from annotate_feasibility.py once per analytical subquery, on that subquery's own
    (self-contained) query text -- not once for the whole original query -- so entities from one
    subquery cannot leak into another subquery's feasibility check.
    """

    cleaned = _clean_variant_mapping(spacy_config.variant_mapping)
    normalized = normalize_query(query.lower(), cleaned)
    return get_spacy_filter(normalized.replace("'", ""), spacy_config)
