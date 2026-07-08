import numpy as np
import logging

import streamlit as st

from src.integrations.embedding import query_embeddings, get_similarity_score
from src.integrations.mongodb import db
from src.data.data_interface import InsightsReportConfig


logger = logging.getLogger(__name__)

CANDIDATE_POOL = 70


def check_client_deck_data_exists(client_code: str) -> bool:
    collection = db["client-deck"]
    client_codes = collection.distinct("client_code")

    if client_code in client_codes:
        return True

    return False


def if_query_insight_reports(client_code: str, model_group_id: int, ner_filters: dict) -> bool:
    if not check_client_deck_data_exists(client_code):
        return False

    insights_report_config = InsightsReportConfig.from_local(client_code, model_group_id)
    trigger_dict = insights_report_config.triggers

    na_terms = ner_filters.get("terms_from_insights", [])
    intention = ner_filters.get("intention", [])[0]
    has_diag_tagging = bool(ner_filters.get("diag_tagging", []))

    intention_na_words = set(na_terms) - set(trigger_dict.get(intention, []))

    if intention_na_words:
        return True

    if "planner" in intention and has_diag_tagging:
        return True

    # demo purpose
    if client_code == "INDEED":
        return True

    return False


class DocumentRetriever:
    def __init__(self, client_code: str, collection_type: str = None):
        if collection_type == "client-deck":
            self.collection = db.get_collection("client-deck")

            if check_client_deck_data_exists(client_code):
                cursor = self.collection.find(
                    {"client_code": client_code},
                    {
                        "embedding": 1,
                        "text": 1,
                        "file": 1,
                        "page": 1,
                        "core_dimension_text": 1,
                        'halo_level': 1,
                        'halo_tagging': 1,
                        'custom_level': 1,
                        'custom_tagging': 1,
                    },
                )

                docs = [doc for doc in cursor]

                self.doc_id = [doc["_id"] for doc in docs]
                self.doc_embeddings = [doc.get("embedding", []) for doc in docs]
                self.documents = [doc.get("text", "") for doc in docs]
                self.filename = [doc["file"] for doc in docs]
                self.pages = [doc["page"] for doc in docs]
                self.core_dimension_text = [doc.get("core_dimension_text", "") for doc in docs]
                self.halo_level_text = [doc.get("halo_level", "") for doc in docs]
                self.halo_tagging_text = [doc.get("halo_tagging", "") for doc in docs]
                self.halo_text = [(a + ", " + b) if (a and b) else None
                                  for a, b in zip(self.halo_level_text, self.halo_tagging_text)]
                self.custom_level_text = [doc.get("custom_level", "") for doc in docs]
                self.custom_tagging_text = [doc.get("custom_tagging", "") for doc in docs]

                self.retrieved_docs = []
            else:
                raise RuntimeError(f"client_code={client_code} collection_type={collection_type} does not exist")
        else:
            raise RuntimeError(f"Collection type must be specified for DocumentRetriever")

    def retrieve_insights_documents(self, query, ner_filter, top_k=5):
        target = query_embeddings(query, target_system="insight-report")
        na_terms = ner_filter.get("terms_from_insights", [])
        core_dimensions = ner_filter.get("core_dimension", {})
        core_dimension_composite = ner_filter.get("core_dimension_composite", {})
        custom_level = ner_filter.get("custom_level", {})
        custom_tagging = ner_filter.get("custom_tagging", {})
        halo_level_col = ner_filter["level"].get("halo_level", [])
        halo_tagging_col = ner_filter["level"].get("halo_tagging", [])
        halo_level, halo_tagging = [], []
        if halo_level_col:
            halo_level = [x for x in ner_filter.get(halo_level_col[0], []) if x != "overall"]
        if halo_tagging_col:
            halo_tagging = [x for x in ner_filter.get(halo_tagging_col[0], []) if x != "overall"]

        # Priority-ordered buckets: highest priority first; lowest dropped first during relaxation.
        # (bucket_name, slide_text_list, query_value_set)
        bucket_defs = [
            ("halo", self.halo_text, halo_level + halo_tagging),
            ("core_dimension_text", self.core_dimension_text,
             na_terms + list(core_dimensions.keys()) + list(core_dimension_composite.keys())),
            ("custom_tagging", self.custom_tagging_text, list(custom_tagging.keys())),
            ("custom_level", self.custom_level_text, list(custom_level.keys())),
        ]

        all_indices = list(range(len(self.doc_embeddings)))

        # Active only if it has values and is not an "overall" no-op.
        active_buckets = [
            (name, text_list, values)
            for name, text_list, values in bucket_defs
            if values and "overall" not in values
        ]

        bucket_match_sets = {
            name: set(_prefilter_slides(all_indices, text_list, values))
            for name, text_list, values in active_buckets
        }

        # Gradual relaxation: AND across kept buckets; drop lowest priority first until non-empty.
        kept = [name for name, _, _ in active_buckets]
        dropped = []
        candidate_pool = []
        while kept:
            pool = set(all_indices)
            for name in kept:
                pool &= bucket_match_sets[name]
            if pool:
                candidate_pool = sorted(pool)
                break
            dropped.append(kept.pop())

        if not candidate_pool:
            candidate_pool = all_indices
            dropped = []

        # CANDIDATE_POOL cap; prefer candidates matching more dropped buckets.
        if len(candidate_pool) > CANDIDATE_POOL:
            if dropped:
                dropped_value_map = {
                    name: (text_list, values)
                    for name, text_list, values in bucket_defs
                    if name in dropped
                }

                def _dropped_score(idx):
                    return sum(
                        1
                        for _tl, _vals in dropped_value_map.values()
                        if _slide_matches_bucket(_tl[idx], _vals)
                    )

                candidate_pool = sorted(candidate_pool, key=_dropped_score, reverse=True)[:CANDIDATE_POOL]
            else:
                candidate_pool = candidate_pool[:CANDIDATE_POOL]

        filtered_indices = candidate_pool
        filtered_embeddings = [self.doc_embeddings[i] for i in filtered_indices]

        # similarities = get_similarity_score(target, filtered_embeddings)
        similarities = np.array([get_similarity_score(target, [embs]) if len(embs) > 0 else np.array([-1])
                            for embs in filtered_embeddings]).flatten()
        top_filtered_indices = np.argsort(similarities)[::-1][:top_k]
        top_indices = [filtered_indices[i] for i in top_filtered_indices]

        for i, idx in enumerate(top_indices):
            result = {
                'doc_id': self.doc_id[idx],
                'document': self.documents[idx],
                'similarity': similarities[top_filtered_indices[i]],
                'filename': self.filename[idx],
                'page': self.pages[idx],
                'document_index': idx
            }
            self.retrieved_docs.append(result)

    def display_retrieved_documents(self, threshold=0.86):
        high_score_results = [r for r in self.retrieved_docs if r['similarity'] > threshold]

        if high_score_results:
            for i, result in enumerate(high_score_results[:3]):
                with st.expander(f"Document {i + 1} (Score: {result['similarity']:.3f})", expanded=False):
                    st.markdown(result['document'])
                    st.markdown(f"**Source:** {result['filename']}, Page {result['page']}")

                    doc = self.collection.find_one({"_id": result["doc_id"]})
                    st.image(bytes(doc["img"]), width=400, caption=f"Slide {result['page']}")
            return high_score_results
        else:
            # st.write("Apologies, we do not have any relevant documents to answer your query at the moment.")
            st.write("Apologies, we do not have any documents relevant enough to answer your query at the moment.")
            st.write("Here are top three related results for your reference. ")
            for i, result in enumerate(self.retrieved_docs[:3]):
                with st.expander(f"Document {i + 1} (Score: {result['similarity']:.3f})", expanded=False):
                    st.markdown(result['document'])
                    st.markdown(f"**Source:** {result['filename']}, Page {result['page']}")

                    doc = self.collection.find_one({"_id": result["doc_id"]})
                    st.image(bytes(doc["img"]), width=400, caption=f"Slide {result['page']}")
            return high_score_results


def _slide_matches_bucket(text: str, target_values: list) -> bool:
    """True if any target value (lowercased) is in the slide's comma-split lowercased text.
    Mirrors _prefilter_slides per-slide logic; used for the dropped-bucket tiebreak."""
    _t = text.lower() if isinstance(text, str) else ""
    _l = [x.strip() for x in _t.split(",")]
    return any(v.lower() in _l for v in target_values)


def _prefilter_slides(filtered_indices: list, text_list: list, target_list: list) -> list:
    if not target_list or len(target_list) == 0:
        return filtered_indices
    if "overall" in target_list:
        return filtered_indices

    res = []
    for idx in filtered_indices:
        text = text_list[idx]
        _t = text.lower() if isinstance(text, str) else ""
        _l = [x.strip() for x in _t.split(",")]
        if any(term.lower() in _l for term in target_list):
            res.append(idx)
    return res
