import numpy as np
import logging

import streamlit as st

from src.integrations.embedding import query_embeddings, get_similarity_score
from src.integrations.mongodb import db
from src.model.insight_report import DocumentRetriever
from src.model.common import tokenize_text

logger = logging.getLogger(__name__)

class ROIGenomeDocumentRetriever(DocumentRetriever):
    def __init__(self, client_code: str, collection_type: str = None):

        if collection_type == "roi-genome-deck":
            self.collection = db.get_collection("roi-genome-deck")
            cursor = self.collection.find()
            docs = [doc for doc in cursor]

            self.doc_id = [doc["_id"] for doc in docs]
            self.text_embeddings = [doc.get("text_embedding", []) for doc in docs]
            self.simple_kbq_embeddings = [doc.get("KBQ_simple_embedding", []) for doc in docs]
            self.nuanced_kbq_embeddings = [doc.get("KBQ_nuanced_embedding", []) for doc in docs]

            self.text = [doc.get("text", "") for doc in docs]
            self.simple_kbq = [doc.get("KBQ_simple", "") for doc in docs]
            self.nuanced_kbq = [doc.get("KBQ_nuanced", "") for doc in docs]

            self.filename = [doc["file"] for doc in docs]
            self.pages = [doc["page"] for doc in docs]

            self.retrieved_docs = []

        else:
            raise RuntimeError(f"Collection type must be specified for DocumentRetriever")

    def retrieve_documents(self, text, use_query, top_k=5):
        tokenized_text = tokenize_text(text)
        token_limit = 514

        if len(tokenized_text) > token_limit:
            keep_tokens = tokenized_text[:token_limit]
            keep_text = text[:keep_tokens[-1]["stop"]]
        else:
            keep_text = text

        target = query_embeddings(keep_text, target_system="insight-report")
        if use_query:
            similarities_simple = np.array([get_similarity_score(target, [embs]) if len(embs) > 0 else np.array([-1])
                                   for embs in self.simple_kbq_embeddings]).flatten()
            similarities_nuanced = np.array([get_similarity_score(target, [embs]) if len(embs) > 0 else np.array([-1])
                                   for embs in self.nuanced_kbq_embeddings]).flatten()
            similarities = [max(a, b) for a, b in zip(similarities_simple, similarities_nuanced)]
        else:
            similarities = np.array([get_similarity_score(target, [embs]) if len(embs) > 0 else np.array([-1])
                            for embs in self.text_embeddings]).flatten()

        top_indices = np.argsort(similarities)[::-1][:top_k]

        for i, idx in enumerate(top_indices):
            result = {
                'doc_id': self.doc_id[idx],
                'document': self.text[idx],
                'similarity': similarities[idx],
                'filename': self.filename[idx],
                'page': self.pages[idx],
                'document_index': idx
            }
            self.retrieved_docs.append(result)
