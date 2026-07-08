import os
from typing import List, Union
import pickle
import random
import logging
import requests

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics.pairwise import cosine_similarity
from opentelemetry import trace

from src.data.local import get_client_data_dir_path
from src.utils import bool_from_env
from src.integrations.observability import init_tracer


logger = logging.getLogger(__name__)

init_tracer(os.getenv("APP_ID"), os.getenv("APP_ENV", "DEV"))
tracer = trace.get_tracer(__name__)


def query_embeddings(texts: Union[List[str], str], target_system=None) -> Union[List[List[float]], List[float]]:
    if os.getenv("EMBEDDING_SERVICE_URL") is None:
        raise RuntimeError("EMBEDDING_SERVICE_URL is not defined")

    if target_system == "insight-report":
        embedding_service_url = os.getenv("EMBEDDING_SERVICE_URL_CLIENT_DECK")
    else:
        embedding_service_url = os.getenv("EMBEDDING_SERVICE_URL")

    if isinstance(texts, str):
        texts = [texts]

    with tracer.start_as_current_span("embedding"):
        response = requests.post(
            embedding_service_url,
            headers={"accept": "application/json", "Content-Type": "application/json"},
            json={"inputs": texts}
        )
    response.raise_for_status()
    embeddings = response.json()

    if len(texts) == 1:
        return embeddings[0]

    return embeddings


def get_embedding_file_path() -> str:
    path = os.path.join(get_client_data_dir_path(), "cache", "embeddings.pkl")

    return path


def ensure_cache_dir_exists() -> str:
    cache_path = os.path.join(get_client_data_dir_path(), "cache")
    if not os.path.exists(cache_path):
        os.makedirs(cache_path, exist_ok=True)
        os.chmod(cache_path, 0o777)
    return cache_path


def load_embeddings():
    path = get_embedding_file_path()
    if not os.path.exists(path):
        raise RuntimeError("Embedding cache not found at: {}".format(path))

    with open(path, "rb") as f:
        embeddings = pickle.load(f)

    return embeddings


def save_embeddings(embeddings: List[List[float]]) -> None:
    ensure_cache_dir_exists()  # Ensure 'cache' folder is available
    path = get_embedding_file_path()

    with open(path, "wb") as f:
        pickle.dump(embeddings, f)
        os.chmod(path, 0o666)

    return None


def validate_embeddings(cached_embeddings: List[List[float]], texts: List[str]) -> bool:
    """
    Validate that the cached embeddings match newly fetched embeddings
    Fail when:
    1. The counts do not match
    2. The embedding dimensions differ
    3. The values deviate more than rtol=1e-3 / atol=1e-3

    WARNING: requires a functional embedding service
    """
    if len(cached_embeddings) != len(texts):
        logger.warning("Number of cached embeddings != number of texts")
        return False

    # Get sample and re-create embeddings
    sample_size = min(5, len(texts))
    idxs = random.sample(range(len(texts)), sample_size)
    texts_subset = [texts[i] for i in idxs]
    sample_embeddings_gt = query_embeddings(texts_subset)
    sample_embeddings_cached = [cached_embeddings[i] for i in idxs]

    # Validate embedding dimension
    if len(sample_embeddings_cached[0]) != len(sample_embeddings_gt[0]):
        logger.warning("Embedding dimensionality mismatch")
        return False

    # Validate content
    np_cached = np.array(sample_embeddings_cached, dtype=np.float32)
    np_gt = np.array(sample_embeddings_gt, dtype=np.float32)

    # Single vs. batch mode from embedding host can have small differences
    # Use lenient rtol=1e-3 / atol=1e-3
    if not np.allclose(np_cached, np_gt, rtol=1e-3, atol=1e-3):
        logger.warning("Cached embeddings differ from fetched embeddings beyond tolerance.")
        return False

    return True


def get_cached_embeddings(texts: List[str]) -> List[List[float]]:
    """
    If caching is enabled (EMBEDDING_CACHE=1), load from disk
    If the cache does not exist or fails validation, re-queries embedding service
    """
    if len(texts) < 2:
        raise RuntimeError("Embedding cache requires at least two texts for validation sampling.")

    use_cache = bool_from_env("EMBEDDING_CACHE", default=False)
    use_validation = bool_from_env("EMBEDDING_CACHE_VALIDATION", default=False)

    if use_cache:
        path = get_embedding_file_path()

        if not os.path.exists(path):
            # Cache not exist
            logger.warning("Embedding cache does not exist; creating...")
            embeddings = query_embeddings(texts)
            save_embeddings(embeddings)
        else:
            # Load existing cache
            embeddings = load_embeddings()

        # Validate the cached embeddings
        if use_validation:
            if not validate_embeddings(embeddings, texts):
                logger.warning("Cached embeddings invalid; recreating...")
                embeddings = query_embeddings(texts)
                save_embeddings(embeddings)
    else:
        # No cache usage; always query
        embeddings = query_embeddings(texts)

    return embeddings


def get_similarity_score(target: List[float], embeddings: List[List[float]]) -> NDArray[np.float64]:
    """
    Calculates cosine similarity score between a target embedding and a list of embeddings

    :param target: The target embedding vector
    :param embeddings: A list of embedding vectors to compare against
    :return: A list of cosine similarity scores
    """

    target_vector = np.array(target).reshape(1, -1)
    embeddings_matrix = np.array(embeddings)

    similarities = cosine_similarity(target_vector, embeddings_matrix).flatten()

    return similarities


def get_most_similar(target: List[float], embeddings: List[List[float]]) -> int:
    """
    Finds the index of the most similar vector to the target vector based on cosine similarity.

    :param target: The target embedding vector
    :param embeddings: A list of embedding vectors to compare against
    :return: The index of the most similar vector to the target vector based on cosine similarity
    """
    similarities = get_similarity_score(target, embeddings)
    idx_most_similar = int(np.argmax(similarities))

    return idx_most_similar
