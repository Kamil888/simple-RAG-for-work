from .embedder import embed_query
from .vector_store import query as vs_query


def retrieve(user_query: str, n_results: int = 5) -> list[dict]:
    embedding = embed_query(user_query)
    return vs_query(embedding, n_results=n_results)
