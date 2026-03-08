import chromadb

CHROMA_PATH = "./data/chroma_db"
COLLECTION_NAME = "documents"

_client = None
_collection = None


def _get_collection():
    global _client, _collection
    if _collection is None:
        _client = chromadb.PersistentClient(path=CHROMA_PATH)
        _collection = _client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def add_document(chunks: list[dict], embeddings: list[list[float]], filename: str) -> None:
    col = _get_collection()
    # Delete existing chunks for this file before re-adding
    try:
        col.delete(where={"source": filename})
    except Exception:
        pass  # collection may be empty; ChromaDB raises if no matching docs

    if not chunks:
        return

    ids = [f"{filename}::chunk::{i}" for i in range(len(chunks))]
    col.add(
        ids=ids,
        embeddings=embeddings,
        documents=[c["text"] for c in chunks],
        metadatas=chunks,
    )


def query(query_embedding: list[float], n_results: int = 5) -> list[dict]:
    col = _get_collection()
    total = col.count()
    if total == 0:
        return []
    n = min(n_results, total)
    results = col.query(
        query_embeddings=[query_embedding],
        n_results=n,
        include=["documents", "metadatas", "distances"],
    )
    output = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        entry = dict(meta)
        entry["text"] = doc
        entry["score"] = round(1 - dist, 4)  # cosine distance → similarity
        output.append(entry)
    return output


def list_indexed_files() -> dict[str, int]:
    col = _get_collection()
    if col.count() == 0:
        return {}
    all_meta = col.get(include=["metadatas"])["metadatas"]
    counts: dict[str, int] = {}
    for meta in all_meta:
        src = meta.get("source", "unknown")
        counts[src] = counts.get(src, 0) + 1
    return counts


def remove_document(filename: str) -> None:
    col = _get_collection()
    try:
        col.delete(where={"source": filename})
    except Exception:
        pass


def clear_all() -> None:
    global _collection
    if _client is not None:
        _client.delete_collection(COLLECTION_NAME)
        _collection = None
