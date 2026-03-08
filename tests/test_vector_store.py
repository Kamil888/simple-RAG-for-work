import os
import tempfile
import pytest


@pytest.fixture(autouse=True)
def tmp_chroma(monkeypatch, tmp_path):
    """Redirect ChromaDB to a temp directory so tests don't pollute data/."""
    import rag.vector_store as vs
    monkeypatch.setattr(vs, "CHROMA_PATH", str(tmp_path / "chroma_test"))
    vs._client = None
    vs._collection = None
    yield
    vs._client = None
    vs._collection = None


def test_add_and_query():
    from rag.vector_store import add_document, query

    chunks = [
        {"text": "The sky is blue.", "source": "doc.pdf", "page": 1, "chunk_index": 0},
        {"text": "The ocean is deep.", "source": "doc.pdf", "page": 2, "chunk_index": 1},
    ]
    embeddings = [[0.1] * 384, [0.2] * 384]
    add_document(chunks, embeddings, "doc.pdf")

    results = query([0.1] * 384, n_results=1)
    assert len(results) == 1
    assert results[0]["source"] == "doc.pdf"


def test_re_upload_replaces_chunks():
    from rag.vector_store import add_document, query, list_indexed_files

    chunks_v1 = [{"text": "Old content.", "source": "doc.pdf", "page": 1, "chunk_index": 0}]
    chunks_v2 = [{"text": "New content.", "source": "doc.pdf", "page": 1, "chunk_index": 0}]
    emb = [[0.5] * 384]

    add_document(chunks_v1, emb, "doc.pdf")
    add_document(chunks_v2, emb, "doc.pdf")

    indexed = list_indexed_files()
    assert indexed.get("doc.pdf") == 1  # only 1 chunk, not 2

    results = query([0.5] * 384, n_results=1)
    assert results[0]["text"] == "New content."


def test_list_indexed_files_empty():
    from rag.vector_store import list_indexed_files
    assert list_indexed_files() == {}


def test_clear_all():
    from rag.vector_store import add_document, clear_all, list_indexed_files

    chunks = [{"text": "Hello.", "source": "a.pdf", "page": 1, "chunk_index": 0}]
    add_document(chunks, [[0.1] * 384], "a.pdf")
    clear_all()
    assert list_indexed_files() == {}
