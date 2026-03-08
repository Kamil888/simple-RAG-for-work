import pytest
from rag.llm.base import BaseLLM
from rag.pipeline import run_query, PipelineResult


class MockLLM(BaseLLM):
    def complete(self, system_prompt: str, user_message: str) -> str:
        return f"Answer to: {user_message}"


@pytest.fixture(autouse=True)
def tmp_chroma(monkeypatch, tmp_path):
    import rag.vector_store as vs
    monkeypatch.setattr(vs, "CHROMA_PATH", str(tmp_path / "chroma_test"))
    vs._client = None
    vs._collection = None
    yield
    vs._client = None
    vs._collection = None


def test_run_query_no_documents():
    result = run_query("What is revenue?", MockLLM())
    assert isinstance(result, PipelineResult)
    assert "No documents" in result.answer
    assert result.sources == []


def test_run_query_with_documents():
    from rag.vector_store import add_document

    chunks = [{"text": "Revenue was $10M in Q1.", "source": "report.pdf", "page": 3, "chunk_index": 0}]
    add_document(chunks, [[0.3] * 384], "report.pdf")

    # Patch embed_query to return a fixed vector
    import rag.embedder as emb_mod
    import rag.retriever as ret_mod
    ret_mod.embed_query = lambda q: [0.3] * 384

    result = run_query("What was revenue?", MockLLM())
    assert "Answer to:" in result.answer
    assert len(result.sources) == 1
    assert result.sources[0]["source"] == "report.pdf"
    assert result.sources[0]["page"] == 3
