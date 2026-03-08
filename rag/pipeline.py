from dataclasses import dataclass, field
from .retriever import retrieve
from .prompt import build_system_prompt
from .llm.base import BaseLLM


@dataclass
class PipelineResult:
    answer: str
    sources: list[dict] = field(default_factory=list)


def _deduplicate_sources(chunks: list[dict]) -> list[dict]:
    seen = set()
    unique = []
    for chunk in chunks:
        source = chunk.get("source", "")
        loc = chunk.get("slide") or chunk.get("page")
        loc_key = ("slide", loc) if "slide" in chunk else ("page", loc)
        key = (source, loc_key)
        if key not in seen:
            seen.add(key)
            entry = {"source": source}
            if "slide" in chunk:
                entry["slide"] = chunk["slide"]
            else:
                entry["page"] = chunk.get("page")
            unique.append(entry)
    return unique


def run_query(user_query: str, llm: BaseLLM, n_results: int = 5) -> PipelineResult:
    chunks = retrieve(user_query, n_results=n_results)
    if not chunks:
        return PipelineResult(
            answer="No documents are indexed yet. Please upload and embed documents first.",
            sources=[],
        )
    system_prompt = build_system_prompt(chunks)
    answer = llm.complete(system_prompt, user_query)
    sources = _deduplicate_sources(chunks)
    return PipelineResult(answer=answer, sources=sources)
