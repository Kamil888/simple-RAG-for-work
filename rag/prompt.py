SYSTEM_PROMPT_TEMPLATE = """\
You are a helpful assistant that answers questions based strictly on the provided document context.

Rules:
- Answer only using information from the context below.
- If the answer is not in the context, say "I could not find relevant information in the provided documents."
- When you use information from a source, cite it inline using the exact format shown, e.g. [report.pdf, page 7].
- Be concise and factual.

Context:
{context}
"""


def build_system_prompt(chunks: list[dict]) -> str:
    parts = []
    for chunk in chunks:
        source = chunk.get("source", "unknown")
        if "slide" in chunk:
            loc = f"slide {chunk['slide']}"
        else:
            loc = f"page {chunk.get('page', '?')}"
        parts.append(f"[Source: {source}, {loc}]\n{chunk['text']}")
    context = "\n\n".join(parts)
    return SYSTEM_PROMPT_TEMPLATE.format(context=context)
