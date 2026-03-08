from pypdf import PdfReader

CHUNK_SIZE = 2000
CHUNK_OVERLAP = 200


def _split_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return [c for c in chunks if c.strip()]


def parse_pdf(file_bytes: bytes, filename: str) -> list[dict]:
    reader = PdfReader(file_bytes)
    chunks = []
    chunk_index = 0
    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = text.strip()
        if not text:
            continue
        for fragment in _split_text(text):
            chunks.append({
                "text": fragment,
                "source": filename,
                "page": page_num,
                "chunk_index": chunk_index,
            })
            chunk_index += 1
    return chunks
