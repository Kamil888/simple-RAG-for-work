from .pdf_parser import parse_pdf
from .docx_parser import parse_docx
from .pptx_parser import parse_pptx


def load_and_chunk(file_bytes: bytes, filename: str) -> list[dict]:
    ext = filename.rsplit(".", 1)[-1].lower()
    if ext == "pdf":
        return parse_pdf(file_bytes, filename)
    elif ext == "docx":
        return parse_docx(file_bytes, filename)
    elif ext == "pptx":
        return parse_pptx(file_bytes, filename)
    else:
        raise ValueError(f"Unsupported file type: .{ext}")
