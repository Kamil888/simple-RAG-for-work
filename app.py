import os
import shutil
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from rag.ingestion.loader import load_and_chunk
from rag.embedder import embed_texts
from rag.vector_store import add_document, list_indexed_files, remove_document, clear_all
from rag.pipeline import run_query
from rag.llm.claude_llm import ClaudeLLM
from rag.llm.openai_llm import OpenAILLM

st.set_page_config(page_title="RAG for Work", page_icon="📄", layout="wide")

SOURCES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "sources")
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".pptx"}

REFRESH_OPTIONS = {
    "Off":           None,
    "Every 5 min":   300,
    "Every 10 min":  600,
    "Every 30 min":  1800,
    "Every 5 hours": 18000,
    "Every day":     86400,
    "Every 2 days":  172800,
    "Every 3 days":  259200,
    "Every week":    604800,
}

# ── Session state ────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "llm_choice" not in st.session_state:
    st.session_state.llm_choice = "Claude"
if "embed_log" not in st.session_state:
    st.session_state.embed_log = []
if "refresh_label" not in st.session_state:
    st.session_state.refresh_label = "Off"
if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = None


# ── Helpers ──────────────────────────────────────────────────────────────────
def save_source(filename: str, data: bytes):
    os.makedirs(SOURCES_DIR, exist_ok=True)
    with open(os.path.join(SOURCES_DIR, filename), "wb") as f:
        f.write(data)


def delete_source(filename: str):
    path = os.path.join(SOURCES_DIR, filename)
    if os.path.exists(path):
        os.remove(path)


def embed_file(filename: str, file_bytes: bytes) -> tuple[str, str]:
    """Chunk, embed, and index a file. Returns (level, message)."""
    chunks = load_and_chunk(file_bytes, filename)
    if not chunks:
        return "warning", f"{filename}: no text extracted (scanned/image file?)."
    embeddings = embed_texts([c["text"] for c in chunks])
    add_document(chunks, embeddings, filename)
    return "success", f"{filename}: {len(chunks)} chunks indexed."


def refresh_all() -> list[tuple[str, str]]:
    """Re-embed every file saved in SOURCES_DIR. Returns (level, message) list."""
    results = []
    if not os.path.exists(SOURCES_DIR):
        return results
    for filename in sorted(os.listdir(SOURCES_DIR)):
        if os.path.splitext(filename)[1].lower() not in SUPPORTED_EXTENSIONS:
            continue
        path = os.path.join(SOURCES_DIR, filename)
        try:
            with open(path, "rb") as f:
                file_bytes = f.read()
            level, msg = embed_file(filename, file_bytes)
            results.append((level, msg))
        except Exception as e:
            results.append(("error", f"{filename}: {e}"))
    return results


def test_claude() -> tuple[bool, str]:
    try:
        import anthropic
        anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY")).messages.create(
            model="claude-haiku-4-5-20251001", max_tokens=10,
            messages=[{"role": "user", "content": "Hi"}],
        )
        return True, "Connected"
    except Exception as e:
        return False, str(e)


def test_openai() -> tuple[bool, str]:
    try:
        import openai
        openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY")).models.list()
        return True, "Connected"
    except Exception as e:
        return False, str(e)


def get_llm():
    if st.session_state.llm_choice == "Claude":
        if not os.getenv("ANTHROPIC_API_KEY"):
            st.error("No Anthropic API key. Add it to .env and restart.")
            st.stop()
        return ClaudeLLM()
    else:
        if not os.getenv("OPENAI_API_KEY"):
            st.error("No OpenAI API key. Add it to .env and restart.")
            st.stop()
        return OpenAILLM()


def format_source(src: dict) -> str:
    name = src.get("source", "unknown")
    return f"**{name}** — slide {src['slide']}" if "slide" in src else f"**{name}** — page {src.get('page', '?')}"


def key_status(var: str) -> str:
    return "✅ Configured" if os.getenv(var) else "❌ Not set"


def show_log():
    for level, msg in st.session_state.embed_log:
        {"success": st.success, "warning": st.warning, "info": st.info}.get(level, st.error)(msg)


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📄 RAG for Work")
    st.divider()

    # LLM
    st.subheader("LLM")
    llm_choice = st.radio(
        "Model", ["Claude", "GPT-4o"],
        index=0 if st.session_state.llm_choice == "Claude" else 1,
        label_visibility="collapsed",
    )
    st.session_state.llm_choice = llm_choice

    if llm_choice == "Claude":
        st.caption(f"API key: {key_status('ANTHROPIC_API_KEY')}")
        if not os.getenv("ANTHROPIC_API_KEY"):
            st.info("Add `ANTHROPIC_API_KEY=sk-ant-...` to `.env`, then restart.")
        elif st.button("Test connection", key="test_claude"):
            with st.spinner("Testing…"):
                ok, msg = test_claude()
            (st.success if ok else st.error)(msg)
    else:
        st.caption(f"API key: {key_status('OPENAI_API_KEY')}")
        if not os.getenv("OPENAI_API_KEY"):
            st.info("Add `OPENAI_API_KEY=sk-...` to `.env`, then restart.")
        elif st.button("Test connection", key="test_openai"):
            with st.spinner("Testing…"):
                ok, msg = test_openai()
            (st.success if ok else st.error)(msg)

    st.divider()

    # Documents
    st.subheader("Documents")
    uploaded_files = st.file_uploader(
        "Upload", type=["pdf", "docx", "pptx"],
        accept_multiple_files=True, label_visibility="collapsed",
    )

    if st.button("Embed Documents", type="primary", disabled=not uploaded_files):
        st.session_state.embed_log = []
        progress = st.progress(0, text="Processing…")
        for i, f in enumerate(uploaded_files):
            progress.progress(i / len(uploaded_files), text=f"Processing {f.name}…")
            try:
                file_bytes = f.read()
                level, msg = embed_file(f.name, file_bytes)
                if level == "success":
                    save_source(f.name, file_bytes)
                st.session_state.embed_log.append((level, msg))
            except Exception as e:
                st.session_state.embed_log.append(("error", f"{f.name}: {e}"))
        progress.progress(1.0, text="Done!")
        st.rerun()

    show_log()

    st.divider()

    # Auto-refresh
    st.subheader("Auto-refresh")
    refresh_label = st.selectbox(
        "Interval", options=list(REFRESH_OPTIONS.keys()),
        index=list(REFRESH_OPTIONS.keys()).index(st.session_state.refresh_label),
        label_visibility="collapsed",
    )
    st.session_state.refresh_label = refresh_label

    if st.button("Refresh Now"):
        with st.spinner("Re-embedding all documents…"):
            results = refresh_all()
        st.session_state.embed_log = results if results else [("info", "No documents to refresh.")]
        st.session_state.last_refresh = datetime.now().strftime("%H:%M:%S")
        st.rerun()

    if st.session_state.last_refresh:
        st.caption(f"Last refresh: {st.session_state.last_refresh}")

    st.divider()

    # Indexed files
    st.subheader("Indexed files")
    indexed = list_indexed_files()
    if indexed:
        for fname, count in sorted(indexed.items()):
            c1, c2 = st.columns([3, 1])
            c1.caption(f"📎 {fname} ({count} chunks)")
            if c2.button("✕", key=f"del_{fname}", help=f"Remove {fname}"):
                remove_document(fname)
                delete_source(fname)
                st.rerun()
    else:
        st.caption("No documents indexed yet.")

    st.divider()
    if st.button("Clear All", type="secondary"):
        clear_all()
        if os.path.exists(SOURCES_DIR):
            shutil.rmtree(SOURCES_DIR)
        st.rerun()


# ── Auto-refresh fragment ────────────────────────────────────────────────────
refresh_interval = REFRESH_OPTIONS[st.session_state.refresh_label]
if refresh_interval is not None:
    @st.fragment(run_every=refresh_interval)
    def _auto_refresh():
        results = refresh_all()
        now = datetime.now().strftime("%H:%M:%S")
        st.session_state.last_refresh = now
        if results:
            for lvl, msg in results:
                icon = "✅" if lvl == "success" else ("⚠️" if lvl == "warning" else "❌")
                st.toast(f"{icon} {msg}")
            st.session_state.embed_log = results
            st.rerun(scope="app")

    _auto_refresh()


# ── Chat ─────────────────────────────────────────────────────────────────────
st.header("Ask your documents")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for src in msg["sources"]:
                    st.markdown(format_source(src))

if prompt := st.chat_input("Ask a question about your documents…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                result = run_query(prompt, get_llm())
                st.markdown(result.answer)
                if result.sources:
                    with st.expander("Sources"):
                        for src in result.sources:
                            st.markdown(format_source(src))
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result.answer,
                    "sources": result.sources,
                })
            except Exception as e:
                st.error(f"Error: {e}")
