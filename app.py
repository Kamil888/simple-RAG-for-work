import json
import os
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

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
PATHS_FILE = os.path.join(DATA_DIR, ".paths.json")      # {filename: full_path}
MTIME_FILE = os.path.join(DATA_DIR, ".mtime_index.json") # {full_path: mtime}

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


# ── Persistence ──────────────────────────────────────────────────────────────
def _load(path: str) -> dict:
    if os.path.exists(path):
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save(path: str, data: dict):
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# ── Core embed / refresh logic ───────────────────────────────────────────────
def embed_from_path(filepath: str) -> tuple[str, str]:
    """Read file from disk, chunk, embed, and store. Returns (level, message)."""
    filename = os.path.basename(filepath)
    with open(filepath, "rb") as f:
        file_bytes = f.read()
    chunks = load_and_chunk(file_bytes, filename)
    if not chunks:
        return "warning", f"{filename}: no text could be extracted (scanned/image?)."
    embeddings = embed_texts([c["text"] for c in chunks])
    add_document(chunks, embeddings, filename)
    return "success", f"{filename}: {len(chunks)} chunks indexed."


def refresh_changed(force: bool = False) -> list[tuple[str, str]]:
    """
    Re-embed files whose mtime changed since last embed.
    If force=True, re-embed all regardless of mtime (used by Refresh Now).
    """
    paths = _load(PATHS_FILE)   # {filename: full_path}
    mtimes = _load(MTIME_FILE)  # {full_path: last_embedded_mtime}
    results = []

    for filename, filepath in paths.items():
        if not os.path.exists(filepath):
            results.append(("error", f"{filename}: original file not found at {filepath}"))
            continue
        current_mtime = os.path.getmtime(filepath)
        if not force and current_mtime <= mtimes.get(filepath, 0):
            continue  # unchanged — skip
        try:
            level, msg = embed_from_path(filepath)
            if level == "success":
                mtimes[filepath] = current_mtime
                _save(MTIME_FILE, mtimes)
            results.append((level, msg))
        except Exception as e:
            results.append(("error", f"{filename}: {e}"))

    return results


# ── LLM helpers ──────────────────────────────────────────────────────────────
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
if "_last_auto_refresh_ts" not in st.session_state:
    st.session_state._last_auto_refresh_ts = datetime.now().timestamp()


# ── Auto-refresh fragment ────────────────────────────────────────────────────
# Defined here (before sidebar) so it can be called inside `with st.sidebar`.
# Polls every 60 s; manual timestamp check enforces the user-selected interval
# so full-page reruns triggered by user interactions don't reset the clock.
@st.fragment(run_every=60)
def _auto_refresh():
    interval = REFRESH_OPTIONS[st.session_state.refresh_label]
    if interval is not None:
        now = datetime.now().timestamp()
        if now - st.session_state._last_auto_refresh_ts >= interval:
            results = refresh_changed(force=False)
            st.session_state._last_auto_refresh_ts = now
            st.session_state.last_refresh = datetime.now().strftime("%H:%M:%S")
            for lvl, msg in results:
                icon = "✅" if lvl == "success" else ("⚠️" if lvl == "warning" else "❌")
                st.toast(f"{icon} {msg}")
    if st.session_state.last_refresh:
        st.caption(f"Last refresh: {st.session_state.last_refresh}")


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
    st.caption("Paste the full path to a file on your computer.")
    new_path = st.text_input(
        "File path",
        placeholder=r"e.g. C:\Users\You\Documents\report.docx",
        label_visibility="collapsed",
    )

    if st.button("Add & Index", type="primary", disabled=not new_path.strip()):
        filepath = new_path.strip().strip("\"'")
        filename = os.path.basename(filepath)
        st.session_state.embed_log = []
        if not os.path.exists(filepath):
            st.session_state.embed_log = [("error", f"File not found: {filepath}")]
        elif os.path.splitext(filename)[1].lower() not in {".pdf", ".docx", ".pptx"}:
            st.session_state.embed_log = [("error", "Unsupported file type. Use PDF, DOCX, or PPTX.")]
        else:
            with st.spinner(f"Indexing {filename}…"):
                try:
                    level, msg = embed_from_path(filepath)
                    if level == "success":
                        paths = _load(PATHS_FILE)
                        paths[filename] = filepath
                        _save(PATHS_FILE, paths)
                        mtimes = _load(MTIME_FILE)
                        mtimes[filepath] = os.path.getmtime(filepath)
                        _save(MTIME_FILE, mtimes)
                    st.session_state.embed_log = [(level, msg)]
                except Exception as e:
                    st.session_state.embed_log = [("error", f"{filename}: {e}")]
        st.rerun()

    show_log()

    st.divider()

    # Auto-refresh
    st.subheader("Auto-refresh")
    refresh_label = st.selectbox(
        "Interval",
        options=list(REFRESH_OPTIONS.keys()),
        index=list(REFRESH_OPTIONS.keys()).index(st.session_state.refresh_label),
        label_visibility="collapsed",
    )
    st.session_state.refresh_label = refresh_label

    if st.button("Refresh Now"):
        with st.spinner("Re-embedding all documents…"):
            results = refresh_changed(force=True)
        st.session_state.embed_log = results if results else [("info", "No documents indexed yet.")]
        st.session_state.last_refresh = datetime.now().strftime("%H:%M:%S")
        st.rerun()

    _auto_refresh()  # renders caption + runs auto-refresh logic; updates in place

    st.divider()

    # Indexed files
    st.subheader("Indexed files")
    indexed = list_indexed_files()
    paths = _load(PATHS_FILE)
    if indexed:
        for fname, count in sorted(indexed.items()):
            c1, c2 = st.columns([3, 1])
            c1.caption(f"📎 {fname} ({count} chunks)")
            if c2.button("✕", key=f"del_{fname}", help=f"Remove {fname}"):
                remove_document(fname)
                paths.pop(fname, None)
                _save(PATHS_FILE, paths)
                st.rerun()
    else:
        st.caption("No documents indexed yet.")

    st.divider()
    if st.button("Clear All", type="secondary"):
        clear_all()
        _save(PATHS_FILE, {})
        _save(MTIME_FILE, {})
        st.rerun()


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
