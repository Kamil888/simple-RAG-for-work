import json
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

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
MTIME_INDEX_PATH = os.path.join(DATA_DIR, ".mtime_index.json")
CONFIG_PATH = os.path.join(DATA_DIR, ".config.json")
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".pptx"}

REFRESH_OPTIONS = {
    "Off":            None,
    "Every 5 min":    300,
    "Every 10 min":   600,
    "Every 30 min":   1800,
    "Every 5 hours":  18000,
    "Every day":      86400,
    "Every 2 days":   172800,
    "Every 3 days":   259200,
    "Every week":     604800,
}


# ── Persistence helpers ──────────────────────────────────────────────────────
def load_config() -> dict:
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_config(config: dict):
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f)


def load_mtime_index() -> dict:
    if os.path.exists(MTIME_INDEX_PATH):
        try:
            with open(MTIME_INDEX_PATH) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_mtime_index(index: dict):
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(MTIME_INDEX_PATH, "w") as f:
        json.dump(index, f)


# ── Watch-folder refresh ─────────────────────────────────────────────────────
def refresh_watch_folder(watch_folder: str) -> list[tuple[str, str, str]]:
    """
    Scan watch_folder for supported files. Re-embed any whose mtime changed
    since the last successful embed. Returns (level, filename, message) tuples.
    """
    results = []
    if not watch_folder or not os.path.isdir(watch_folder):
        return results

    mtime_index = load_mtime_index()

    for filename in sorted(os.listdir(watch_folder)):
        ext = os.path.splitext(filename)[1].lower()
        if ext not in SUPPORTED_EXTENSIONS:
            continue
        path = os.path.join(watch_folder, filename)
        try:
            current_mtime = os.path.getmtime(path)
        except OSError:
            continue

        index_key = os.path.join(watch_folder, filename)  # full path as key
        if current_mtime <= mtime_index.get(index_key, 0):
            continue  # unchanged since last embed

        try:
            with open(path, "rb") as f:
                file_bytes = f.read()
            chunks = load_and_chunk(file_bytes, filename)
            if not chunks:
                results.append(("warning", filename, "no text extracted"))
                continue
            embeddings = embed_texts([c["text"] for c in chunks])
            add_document(chunks, embeddings, filename)
            mtime_index[index_key] = current_mtime
            save_mtime_index(mtime_index)
            results.append(("success", filename, f"{len(chunks)} chunks re-indexed"))
        except Exception as e:
            results.append(("error", filename, str(e)))

    return results


# ── LLM helpers ─────────────────────────────────────────────────────────────
def test_claude_connection() -> tuple[bool, str]:
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=10,
            messages=[{"role": "user", "content": "Hi"}],
        )
        return True, "Connected"
    except Exception as e:
        return False, str(e)


def test_openai_connection() -> tuple[bool, str]:
    try:
        import openai
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        client.models.list()
        return True, "Connected"
    except Exception as e:
        return False, str(e)


def get_llm():
    choice = st.session_state.llm_choice
    if choice == "Claude":
        if not os.getenv("ANTHROPIC_API_KEY"):
            st.error("No Anthropic API key set. See sidebar for instructions.")
            st.stop()
        return ClaudeLLM()
    else:
        if not os.getenv("OPENAI_API_KEY"):
            st.error("No OpenAI API key set. See sidebar for instructions.")
            st.stop()
        return OpenAILLM()


def format_source(src: dict) -> str:
    name = src.get("source", "unknown")
    if "slide" in src:
        return f"**{name}** — slide {src['slide']}"
    return f"**{name}** — page {src.get('page', '?')}"


def key_status(env_var: str) -> str:
    return "✅ Configured" if os.getenv(env_var) else "❌ Not set"


# ── Session state init ──────────────────────────────────────────────────────
_cfg = load_config()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "llm_choice" not in st.session_state:
    st.session_state.llm_choice = "Claude"
if "embed_log" not in st.session_state:
    st.session_state.embed_log = []
if "refresh_label" not in st.session_state:
    st.session_state.refresh_label = _cfg.get("refresh_label", "Off")
if "watch_folder" not in st.session_state:
    st.session_state.watch_folder = _cfg.get("watch_folder", "")
if "last_auto_check" not in st.session_state:
    st.session_state.last_auto_check = None


# ── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📄 RAG for Work")
    st.divider()

    # ── LLM selector ──
    st.subheader("LLM")
    llm_choice = st.radio(
        "Select model",
        ["Claude", "GPT-4o"],
        index=0 if st.session_state.llm_choice == "Claude" else 1,
        label_visibility="collapsed",
    )
    st.session_state.llm_choice = llm_choice

    if llm_choice == "Claude":
        st.caption(f"Anthropic API key: {key_status('ANTHROPIC_API_KEY')}")
        if not os.getenv("ANTHROPIC_API_KEY"):
            st.info("Add `ANTHROPIC_API_KEY=sk-ant-...` to the `.env` file, then restart.")
        else:
            if st.button("Test connection", key="test_claude"):
                with st.spinner("Testing…"):
                    ok, msg = test_claude_connection()
                if ok:
                    st.success(msg)
                else:
                    st.error(msg)
    else:
        st.caption(f"OpenAI API key: {key_status('OPENAI_API_KEY')}")
        if not os.getenv("OPENAI_API_KEY"):
            st.info("Add `OPENAI_API_KEY=sk-...` to the `.env` file, then restart.")
        else:
            if st.button("Test connection", key="test_openai"):
                with st.spinner("Testing…"):
                    ok, msg = test_openai_connection()
                if ok:
                    st.success(msg)
                else:
                    st.error(msg)

    st.divider()

    # ── Watch folder ──
    st.subheader("Watch folder")
    st.caption("Point to the folder where your source files live. The app re-embeds any file you edit and save.")
    watch_folder_input = st.text_input(
        "Folder path",
        value=st.session_state.watch_folder,
        placeholder=r"e.g. C:\Users\You\Documents\work-docs",
        label_visibility="collapsed",
    )
    if watch_folder_input != st.session_state.watch_folder:
        st.session_state.watch_folder = watch_folder_input
        cfg = load_config()
        cfg["watch_folder"] = watch_folder_input
        save_config(cfg)

    if watch_folder_input:
        if os.path.isdir(watch_folder_input):
            files_found = [
                f for f in os.listdir(watch_folder_input)
                if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS
            ]
            st.caption(f"✅ Valid folder — {len(files_found)} supported file(s) found.")
        else:
            st.warning("Folder not found. Check the path.")

    # ── Auto-refresh interval ──
    st.subheader("Auto-refresh")
    refresh_label = st.selectbox(
        "Re-embed when files change",
        options=list(REFRESH_OPTIONS.keys()),
        index=list(REFRESH_OPTIONS.keys()).index(st.session_state.refresh_label),
        label_visibility="collapsed",
        disabled=not (watch_folder_input and os.path.isdir(watch_folder_input)),
    )
    if refresh_label != st.session_state.refresh_label:
        st.session_state.refresh_label = refresh_label
        cfg = load_config()
        cfg["refresh_label"] = refresh_label
        save_config(cfg)

    if st.button("Refresh Now"):
        with st.spinner("Scanning watch folder…"):
            results = refresh_watch_folder(st.session_state.watch_folder)
        if results:
            st.session_state.embed_log = [
                (lvl, f"{fname}: {msg}") for lvl, fname, msg in results
            ]
        else:
            st.session_state.embed_log = [("info", "No changes detected.")]
        st.rerun()

    if st.session_state.last_auto_check:
        st.caption(f"Last auto-check: {st.session_state.last_auto_check}")

    st.divider()

    # ── Manual upload ──
    st.subheader("One-off upload")
    st.caption("For files not in the watch folder. These won't auto-refresh.")
    uploaded_files = st.file_uploader(
        "Upload documents",
        type=["pdf", "docx", "pptx"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if st.button("Embed Documents", type="primary", disabled=not uploaded_files):
        st.session_state.embed_log = []
        progress = st.progress(0, text="Processing...")
        for i, f in enumerate(uploaded_files):
            progress.progress(i / len(uploaded_files), text=f"Parsing {f.name}...")
            try:
                file_bytes = f.read()
                chunks = load_and_chunk(file_bytes, f.name)
                if not chunks:
                    st.session_state.embed_log.append(
                        ("warning", f"{f.name}: no text extracted (scanned/image file?).")
                    )
                    continue
                progress.progress((i + 0.5) / len(uploaded_files), text=f"Embedding {f.name}...")
                embeddings = embed_texts([c["text"] for c in chunks])
                add_document(chunks, embeddings, f.name)
                st.session_state.embed_log.append(
                    ("success", f"{f.name}: {len(chunks)} chunks indexed.")
                )
            except Exception as e:
                st.session_state.embed_log.append(("error", f"{f.name}: {e}"))
        progress.progress(1.0, text="Done!")
        st.rerun()

    for level, msg in st.session_state.embed_log:
        if level == "success":
            st.success(msg)
        elif level == "warning":
            st.warning(msg)
        elif level == "info":
            st.info(msg)
        else:
            st.error(msg)

    st.divider()

    # ── Indexed files ──
    st.subheader("Indexed files")
    indexed = list_indexed_files()
    if indexed:
        for fname, count in sorted(indexed.items()):
            col1, col2 = st.columns([3, 1])
            col1.caption(f"📎 {fname} ({count} chunks)")
            if col2.button("✕", key=f"del_{fname}", help=f"Remove {fname}"):
                remove_document(fname)
                st.rerun()
    else:
        st.caption("No documents indexed yet.")

    st.divider()
    if st.button("Clear All Documents", type="secondary"):
        clear_all()
        save_mtime_index({})
        st.rerun()


# ── Auto-refresh fragment ────────────────────────────────────────────────────
refresh_interval = REFRESH_OPTIONS[st.session_state.refresh_label]
watch_folder = st.session_state.watch_folder

if refresh_interval is not None and watch_folder and os.path.isdir(watch_folder):
    @st.fragment(run_every=refresh_interval)
    def _auto_refresh():
        results = refresh_watch_folder(watch_folder)
        st.session_state.last_auto_check = datetime.now().strftime("%H:%M:%S")
        if results:
            for lvl, fname, msg in results:
                icon = "✅" if lvl == "success" else ("⚠️" if lvl == "warning" else "❌")
                st.toast(f"{icon} {fname}: {msg}")
            st.session_state.embed_log = [
                (lvl, f"{fname} (auto): {msg}") for lvl, fname, msg in results
            ]
            st.rerun(scope="app")

    _auto_refresh()


# ── Main chat area ───────────────────────────────────────────────────────────
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
                llm = get_llm()
                result = run_query(prompt, llm)
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
