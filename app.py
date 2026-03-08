import os
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

# ── Session state init ──────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "llm_choice" not in st.session_state:
    st.session_state.llm_choice = "Claude"
if "embed_log" not in st.session_state:
    st.session_state.embed_log = []  # list of (level, message) persisted across reruns


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
            st.error("ANTHROPIC_API_KEY is not configured. See sidebar for instructions.")
            st.stop()
        return ClaudeLLM()
    else:
        if not os.getenv("OPENAI_API_KEY"):
            st.error("OPENAI_API_KEY is not configured. See sidebar for instructions.")
            st.stop()
        return OpenAILLM()


def format_source(src: dict) -> str:
    name = src.get("source", "unknown")
    if "slide" in src:
        return f"**{name}** — slide {src['slide']}"
    return f"**{name}** — page {src.get('page', '?')}"


def key_status(env_var: str) -> str:
    return "✅ Configured" if os.getenv(env_var) else "❌ Not set"


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

    # ── API key status ──
    if llm_choice == "Claude":
        st.caption(f"Anthropic API key: {key_status('ANTHROPIC_API_KEY')}")
        if not os.getenv("ANTHROPIC_API_KEY"):
            st.info("Add `ANTHROPIC_API_KEY=sk-ant-...` to the `.env` file in the project folder, then restart the app.")
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
            st.info("Add `OPENAI_API_KEY=sk-...` to the `.env` file in the project folder, then restart the app.")
        else:
            if st.button("Test connection", key="test_openai"):
                with st.spinner("Testing…"):
                    ok, msg = test_openai_connection()
                if ok:
                    st.success(msg)
                else:
                    st.error(msg)

    st.divider()

    # ── Document upload ──
    st.subheader("Documents")
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
                    st.session_state.embed_log.append(("warning", f"{f.name}: no text could be extracted (scanned/image file?)."))
                    continue
                progress.progress((i + 0.5) / len(uploaded_files), text=f"Embedding {f.name}...")
                embeddings = embed_texts([c["text"] for c in chunks])
                add_document(chunks, embeddings, f.name)
                st.session_state.embed_log.append(("success", f"{f.name}: {len(chunks)} chunks indexed."))
            except Exception as e:
                st.session_state.embed_log.append(("error", f"{f.name}: {e}"))
        progress.progress(1.0, text="Done!")
        st.rerun()

    for level, msg in st.session_state.embed_log:
        if level == "success":
            st.success(msg)
        elif level == "warning":
            st.warning(msg)
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
        st.rerun()


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
