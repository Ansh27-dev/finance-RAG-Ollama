from pathlib import Path

import streamlit as st

from src.config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    DEFAULT_CHAT_MODEL,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_EMBED_MODEL,
    DEFAULT_OLLAMA_HOST,
    DEFAULT_TOP_K,
    PDF_DIR,
)
from src.ollama_client import OllamaError
from src.rag import FinanceRAG


st.set_page_config(page_title="Finance RAG - Ollama", layout="wide")


def get_rag() -> FinanceRAG:
    return FinanceRAG(
        ollama_host=st.session_state.ollama_host,
        chat_model=st.session_state.chat_model,
        embed_model=st.session_state.embed_model,
        collection_name=COLLECTION_NAME,
        persist_dir=CHROMA_DIR,
    )


def save_uploaded_files(uploaded_files) -> list[Path]:
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    saved_paths = []
    for uploaded_file in uploaded_files:
        safe_name = Path(uploaded_file.name).name
        target = PDF_DIR / safe_name
        target.write_bytes(uploaded_file.getbuffer())
        saved_paths.append(target)
    return saved_paths


if "history" not in st.session_state:
    st.session_state.history = []
if "indexed" not in st.session_state:
    st.session_state.indexed = False
if "ollama_host" not in st.session_state:
    st.session_state.ollama_host = DEFAULT_OLLAMA_HOST
if "chat_model" not in st.session_state:
    st.session_state.chat_model = DEFAULT_CHAT_MODEL
if "embed_model" not in st.session_state:
    st.session_state.embed_model = DEFAULT_EMBED_MODEL


st.title("Finance RAG for Quarterly Reports")

with st.sidebar:
    st.header("Ollama")
    st.text_input("Host", key="ollama_host")
    st.text_input("Chat model", key="chat_model")
    st.text_input("Embedding model", key="embed_model")
    top_k = st.slider("Retrieved chunks", 2, 8, DEFAULT_TOP_K)

    st.header("Chunking")
    chunk_size = st.slider("Chunk size", 800, 1600, DEFAULT_CHUNK_SIZE, step=100)
    chunk_overlap = st.slider(
        "Chunk overlap",
        100,
        300,
        DEFAULT_CHUNK_OVERLAP,
        step=20,
    )

left, right = st.columns([0.42, 0.58], gap="large")

with left:
    st.subheader("1. Upload and Index")
    uploaded_files = st.file_uploader(
        "Upload 3-4 quarterly financial report PDFs",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if st.button("Index Uploaded PDFs", type="primary", use_container_width=True):
        if not uploaded_files:
            st.warning("Upload at least one PDF before indexing.")
        else:
            try:
                saved_paths = save_uploaded_files(uploaded_files)
                with st.status("Indexing PDFs with Ollama embeddings...", expanded=True):
                    st.write("Extracting selectable text from PDFs")
                    rag = get_rag()
                    stats = rag.ingest_pdfs(
                        saved_paths,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        reset_collection=True,
                    )
                    st.write("Saving chunks to persistent ChromaDB")
                st.session_state.indexed = True
                st.success(
                    f"Indexed {stats['files_processed']} files, "
                    f"{stats['pages_with_text']} text pages, "
                    f"{stats['chunks_indexed']} chunks."
                )
            except (OllamaError, ValueError) as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error(f"Indexing failed: {exc}")

    st.subheader("Store Status")
    try:
        current_stats = get_rag().stats()
        st.metric("Chunks in Chroma", current_stats["chunk_count"])
        st.caption(f"Collection: {current_stats['collection_name']}")
        st.caption(f"Persistence: {current_stats['persist_dir']}")
        if current_stats["chunk_count"] > 0:
            st.session_state.indexed = True
    except Exception as exc:
        st.caption(f"Store not ready: {exc}")

with right:
    st.subheader("2. Ask Questions")
    question = st.text_area(
        "Question",
        placeholder="Example: What was the revenue in the latest quarter?",
        height=100,
    )

    ask_disabled = not st.session_state.indexed
    if st.button("Ask", disabled=ask_disabled, use_container_width=True):
        if not question.strip():
            st.warning("Type a question first.")
        else:
            try:
                with st.spinner("Retrieving relevant chunks and generating answer..."):
                    result = get_rag().answer(question.strip(), top_k=top_k)
                st.session_state.history.insert(
                    0,
                    {
                        "question": question.strip(),
                        "answer": result["answer"],
                        "sources": result["sources"],
                        "retrieved_chunks": result["retrieved_chunks"],
                    },
                )
            except (OllamaError, ValueError) as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error(f"Answering failed: {exc}")

    if ask_disabled:
        st.info("Index PDFs first, then questions will be enabled.")

    for item_index, item in enumerate(st.session_state.history, start=1):
        with st.container(border=True):
            st.markdown(f"**Q{item_index}. {item['question']}**")
            st.write(item["answer"])

            st.markdown("**Sources**")
            for source in item["sources"]:
                st.write(
                    f"- {source['file_name']} | page {source['page_number']} | "
                    f"{source['quarter']}"
                )

            with st.expander("Retrieved chunks"):
                for chunk in item["retrieved_chunks"]:
                    metadata = chunk["metadata"]
                    st.caption(
                        f"{metadata['file_name']} - page {metadata['page_number']} "
                        f"- distance {chunk['distance']:.4f}"
                    )
                    st.text(chunk["text"][:1500])

