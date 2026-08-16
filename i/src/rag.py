from pathlib import Path
from typing import Any

from src.chunking import chunk_pages
from src.config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    DEFAULT_CHAT_MODEL,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_EMBED_MODEL,
    DEFAULT_OLLAMA_HOST,
)
from src.ollama_client import OllamaClient
from src.pdf_loader import extract_pdf_pages
from src.vector_store import FinanceVectorStore


class FinanceRAG:
    def __init__(
        self,
        ollama_host: str = DEFAULT_OLLAMA_HOST,
        chat_model: str = DEFAULT_CHAT_MODEL,
        embed_model: str = DEFAULT_EMBED_MODEL,
        collection_name: str = COLLECTION_NAME,
        persist_dir: Path = CHROMA_DIR,
    ):
        self.ollama = OllamaClient(ollama_host)
        self.chat_model = chat_model
        self.embed_model = embed_model
        self.store = FinanceVectorStore(persist_dir, collection_name)

    def ingest_pdfs(
        self,
        pdf_paths: list[Path],
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        reset_collection: bool = True,
    ) -> dict[str, Any]:
        self.ollama.check_connection()
        if reset_collection:
            self.store.reset()

        all_pages = []
        page_counts: dict[str, int] = {}
        for pdf_path in pdf_paths:
            pages = extract_pdf_pages(pdf_path)
            all_pages.extend(pages)
            page_counts[pdf_path.name] = len(pages)

        chunks = chunk_pages(all_pages, chunk_size, chunk_overlap)
        embeddings = self.ollama.embed(
            [chunk["text"] for chunk in chunks],
            model=self.embed_model,
        )
        self.store.add_chunks(chunks, embeddings)

        return {
            "files_processed": len(pdf_paths),
            "pages_with_text": len(all_pages),
            "chunks_indexed": len(chunks),
            "store_count": self.store.count(),
            "page_counts": page_counts,
        }

    def answer(self, question: str, top_k: int = 4) -> dict[str, Any]:
        if self.store.count() == 0:
            raise ValueError("No documents have been indexed yet.")

        query_embedding = self.ollama.embed([question], model=self.embed_model)[0]
        retrieved = self.store.query(query_embedding, top_k=top_k)
        prompt = build_prompt(question, retrieved)
        answer = self.ollama.chat(prompt, model=self.chat_model, temperature=0.0)

        return {
            "answer": answer,
            "sources": format_sources(retrieved),
            "retrieved_chunks": retrieved,
        }

    def stats(self) -> dict[str, Any]:
        return {
            "collection_name": self.store.collection_name,
            "chunk_count": self.store.count(),
            "chat_model": self.chat_model,
            "embed_model": self.embed_model,
            "persist_dir": str(self.store.persist_dir),
        }


def build_prompt(question: str, retrieved_chunks: list[dict[str, Any]]) -> str:
    context_blocks = []
    for index, chunk in enumerate(retrieved_chunks, start=1):
        metadata = chunk["metadata"]
        source = (
            f"{metadata['file_name']}, page {metadata['page_number']}, "
            f"quarter {metadata.get('quarter', 'unknown')}"
        )
        context_blocks.append(f"[Source {index}: {source}]\n{chunk['text']}")

    context = "\n\n---\n\n".join(context_blocks)
    return f"""
You are a careful financial analyst answering questions from quarterly company reports.

Rules:
- Use only the provided context.
- If the answer is not present in the context, say: "The provided reports do not contain enough information to answer that."
- Include numbers with their unit and period when available.
- Do not use outside knowledge.
- Keep the answer concise and factual.

Context:
{context}

Question:
{question}

Answer:
""".strip()


def format_sources(retrieved_chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    sources = []
    for chunk in retrieved_chunks:
        metadata = chunk["metadata"]
        key = (metadata["file_name"], metadata["page_number"])
        if key in seen:
            continue
        seen.add(key)
        sources.append(
            {
                "file_name": metadata["file_name"],
                "page_number": metadata["page_number"],
                "quarter": metadata.get("quarter", "unknown"),
            }
        )
    return sources

