from typing import Any


SEPARATORS = ["\n\n", "\n", ". ", "; ", ", ", " "]


def chunk_pages(
    pages: list[dict[str, Any]],
    chunk_size: int,
    chunk_overlap: int,
) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []

    for page in pages:
        source_label = (
            f"Source: {page['file_name']} | Quarter: {page['quarter']} | "
            f"Page: {page['page_number']}"
        )
        text = f"{source_label}\n\n{page['text']}"
        for chunk_index, chunk_text in enumerate(
            split_text(text, chunk_size, chunk_overlap)
        ):
            chunks.append(
                {
                    "id": stable_chunk_id(page, chunk_index),
                    "text": chunk_text,
                    "metadata": {
                        "file_name": page["file_name"],
                        "page_number": page["page_number"],
                        "quarter": page["quarter"],
                        "chunk_index": chunk_index,
                    },
                }
            )

    return chunks


def split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    text = text.strip()
    if len(text) <= chunk_size:
        return [text] if text else []

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        split_at = best_split_position(text, start, end)
        chunk = text[start:split_at].strip()
        if chunk:
            chunks.append(chunk)

        if split_at >= len(text):
            break
        next_start = max(0, split_at - chunk_overlap)
        start = split_at if next_start <= start else next_start

    return chunks


def best_split_position(text: str, start: int, hard_end: int) -> int:
    window = text[start:hard_end]
    min_position = int(len(window) * 0.55)

    for separator in SEPARATORS:
        position = window.rfind(separator)
        if position >= min_position:
            return start + position + len(separator)

    return hard_end


def stable_chunk_id(page: dict[str, Any], chunk_index: int) -> str:
    safe_file = page["file_name"].replace(" ", "_")
    return f"{safe_file}:p{page['page_number']}:c{chunk_index}"
