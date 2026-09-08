import hashlib

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 120


def create_chunk_id(document: Document, chunk_position: int) -> str:
    """
    Create a deterministic ID for a chunk.

    The ID is based on the source document, page information,
    chunk position, and chunk content.
    """

    source = document.metadata.get("source", "")
    page = document.metadata.get("page", "")
    content = document.page_content

    raw_id = f"{source}|{page}|{chunk_position}|{content}"

    return hashlib.sha256(
        raw_id.encode("utf-8")
    ).hexdigest()[:16]


def split_documents(
    documents: list[Document],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Document]:
    """
    Split documents into smaller chunks while preserving metadata.
    """

    if chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be smaller than chunk_size."
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            "",
        ],
    )

    chunks = splitter.split_documents(documents)

    position_by_document = {}

    for chunk in chunks:
        source = chunk.metadata.get("source", "")
        page = chunk.metadata.get("page", "")

        document_key = f"{source}|{page}"

        position = position_by_document.get(document_key, 0)

        chunk.metadata["chunk_position"] = position
        chunk.metadata["chunk_id"] = create_chunk_id(
            chunk,
            position,
        )

        position_by_document[document_key] = position + 1

    return chunks