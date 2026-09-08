from src.ingestion.loader import load_study_folder
from src.ingestion.cleaner import clean_documents
from src.ingestion.chunker import split_documents
from src.embedding import GeminiEmbedder
from src.vectorstore import ChromaVectorStore


def main() -> None:
    print()
    print("=" * 60)
    print("STUDY INDEXING")
    print("=" * 60)

    # 1. Load
    documents = load_study_folder("data/study")

    print(f"Documents loaded: {len(documents)}")

    # 2. Clean
    cleaned_documents = clean_documents(documents)

    print(
        f"Documents after cleaning: "
        f"{len(cleaned_documents)}"
    )

    # 3. Chunk
    chunks = split_documents(cleaned_documents)

    print(f"Chunks created: {len(chunks)}")

    # 4. Embed
    embedder = GeminiEmbedder()

    print()
    print("Generating embeddings...")

    texts = [
        chunk.page_content
        for chunk in chunks
    ]

    embeddings = embedder.embed_documents(texts)

    print(
        f"Embeddings generated: "
        f"{len(embeddings)}"
    )

    # 5. Store
    store = ChromaVectorStore()

    store.add_chunks(
        chunks,
        embeddings,
    )

    print()
    print("=" * 60)
    print("INDEXING COMPLETE")
    print("=" * 60)

    print(
        f"Chunks stored in Chroma: "
        f"{store.count()}"
    )


if __name__ == "__main__":
    main()