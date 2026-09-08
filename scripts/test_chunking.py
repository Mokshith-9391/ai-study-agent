from src.ingestion.loader import load_study_folder
from src.ingestion.chunker import split_documents


def main() -> None:
    documents = load_study_folder("data/study")

    chunks = split_documents(documents)

    print()
    print("=" * 60)
    print("CHUNKING RESULT")
    print("=" * 60)

    print(f"Documents: {len(documents)}")
    print(f"Chunks:    {len(chunks)}")

    print()
    print("=" * 60)
    print("FIRST 10 CHUNKS")
    print("=" * 60)

    for index, chunk in enumerate(chunks[:10], start=1):
        print()
        print(f"--- Chunk {index} ---")

        print("Chunk ID:")
        print(chunk.metadata.get("chunk_id"))

        print("Source:")
        print(chunk.metadata.get("source"))

        print("Page:")
        print(chunk.metadata.get("page"))

        print("Characters:")
        print(len(chunk.page_content))

        print("Content:")
        print(chunk.page_content[:800])


if __name__ == "__main__":
    main()