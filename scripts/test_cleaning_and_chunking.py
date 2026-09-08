from src.ingestion.loader import load_study_folder
from src.ingestion.cleaner import clean_documents
from src.ingestion.chunker import split_documents


def main() -> None:
    documents = load_study_folder("data/study")

    cleaned_documents = clean_documents(documents)

    chunks = split_documents(cleaned_documents)

    print()
    print("=" * 60)
    print("CLEANING + CHUNKING RESULT")
    print("=" * 60)

    print(f"Original documents: {len(documents)}")
    print(f"Cleaned documents:  {len(cleaned_documents)}")
    print(f"Final chunks:       {len(chunks)}")

    print()
    print("=" * 60)
    print("FIRST 10 CHUNKS")
    print("=" * 60)

    for index, chunk in enumerate(chunks[:10], start=1):
        print()
        print(f"--- Chunk {index} ---")
        print("Chunk ID:", chunk.metadata.get("chunk_id"))
        print("Source:", chunk.metadata.get("source"))
        print("Page:", chunk.metadata.get("page"))
        print("Characters:", len(chunk.page_content))
        print("Content:")
        print(chunk.page_content[:800])


if __name__ == "__main__":
    main()