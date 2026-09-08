from src.vectorstore import ChromaVectorStore


def main() -> None:
    store = ChromaVectorStore()

    print()
    print("=" * 60)
    print("CHROMA TEST")
    print("=" * 60)

    print(f"Database path: {store.path}")
    print(f"Collection: study_documents")
    print(f"Stored chunks: {store.count()}")

    print()
    print("Chroma is working.")


if __name__ == "__main__":
    main()