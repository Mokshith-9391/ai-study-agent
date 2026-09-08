from src.ingestion.loader import load_study_folder


def main() -> None:
    documents = load_study_folder("data/study")

    print()
    print("=" * 60)
    print("INGESTION RESULT")
    print("=" * 60)

    print(f"Documents loaded: {len(documents)}")

    for index, document in enumerate(documents[:5], start=1):
        print()
        print(f"--- Document {index} ---")

        print("Source:")
        print(document.metadata.get("source"))

        print("\nMetadata:")
        print(document.metadata)

        print("\nContent preview:")
        print(document.page_content[:500])


if __name__ == "__main__":
    main()