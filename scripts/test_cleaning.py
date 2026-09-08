from src.ingestion.loader import load_study_folder
from src.ingestion.cleaner import clean_documents


def main() -> None:
    documents = load_study_folder("data/study")
    cleaned_documents = clean_documents(documents)

    print()
    print("=" * 60)
    print("CLEANING RESULT")
    print("=" * 60)

    print(f"Original documents: {len(documents)}")
    print(f"Cleaned documents:  {len(cleaned_documents)}")

    for index, document in enumerate(cleaned_documents[:5], start=1):
        print()
        print(f"--- Document {index} ---")
        print("Source:", document.metadata.get("source"))
        print("Page:", document.metadata.get("page"))
        print("Characters:", len(document.page_content))
        print()
        print(document.page_content[:800])


if __name__ == "__main__":
    main()