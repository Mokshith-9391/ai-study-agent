from src.ingestion.loader import load_study_folder
from src.ingestion.cleaner import clean_documents
from src.ingestion.boilerplate import (
    find_repeated_lines,
    remove_repeated_lines,
)
from src.ingestion.chunker import split_documents


def main() -> None:
    documents = load_study_folder("data/study")

    cleaned_documents = clean_documents(documents)

    repeated_lines = find_repeated_lines(
        cleaned_documents,
        min_occurrences=3,
    )

    print()
    print("=" * 60)
    print("REPEATED LINE ANALYSIS")
    print("=" * 60)

    print(f"Repeated candidate lines: {len(repeated_lines)}")

    for line in sorted(repeated_lines):
        print(repr(line))

    boilerplate_cleaned = remove_repeated_lines(
        cleaned_documents,
        repeated_lines,
    )

    chunks_before = split_documents(cleaned_documents)
    chunks_after = split_documents(boilerplate_cleaned)

    print()
    print("=" * 60)
    print("CHUNK COMPARISON")
    print("=" * 60)

    print(f"Documents before cleanup: {len(cleaned_documents)}")
    print(f"Documents after cleanup:  {len(boilerplate_cleaned)}")
    print()
    print(f"Chunks before cleanup:    {len(chunks_before)}")
    print(f"Chunks after cleanup:     {len(chunks_after)}")


if __name__ == "__main__":
    main()