from collections import Counter

from langchain_core.documents import Document


def find_repeated_lines(
    documents: list[Document],
    min_occurrences: int = 3,
    max_line_length: int = 120,
) -> set[str]:
    """
    Find short lines repeated across multiple documents/pages.

    These are candidates for headers, footers, or other PDF boilerplate.
    """

    line_counts = Counter()

    for document in documents:
        # Count a line at most once per document/page.
        lines_seen_in_document = set()

        for line in document.page_content.splitlines():
            line = line.strip()

            if not line:
                continue

            if len(line) > max_line_length:
                continue

            lines_seen_in_document.add(line)

        for line in lines_seen_in_document:
            line_counts[line] += 1

    return {
        line
        for line, count in line_counts.items()
        if count >= min_occurrences
    }


def remove_repeated_lines(
    documents: list[Document],
    repeated_lines: set[str],
) -> list[Document]:
    """
    Remove lines identified as repeated boilerplate.

    Original metadata is preserved.
    """

    cleaned_documents = []

    for document in documents:
        lines = document.page_content.splitlines()

        filtered_lines = [
            line
            for line in lines
            if line.strip() not in repeated_lines
        ]

        cleaned_text = "\n".join(filtered_lines).strip()

        if not cleaned_text:
            continue

        cleaned_documents.append(
            Document(
                page_content=cleaned_text,
                metadata=dict(document.metadata),
            )
        )

    return cleaned_documents