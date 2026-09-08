from src.evidence import Evidence
from src.citations import (
    CitationError,
    render_citations,
)


def main():
    evidence = [
        Evidence(
            chunk_id="abc123",
            content="RAG content",
            source="machine-learning/Rag.pdf",
            page="8",
            distance=0.45,
        ),
        Evidence(
            chunk_id="def456",
            content="More RAG content",
            source="machine-learning/Rag.pdf",
            page="46",
            distance=0.48,
        ),
    ]

    print("=" * 70)
    print("CITATION RENDERER TEST")
    print("=" * 70)

    valid_cases = [
        "RAG retrieves relevant chunks [E1].",
        "RAG retrieves relevant chunks [E1, E2].",
        "RAG retrieves relevant chunks [E1,E2].",
    ]

    for original in valid_cases:
        rendered = render_citations(
            answer=original,
            evidence=evidence,
        )

        print("\nOriginal:")
        print(original)

        print("\nRendered:")
        print(rendered)

        print("-" * 70)

    invalid_cases = [
        "RAG retrieves relevant chunks [E99].",
        "RAG retrieves relevant chunks [E1, E99].",
    ]

    for invalid_answer in invalid_cases:
        try:
            render_citations(
                answer=invalid_answer,
                evidence=evidence,
            )
        except CitationError as error:
            print("\nInvalid citation correctly rejected:")
            print(invalid_answer)
            print(f"Error: {error}")
        else:
            raise AssertionError(
                f"Invalid citation was not rejected: {invalid_answer}"
            )

    print("\n" + "=" * 70)
    print("ALL CITATION TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()