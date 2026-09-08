import re

from src.evidence import Evidence


EVIDENCE_GROUP_PATTERN = re.compile(
    r"\[((?:E\d+)(?:\s*,\s*E\d+)*)\]"
)


class CitationError(ValueError):
    """
    Raised when the LLM produces an invalid evidence reference.
    """


def validate_citations(
    answer: str,
    evidence: list[Evidence],
) -> None:
    """
    Validate all evidence references in an answer.

    Raises CitationError if the answer references
    evidence that does not exist.
    """

    matches = EVIDENCE_GROUP_PATTERN.findall(answer)

    for match in matches:
        references = match.split(",")

        for reference in references:
            reference = reference.strip()

            evidence_number = int(
                reference[1:]
            )

            if not (
                1 <= evidence_number <= len(evidence)
            ):
                raise CitationError(
                    f"Invalid evidence reference: "
                    f"[{reference}]. "
                    f"Only E1-E{len(evidence)} are valid."
                )


def render_citations(
    answer: str,
    evidence: list[Evidence],
) -> str:
    """
    Replace valid evidence references with
    application-controlled citations.

    Invalid evidence references cause an error
    instead of being silently discarded.
    """

    validate_citations(
        answer=answer,
        evidence=evidence,
    )

    def replace_group(
        match: re.Match,
    ) -> str:
        references = match.group(1).split(",")

        citations = []

        for reference in references:
            reference = reference.strip()

            evidence_number = int(
                reference[1:]
            )

            citation = evidence[
                evidence_number - 1
            ].citation()

            if citation not in citations:
                citations.append(citation)

        return "[" + "; ".join(citations) + "]"

    return EVIDENCE_GROUP_PATTERN.sub(
        replace_group,
        answer,
    )