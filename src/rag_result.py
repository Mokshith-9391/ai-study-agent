from dataclasses import dataclass

from src.evidence import Evidence


@dataclass
class RAGResult:
    """
    Structured result returned by the RAG pipeline.
    """

    question: str
    answer: str
    evidence: list[Evidence]

    @property
    def sources(self) -> list[str]:
        """
        Return unique human-readable citations.
        """

        citations = []

        for item in self.evidence:
            citation = item.citation()

            if citation not in citations:
                citations.append(citation)

        return citations