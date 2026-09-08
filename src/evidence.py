from dataclasses import dataclass


@dataclass
class Evidence:
    """
    A retrieved chunk together with its source metadata.
    """

    chunk_id: str
    content: str
    source: str
    page: str | None
    distance: float

    def citation(self) -> str:
        """
        Return a human-readable citation.
        """

        if self.page is not None:
            return f"{self.source}, page {self.page}"

        return self.source