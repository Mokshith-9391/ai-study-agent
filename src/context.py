from src.evidence import Evidence


def build_context(
    evidence: list[Evidence],
) -> str:
    """
    Convert structured evidence into the context
    that will be provided to the LLM.
    """

    context_parts = []

    for index, item in enumerate(
        evidence,
        start=1,
    ):
        context_parts.append(
            f"""
EVIDENCE {index}
Source: {item.source}
Page: {item.page if item.page is not None else "unknown"}
Chunk ID: {item.chunk_id}
Retrieval distance: {item.distance:.4f}

Content:
{item.content}
""".strip()
        )

    return "\n\n".join(context_parts)