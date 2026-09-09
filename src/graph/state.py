from typing import Any, Literal, TypedDict


StudyMode = Literal[
    "answer",
    "explain",
    "summarize",
    "quiz",
    "flashcards",
    "compare",
]


class StudyState(TypedDict, total=False):
    # ========================================================
    # INPUT
    # ========================================================

    user_request: str
    mode: StudyMode

    topic_a: str
    topic_b: str

    number_of_questions: int
    number_of_cards: int

    # ========================================================
    # SESSION / MEMORY
    # ========================================================

    session_id: int | None

    previous_messages: list[Any]

    # ========================================================
    # RETRIEVAL
    # ========================================================

    evidence: list[Any]
    context: str

    # ========================================================
    # OUTPUT
    # ========================================================

    result: Any

    # ========================================================
    # ERROR
    # ========================================================

    error: str