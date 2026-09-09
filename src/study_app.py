from typing import Any

from src.graph.workflow import build_study_graph
from src.memory.repository import MemoryRepository
from src.rag_result import RAGResult
from src.study_agent import StudyAgent


class StudyApp:
    """
    Application-facing layer.

    Owns:
    - the active study session
    - the single MemoryRepository used by the application
    - the LangGraph workflow

    StudyAgent and LangGraph share the same MemoryRepository instance.
    """

    def __init__(
        self,
        agent: StudyAgent | None = None,
        memory_repository: MemoryRepository | None = None,
    ) -> None:
        # --------------------------------------------------------
        # Create exactly one repository for the application.
        # --------------------------------------------------------

        self.memory_repository = (
            memory_repository or MemoryRepository()
        )

        # --------------------------------------------------------
        # Ensure StudyAgent uses the exact same repository.
        # --------------------------------------------------------

        if agent is None:
            self.agent = StudyAgent(
                memory=self.memory_repository
            )
        else:
            self.agent = agent
            self.agent.memory = self.memory_repository

        # Application-owned active session.
        self._session_id: int | None = None

        # --------------------------------------------------------
        # LangGraph receives the exact same repository.
        # --------------------------------------------------------

        self.graph = build_study_graph(
            agent=self.agent,
            memory_repository=self.memory_repository,
        )

    @property
    def session_id(self) -> int | None:
        """Return the currently active study session ID."""
        return self._session_id

    def run(
        self,
        mode: str,
        user_request: str,
        session_id: int | None = None,
        **kwargs: Any,
    ) -> dict:
        if not isinstance(user_request, str):
            raise TypeError(
                "user_request must be a string."
            )

        user_request = user_request.strip()

        if not user_request:
            raise ValueError(
                "Study request must not be empty."
            )

        state = {
            "mode": mode,
            "user_request": user_request,
            "session_id": session_id if session_id is not None else self._session_id,
            **kwargs,
        }

        return self.graph.invoke(state)

    def _text_result(
        self,
        state: dict,
    ) -> RAGResult:
        result = state.get("result")
        evidence = state.get("evidence", [])

        if not isinstance(result, str):
            raise TypeError(
                "Expected text result from LangGraph."
            )

        return RAGResult(
            question=state["user_request"],
            answer=result,
            evidence=evidence,
        )

    def ask(
        self,
        question: str,
        session_id: int | None = None,
    ) -> RAGResult:
        state = self.run(
            mode="answer",
            user_request=question,
            session_id=session_id,
        )

        return self._text_result(state)

    def explain(
        self,
        topic: str,
        session_id: int | None = None,
    ) -> RAGResult:
        state = self.run(
            mode="explain",
            user_request=topic,
            session_id=session_id,
        )

        return self._text_result(state)

    def summarize(
        self,
        topic: str,
        session_id: int | None = None,
    ) -> RAGResult:
        state = self.run(
            mode="summarize",
            user_request=topic,
            session_id=session_id,
        )

        return self._text_result(state)

    def quiz(
        self,
        topic: str,
        number_of_questions: int = 5,
    ):
        if number_of_questions <= 0:
            raise ValueError(
                "number_of_questions must be greater than zero."
            )

        state = self.run(
            mode="quiz",
            user_request=topic,
            number_of_questions=number_of_questions,
        )

        return state["result"]

    def flashcards(
        self,
        topic: str,
        number_of_cards: int = 5,
    ):
        if number_of_cards <= 0:
            raise ValueError(
                "number_of_cards must be greater than zero."
            )

        state = self.run(
            mode="flashcards",
            user_request=topic,
            number_of_cards=number_of_cards,
        )

        return state["result"]

    def compare(
        self,
        topic_a: str,
        topic_b: str,
    ):
        if not isinstance(topic_a, str):
            raise TypeError(
                "topic_a must be a string."
            )

        if not isinstance(topic_b, str):
            raise TypeError(
                "topic_b must be a string."
            )

        topic_a = topic_a.strip()
        topic_b = topic_b.strip()

        if not topic_a:
            raise ValueError(
                "topic_a must not be empty."
            )

        if not topic_b:
            raise ValueError(
                "topic_b must not be empty."
            )

        state = self.run(
            mode="compare",
            user_request=(
                f"Compare {topic_a} and {topic_b}."
            ),
            topic_a=topic_a,
            topic_b=topic_b,
        )

        return state["result"]

    # ============================================================
    # SESSION
    # ============================================================

    def start_session(self):
        """
        Start one durable SQLite-backed study session.

        StudyAgent and LangGraph use the same repository, so the
        session belongs to the same persistence layer used by the
        rest of the application.
        """

        if self._session_id is not None:
            raise RuntimeError(
                f"A study session is already active: "
                f"{self._session_id}"
            )

        session = self.agent.start_session()

        if session is None or not hasattr(
            session,
            "id",
        ):
            raise TypeError(
                "StudyAgent.start_session() must return "
                "a session object containing an id."
            )

        self._session_id = session.id

        return session

    def create_session(self):
        """Create a durable session without replacing the UI's active session."""
        return self.memory_repository.start_session()

    def end_session(self):
        """
        End the active durable session and clear application state.
        """

        if self._session_id is None:
            return

        self.agent.end_session()

        self._session_id = None

    # ============================================================
    # PROGRESS
    # ============================================================

    def progress(self):
        """Return learning progress for all tracked topics."""
        return self.agent.get_all_progress()
