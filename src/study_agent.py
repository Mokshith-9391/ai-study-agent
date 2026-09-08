import json
import re

from src.citations import render_citations
from src.rag import RAGPipeline
from src.rag_result import RAGResult
from src.study_models import (
    Comparison,
    Flashcard,
    QuizQuestion,
)
from src.memory.models import (
    Message,
    StudySession,
    TopicProgress,
)
from src.memory.repository import MemoryRepository
from src.validation import (
    require_valid,
    validate_comparison,
    validate_flashcard_item,
    validate_quiz_item,
)


class StudyAgent:
    """
    High-level study assistant.

    Study modes:
        answer
        explain
        summarize
        quiz
        flashcards
        compare

    Persistent learning state is handled separately through
    MemoryRepository.

    Architecture:

        StudyAgent
            ↓
        RAGPipeline
            ↓
        Retrieval / Evidence
            ↓
        Mode-specific generation
            ↓
        Validation
            ↓
        Structured result
    """

    def __init__(
        self,
        rag: RAGPipeline | None = None,
        memory: MemoryRepository | None = None,
    ):
        self.rag = rag or RAGPipeline()
        self.memory = memory or MemoryRepository()
        self.current_session: StudySession | None = None

    # ============================================================
    # SESSION / MEMORY
    # ============================================================

    def start_session(self) -> StudySession:
        """
        Start a persistent study session.
        """
        self.current_session = self.memory.start_session()
        return self.current_session

    def end_session(self) -> None:
        """
        End the current study session.
        """
        if self.current_session is None:
            return

        self.memory.end_session(
            self.current_session.id
        )

        self.current_session = None

    def record_user_message(
        self,
        content: str,
    ) -> Message:
        """
        Store a user message in the current session.
        """
        session = self._require_session()

        return self.memory.add_message(
            session_id=session.id,
            role="user",
            content=content,
        )

    def record_assistant_message(
        self,
        content: str,
    ) -> Message:
        """
        Store an assistant response in the current session.
        """
        session = self._require_session()

        return self.memory.add_message(
            session_id=session.id,
            role="assistant",
            content=content,
        )

    def record_quiz_attempt(
        self,
        topic: str,
        question: str,
        selected_answer: str,
        correct_answer: str,
    ):
        """
        Record a quiz attempt and update topic progress.
        """
        session_id = (
            self.current_session.id
            if self.current_session
            else None
        )

        return self.memory.record_quiz_attempt(
            session_id=session_id,
            topic=topic,
            question=question,
            selected_answer=selected_answer,
            correct_answer=correct_answer,
        )

    def record_flashcard_review(
        self,
        topic: str,
        front: str,
        remembered: bool,
    ):
        """
        Record a flashcard review and update topic progress.
        """
        session_id = (
            self.current_session.id
            if self.current_session
            else None
        )

        return self.memory.record_flashcard_review(
            session_id=session_id,
            topic=topic,
            front=front,
            remembered=remembered,
        )

    def get_topic_progress(
        self,
        topic: str,
    ) -> TopicProgress | None:
        """
        Return learning progress for one topic.
        """
        return self.memory.get_topic_progress(topic)

    def get_all_progress(
        self,
    ) -> list[TopicProgress]:
        """
        Return learning progress for all tracked topics.
        """
        return self.memory.get_all_topic_progress()

    def _require_session(self) -> StudySession:
        """
        Return the current session or raise an error.
        """
        if self.current_session is None:
            raise RuntimeError(
                "No active study session. "
                "Call start_session() first."
            )

        return self.current_session

    # ============================================================
    # ANSWER
    # ============================================================

    def answer(
        self,
        question: str,
    ) -> RAGResult:
        """
        Answer a study question using RAG.
        """
        return self.rag.ask(
            question=question,
            top_k=5,
        )

    # ============================================================
    # EXPLAIN
    # ============================================================

    def explain(
        self,
        topic: str,
    ) -> RAGResult:
        """
        Explain a topic using retrieved evidence.
        """

        evidence = self.rag.retrieve(
            question=f"Explain {topic} clearly for a student.",
            top_k=5,
        )

        context = self.rag.build_context(
            evidence
        )

        prompt = f"""
You are a study assistant.

Explain the following topic using ONLY the supplied evidence.

TOPIC:
{topic}

EVIDENCE:
{context}

REQUIREMENTS:
- Explain the concept clearly.
- Start with a simple definition.
- Then explain how it works.
- Include important details supported by the evidence.
- Use examples only when supported by the evidence.
- Do not invent facts.
- Cite factual claims using [E1], [E2], etc.
- If the evidence is insufficient, say so.

Return only the explanation.
"""

        raw_answer = self.rag.llm.generate(
            prompt
        )

        answer = render_citations(
            answer=raw_answer,
            evidence=evidence,
        )

        return RAGResult(
            question=topic,
            answer=answer,
            evidence=evidence,
        )

    # ============================================================
    # SUMMARIZE
    # ============================================================

    def summarize(
        self,
        topic: str,
    ) -> RAGResult:
        """
        Summarize a topic using retrieved evidence.
        """

        evidence = self.rag.retrieve(
            question=f"Summarize {topic}.",
            top_k=5,
        )

        context = self.rag.build_context(
            evidence
        )

        prompt = f"""
You are a study assistant.

Create a concise study summary of:

{topic}

Use ONLY the supplied evidence.

EVIDENCE:
{context}

REQUIREMENTS:
- Capture the key concepts.
- Preserve important terminology.
- Prefer bullet points when useful.
- Do not introduce unsupported information.
- Cite factual claims using [E1], [E2], etc.
- If the evidence is insufficient, say so.

Return only the summary.
"""

        raw_answer = self.rag.llm.generate(
            prompt
        )

        answer = render_citations(
            answer=raw_answer,
            evidence=evidence,
        )

        return RAGResult(
            question=topic,
            answer=answer,
            evidence=evidence,
        )

    # ============================================================
    # QUIZ
    # ============================================================

    def quiz(
        self,
        topic: str,
        number_of_questions: int = 5,
    ) -> list[QuizQuestion]:
        """
        Generate validated multiple-choice questions.
        """

        if number_of_questions <= 0:
            raise ValueError(
                "number_of_questions must be greater than zero."
            )

        if number_of_questions > 20:
            raise ValueError(
                "number_of_questions must not exceed 20."
            )

        evidence = self.rag.retrieve(
            question=(
                f"Important concepts and facts about {topic}"
            ),
            top_k=8,
        )

        context = self.rag.build_context(
            evidence
        )

        prompt = f"""
You are a study assistant creating a quiz.

TOPIC:
{topic}

EVIDENCE:
{context}

Create exactly {number_of_questions} multiple-choice questions.

Return ONLY valid JSON.

Required format:

{{
  "questions": [
    {{
      "question": "Question text",
      "options": [
        "Option A",
        "Option B",
        "Option C",
        "Option D"
      ],
      "answer": "Exactly one option",
      "explanation": "Short explanation",
      "evidence": ["E1"]
    }}
  ]
}}

RULES:
- Use ONLY information supported by the evidence.
- Each question must have exactly four options.
- There must be exactly one correct answer.
- "answer" must exactly match one option.
- "evidence" must contain only valid evidence labels.
- Do not invent evidence labels.
- Do not use markdown.
"""

        raw = self.rag.llm.generate(
            prompt
        )

        data = self._parse_json(raw)

        questions_data = data.get(
            "questions"
        )

        if not isinstance(
            questions_data,
            list,
        ):
            raise ValueError(
                "Quiz response does not contain "
                "a valid questions list."
            )

        if len(questions_data) != number_of_questions:
            raise ValueError(
                f"Expected {number_of_questions} questions, "
                f"got {len(questions_data)}."
            )

        questions = []

        for item in questions_data:

            require_valid(
                validate_quiz_item(item)
            )

            question = item["question"]
            options = item["options"]
            answer = item["answer"]
            explanation = item["explanation"]
            labels = item["evidence"]

            resolved_evidence = self._resolve_evidence(
                labels,
                evidence,
            )

            questions.append(
                QuizQuestion(
                    question=question,
                    options=options,
                    answer=answer,
                    explanation=explanation,
                    evidence=resolved_evidence,
                )
            )

        return questions

    # ============================================================
    # FLASHCARDS
    # ============================================================

    def flashcards(
        self,
        topic: str,
        number_of_cards: int = 5,
    ) -> list[Flashcard]:
        """
        Generate validated flashcards.
        """

        if number_of_cards <= 0:
            raise ValueError(
                "number_of_cards must be greater than zero."
            )

        if number_of_cards > 30:
            raise ValueError(
                "number_of_cards must not exceed 30."
            )

        evidence = self.rag.retrieve(
            question=(
                f"Important concepts for flashcards "
                f"about {topic}"
            ),
            top_k=8,
        )

        context = self.rag.build_context(
            evidence
        )

        prompt = f"""
You are a study assistant creating flashcards.

TOPIC:
{topic}

EVIDENCE:
{context}

Create exactly {number_of_cards} useful study flashcards.

Return ONLY valid JSON:

{{
  "flashcards": [
    {{
      "front": "Question or concept",
      "back": "Answer",
      "evidence": ["E1"]
    }}
  ]
}}

RULES:
- Use ONLY information supported by the evidence.
- Keep the front concise.
- Make the back educational but concise.
- Evidence labels must be valid.
- Do not invent evidence labels.
- Do not use markdown.
"""

        raw = self.rag.llm.generate(
            prompt
        )

        data = self._parse_json(raw)

        cards_data = data.get(
            "flashcards"
        )

        if not isinstance(
            cards_data,
            list,
        ):
            raise ValueError(
                "Flashcard response does not contain "
                "a valid flashcards list."
            )

        if len(cards_data) != number_of_cards:
            raise ValueError(
                f"Expected {number_of_cards} flashcards, "
                f"got {len(cards_data)}."
            )

        cards = []

        for item in cards_data:

            require_valid(
                validate_flashcard_item(item)
            )

            labels = item["evidence"]

            resolved_evidence = self._resolve_evidence(
                labels,
                evidence,
            )

            cards.append(
                Flashcard(
                    front=item["front"],
                    back=item["back"],
                    evidence=resolved_evidence,
                )
            )

        return cards

    # ============================================================
    # COMPARE
    # ============================================================

    def compare(
        self,
        topic_a: str,
        topic_b: str,
    ) -> Comparison:
        """
        Compare two topics using retrieved evidence.
        """

        evidence = self.rag.retrieve(
            question=f"Compare {topic_a} and {topic_b}",
            top_k=8,
        )

        context = self.rag.build_context(
            evidence
        )

        prompt = f"""
You are a study assistant.

Compare these two topics:

TOPIC A:
{topic_a}

TOPIC B:
{topic_b}

EVIDENCE:
{context}

Return ONLY valid JSON:

{{
  "topic_a": "{topic_a}",
  "topic_b": "{topic_b}",
  "comparison": "Detailed comparison using evidence citations such as [E1].",
  "evidence": ["E1", "E2"]
}}

RULES:
- Use ONLY information supported by the evidence.
- Explain similarities and differences.
- Do not invent facts.
- Use [E1], [E2], etc. for citations inside the comparison.
- Evidence labels must be valid.
- Do not use markdown outside the JSON string.
"""

        raw = self.rag.llm.generate(
            prompt
        )

        data = self._parse_json(raw)

        require_valid(
            validate_comparison(data)
        )

        labels = data["evidence"]

        resolved_evidence = self._resolve_evidence(
            labels,
            evidence,
        )

        comparison_text = render_citations(
            answer=data["comparison"],
            evidence=evidence,
        )

        return Comparison(
            topic_a=data["topic_a"],
            topic_b=data["topic_b"],
            comparison=comparison_text,
            evidence=resolved_evidence,
        )

    # ============================================================
    # HELPERS
    # ============================================================

    @staticmethod
    def _parse_json(
        raw: str,
    ) -> dict:
        """
        Parse JSON returned by the model.

        Handles accidental markdown code fences.
        """

        raw = raw.strip()

        if raw.startswith("```"):
            raw = re.sub(
                r"^```(?:json)?\s*",
                "",
                raw,
            )

            raw = re.sub(
                r"\s*```$",
                "",
                raw,
            )

        try:
            data = json.loads(raw)

        except json.JSONDecodeError as error:
            raise ValueError(
                "LLM returned invalid JSON."
            ) from error

        if not isinstance(data, dict):
            raise ValueError(
                "LLM JSON response must be an object."
            )

        return data

    @staticmethod
    def _resolve_evidence(
        labels: list[str],
        evidence,
    ):
        """
        Convert E1/E2/... labels into actual Evidence objects.
        """

        if not isinstance(
            labels,
            list,
        ):
            raise ValueError(
                "Evidence must be a list."
            )

        resolved = []

        for label in labels:

            if not isinstance(
                label,
                str,
            ):
                raise ValueError(
                    "Evidence labels must be strings."
                )

            match = re.fullmatch(
                r"E(\d+)",
                label.strip(),
            )

            if not match:
                raise ValueError(
                    f"Invalid evidence label: {label}"
                )

            index = int(
                match.group(1)
            )

            if not (
                1 <= index <= len(evidence)
            ):
                raise ValueError(
                    f"Invalid evidence label: {label}. "
                    f"Only E1-E{len(evidence)} are valid."
                )

            item = evidence[index - 1]

            if item not in resolved:
                resolved.append(item)

        return resolved