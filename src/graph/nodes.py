from src.citations import render_citations
from src.memory.repository import MemoryRepository
from src.study_agent import StudyAgent
from src.study_models import Comparison, Flashcard, QuizQuestion
from src.validation import (
    require_valid,
    validate_comparison,
    validate_flashcard_item,
    validate_quiz_item,
)


# ============================================================
# SUPPORTED MODES
# ============================================================


SUPPORTED_MODES = {
    "answer",
    "explain",
    "summarize",
    "quiz",
    "flashcards",
    "compare",
}


# ============================================================
# MEMORY HELPERS
# ============================================================


def _get_session_id(
    state: dict,
) -> int | None:
    """
    Get the current study session ID.

    Persistent memory is optional. If there is no session ID,
    memory operations are skipped.
    """

    session_id = state.get(
        "session_id"
    )

    if session_id is None:
        return None

    if isinstance(
        session_id,
        bool,
    ):
        raise ValueError(
            "session_id must be an integer."
        )

    try:
        return int(session_id)

    except (TypeError, ValueError) as exc:
        raise ValueError(
            "session_id must be an integer."
        ) from exc


# ============================================================
# MEMORY: LOAD
# ============================================================


def load_memory(
    state: dict,
    agent: StudyAgent,
    memory_repository: MemoryRepository | None = None,
) -> dict:
    """
    Load persistent messages for the current study session.

    The repository is injected by the workflow so that the graph
    and application use the same database connection/configuration.
    """

    session_id = _get_session_id(
        state
    )

    if session_id is None:
        return {
            "previous_messages": []
        }

    if memory_repository is None:
        memory_repository = MemoryRepository()

    messages = memory_repository.get_messages(
        session_id
    )

    return {
        "previous_messages": messages
    }


# ============================================================
# MEMORY: SAVE
# ============================================================


def save_interaction(
    state: dict,
    agent: StudyAgent,
    memory_repository: MemoryRepository | None = None,
) -> dict:
    """
    Persist the completed interaction.

    The same MemoryRepository used by the application is injected
    into the graph.
    """

    session_id = _get_session_id(
        state
    )

    if session_id is None:
        return {}

    if memory_repository is None:
        memory_repository = MemoryRepository()

    # --------------------------------------------------------
    # Save user message
    # --------------------------------------------------------

    memory_repository.add_message(
        session_id,
        "user",
        state["user_request"],
    )

    # --------------------------------------------------------
    # Convert result to persistent text
    # --------------------------------------------------------

    result = state.get(
        "result"
    )

    if isinstance(
        result,
        str,
    ):
        assistant_content = result

    elif isinstance(
        result,
        QuizQuestion,
    ):
        assistant_content = (
            f"Question: {result.question}\n"
            f"Options: {', '.join(result.options)}\n"
            f"Answer: {result.answer}\n"
            f"Explanation: {result.explanation}"
        )

    elif isinstance(
        result,
        Flashcard,
    ):
        assistant_content = (
            f"Front: {result.front}\n"
            f"Back: {result.back}"
        )

    elif isinstance(
        result,
        Comparison,
    ):
        assistant_content = (
            f"{result.topic_a} vs {result.topic_b}\n"
            f"{result.comparison}"
        )

    elif isinstance(
        result,
        list,
    ):
        assistant_parts = []

        for item in result:

            if isinstance(
                item,
                QuizQuestion,
            ):
                assistant_parts.append(
                    f"Question: {item.question}\n"
                    f"Options: {', '.join(item.options)}\n"
                    f"Answer: {item.answer}\n"
                    f"Explanation: {item.explanation}"
                )

            elif isinstance(
                item,
                Flashcard,
            ):
                assistant_parts.append(
                    f"Front: {item.front}\n"
                    f"Back: {item.back}"
                )

            else:
                assistant_parts.append(
                    str(item)
                )

        assistant_content = "\n\n".join(
            assistant_parts
        )

    elif result is None:
        assistant_content = ""

    else:
        assistant_content = str(
            result
        )

    # --------------------------------------------------------
    # Save assistant message
    # --------------------------------------------------------

    if assistant_content.strip():

        memory_repository.add_message(
            session_id,
            "assistant",
            assistant_content,
        )

    return {}


# ============================================================
# REQUEST ROUTING
# ============================================================


def route_request(
    state: dict,
) -> dict:
    """
    Validate the requested study mode.
    """

    mode = state.get(
        "mode"
    )

    if mode not in SUPPORTED_MODES:
        raise ValueError(
            f"Unsupported study mode: {mode}"
        )

    return {
        "mode": mode
    }


# ============================================================
# RETRIEVAL
# ============================================================


def retrieve_evidence(
    state: dict,
    agent: StudyAgent,
) -> dict:
    """
    Retrieve relevant evidence from the vector store.
    """

    mode = state["mode"]

    request = state["user_request"]

    top_k = (
        8
        if mode in {
            "quiz",
            "flashcards",
            "compare",
        }
        else 5
    )

    evidence = agent.rag.retrieve(
        question=request,
        top_k=top_k,
    )

    return {
        "evidence": evidence
    }


# ============================================================
# CONTEXT
# ============================================================


def build_context(
    state: dict,
    agent: StudyAgent,
) -> dict:
    """
    Build the context supplied to the LLM.
    """

    context = agent.rag.build_context(
        state["evidence"]
    )

    return {
        "context": context
    }


# ============================================================
# ANSWER
# ============================================================


def _generate_answer(
    state: dict,
    agent: StudyAgent,
) -> str:

    prompt = f"""
You are a study assistant.

Answer the user's question using ONLY the supplied evidence.

USER QUESTION:
{state["user_request"]}

SUPPLIED EVIDENCE:
{state["context"]}

GROUNDING RULES:
1. Use only information supported by the evidence.
2. Do not invent facts.
3. Do not invent source names, page numbers, or chunk IDs.
4. When making a factual claim based on evidence, cite it using the evidence
   labels exactly as provided, such as [E1] or [E1, E3].
5. If the evidence does not contain enough information, say so clearly.
6. Prefer a clear educational explanation over unnecessary verbosity.

Return only the answer.
"""

    raw_answer = agent.rag.llm.generate(
        prompt
    )

    return render_citations(
        answer=raw_answer,
        evidence=state["evidence"],
    )


# ============================================================
# EXPLAIN
# ============================================================


def _generate_explanation(
    state: dict,
    agent: StudyAgent,
) -> str:

    prompt = f"""
You are a study assistant helping a student understand a topic.

Explain the topic clearly using ONLY the supplied evidence.

TOPIC:
{state["user_request"]}

SUPPLIED EVIDENCE:
{state["context"]}

RULES:
1. Use only information supported by the evidence.
2. Explain concepts step by step when useful.
3. Do not invent facts.
4. Cite factual claims using evidence labels such as [E1] or [E1, E2].
5. If the evidence is insufficient, say so clearly.

Return only the explanation.
"""

    raw_answer = agent.rag.llm.generate(
        prompt
    )

    return render_citations(
        answer=raw_answer,
        evidence=state["evidence"],
    )


# ============================================================
# SUMMARY
# ============================================================


def _generate_summary(
    state: dict,
    agent: StudyAgent,
) -> str:

    prompt = f"""
You are a study assistant.

Create a concise study summary of the requested topic using ONLY the supplied
evidence.

TOPIC:
{state["user_request"]}

SUPPLIED EVIDENCE:
{state["context"]}

RULES:
1. Use only information supported by the evidence.
2. Preserve important concepts and relationships.
3. Do not invent facts.
4. Cite factual claims using evidence labels such as [E1] or [E1, E2].
5. If the evidence is insufficient, say so clearly.

Return only the summary.
"""

    raw_answer = agent.rag.llm.generate(
        prompt
    )

    return render_citations(
        answer=raw_answer,
        evidence=state["evidence"],
    )


# ============================================================
# QUIZ
# ============================================================


def _generate_quiz(
    state: dict,
    agent: StudyAgent,
) -> list[QuizQuestion]:

    count = state.get(
        "number_of_questions",
        5,
    )

    if count <= 0:
        raise ValueError(
            "number_of_questions must be greater than zero."
        )

    prompt = f"""
You are a study assistant creating a multiple-choice quiz.

Create exactly {count} questions about:

{state["user_request"]}

Use ONLY the supplied evidence.

SUPPLIED EVIDENCE:
{state["context"]}

Return valid JSON only.

Format:
[
  {{
    "question": "Question text",
    "options": ["A", "B", "C", "D"],
    "answer": "The exact correct option text",
    "explanation": "Why the answer is correct",
    "evidence": ["E1"]
  }}
]

Rules:
1. Every question must be answerable from the evidence.
2. Do not invent facts.
3. Use exactly four options per question.
4. The answer must exactly match one option.
5. Evidence references must use only supplied labels.
"""

    raw_response = agent.rag.llm.generate(
        prompt
    )

    data = agent._parse_json(
        raw_response
    )

    if not isinstance(
        data,
        list,
    ):
        raise ValueError(
            "Quiz response must be a JSON list."
        )

    if len(data) != count:
        raise ValueError(
            f"Expected {count} quiz questions, "
            f"got {len(data)}."
        )

    questions = []

    for item in data:

        if not isinstance(
            item,
            dict,
        ):
            raise ValueError(
                "Each quiz item must be a JSON object."
            )

        require_valid(
            validate_quiz_item(item)
        )

        evidence = _resolve_evidence(
            item.get("evidence", []),
            state["evidence"],
        )

        question = QuizQuestion(
            question=item["question"],
            options=item["options"],
            answer=item["answer"],
            explanation=item["explanation"],
            evidence=evidence,
        )

        questions.append(
            question
        )

    return questions


# ============================================================
# FLASHCARDS
# ============================================================


def _generate_flashcards(
    state: dict,
    agent: StudyAgent,
) -> list[Flashcard]:

    count = state.get(
        "number_of_cards",
        5,
    )

    if count <= 0:
        raise ValueError(
            "number_of_cards must be greater than zero."
        )

    prompt = f"""
You are a study assistant creating flashcards.

Create exactly {count} useful flashcards about:

{state["user_request"]}

Use ONLY the supplied evidence.

SUPPLIED EVIDENCE:
{state["context"]}

Return valid JSON only.

Format:
[
  {{
    "front": "Question or concept",
    "back": "Answer",
    "evidence": ["E1"]
  }}
]

Rules:
1. Every flashcard must be answerable from the evidence.
2. Do not invent facts.
3. Evidence references must use only supplied labels.
"""

    raw_response = agent.rag.llm.generate(
        prompt
    )

    data = agent._parse_json(
        raw_response
    )

    if not isinstance(
        data,
        list,
    ):
        raise ValueError(
            "Flashcard response must be a JSON list."
        )

    if len(data) != count:
        raise ValueError(
            f"Expected {count} flashcards, "
            f"got {len(data)}."
        )

    cards = []

    for item in data:

        if not isinstance(
            item,
            dict,
        ):
            raise ValueError(
                "Each flashcard item must be a JSON object."
            )

        require_valid(
            validate_flashcard_item(item)
        )

        evidence = _resolve_evidence(
            item.get("evidence", []),
            state["evidence"],
        )

        card = Flashcard(
            front=item["front"],
            back=item["back"],
            evidence=evidence,
        )

        cards.append(
            card
        )

    return cards


# ============================================================
# COMPARISON
# ============================================================


def _generate_comparison(
    state: dict,
    agent: StudyAgent,
) -> Comparison:

    topic_a = state.get(
        "topic_a"
    )

    topic_b = state.get(
        "topic_b"
    )

    if not topic_a or not topic_b:
        raise ValueError(
            "Compare mode requires topic_a and topic_b."
        )

    prompt = f"""
You are a study assistant.

Compare these two topics using ONLY the supplied evidence.

TOPIC A:
{topic_a}

TOPIC B:
{topic_b}

SUPPLIED EVIDENCE:
{state["context"]}

Return valid JSON only.

Format:
{{
  "topic_a": "{topic_a}",
  "topic_b": "{topic_b}",
  "comparison": "Detailed comparison",
  "evidence": ["E1", "E2"]
}}

Rules:
1. Do not invent facts.
2. Only make claims supported by the evidence.
3. Evidence references must use only supplied labels.
"""

    raw_response = agent.rag.llm.generate(
        prompt
    )

    data = agent._parse_json(
        raw_response
    )

    if not isinstance(
        data,
        dict,
    ):
        raise ValueError(
            "Comparison response must be a JSON object."
        )

    require_valid(
        validate_comparison(data)
    )

    evidence = _resolve_evidence(
        data.get("evidence", []),
        state["evidence"],
    )

    return Comparison(
        topic_a=data["topic_a"],
        topic_b=data["topic_b"],
        comparison=data["comparison"],
        evidence=evidence,
    )


# ============================================================
# EVIDENCE RESOLUTION
# ============================================================


def _resolve_evidence(
    references,
    evidence,
):
    """
    Convert E1/E2/... references into actual Evidence objects.
    """

    if not isinstance(
        references,
        list,
    ):
        raise ValueError(
            "Evidence references must be a JSON list."
        )

    resolved = []

    for reference in references:

        if not isinstance(
            reference,
            str,
        ):
            raise ValueError(
                "Evidence reference must be a string."
            )

        reference = reference.strip().upper()

        if not reference.startswith("E"):
            raise ValueError(
                f"Invalid evidence reference: {reference}"
            )

        try:
            number = int(
                reference[1:]
            )

        except ValueError as exc:
            raise ValueError(
                f"Invalid evidence reference: {reference}"
            ) from exc

        if not 1 <= number <= len(evidence):
            raise ValueError(
                f"Evidence reference {reference} "
                f"is out of range."
            )

        resolved.append(
            evidence[number - 1]
        )

    return resolved


# ============================================================
# RESULT GENERATION
# ============================================================


def generate_result(
    state: dict,
    agent: StudyAgent,
) -> dict:

    mode = state["mode"]

    if mode == "answer":

        result = _generate_answer(
            state,
            agent,
        )

    elif mode == "explain":

        result = _generate_explanation(
            state,
            agent,
        )

    elif mode == "summarize":

        result = _generate_summary(
            state,
            agent,
        )

    elif mode == "quiz":

        result = _generate_quiz(
            state,
            agent,
        )

    elif mode == "flashcards":

        result = _generate_flashcards(
            state,
            agent,
        )

    elif mode == "compare":

        result = _generate_comparison(
            state,
            agent,
        )

    else:
        raise ValueError(
            f"Unsupported study mode: {mode}"
        )

    return {
        "result": result
    }


# ============================================================
# RESULT VALIDATION
# ============================================================


def validate_result(
    state: dict,
) -> dict:

    mode = state["mode"]

    result = state["result"]

    # --------------------------------------------------------
    # Text modes
    # --------------------------------------------------------

    if mode in {
        "answer",
        "explain",
        "summarize",
    }:

        if not isinstance(
            result,
            str,
        ):
            raise ValueError(
                f"{mode} result must be a string."
            )

        if not result.strip():
            raise ValueError(
                f"{mode} result must not be empty."
            )

        return {
            "result": result
        }

    # --------------------------------------------------------
    # Quiz
    # --------------------------------------------------------

    if mode == "quiz":

        if not isinstance(
            result,
            list,
        ):
            raise ValueError(
                "Quiz result must be a list."
            )

        if not all(
            isinstance(
                item,
                QuizQuestion,
            )
            for item in result
        ):
            raise ValueError(
                "Quiz result contains an invalid item."
            )

        return {
            "result": result
        }

    # --------------------------------------------------------
    # Flashcards
    # --------------------------------------------------------

    if mode == "flashcards":

        if not isinstance(
            result,
            list,
        ):
            raise ValueError(
                "Flashcard result must be a list."
            )

        if not all(
            isinstance(
                item,
                Flashcard,
            )
            for item in result
        ):
            raise ValueError(
                "Flashcard result contains an invalid item."
            )

        return {
            "result": result
        }

    # --------------------------------------------------------
    # Comparison
    # --------------------------------------------------------

    if mode == "compare":

        if not isinstance(
            result,
            Comparison,
        ):
            raise ValueError(
                "Comparison result has an invalid type."
            )

        return {
            "result": result
        }

    raise ValueError(
        f"Unsupported study mode: {mode}"
    )