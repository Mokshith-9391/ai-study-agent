from langgraph.graph import END, START, StateGraph

from src.graph.nodes import (
    build_context,
    generate_result,
    load_memory,
    retrieve_evidence,
    route_request,
    save_interaction,
    validate_result,
)
from src.graph.state import StudyState
from src.memory.repository import MemoryRepository
from src.study_agent import StudyAgent


def build_study_graph(
    agent: StudyAgent | None = None,
    memory_repository: MemoryRepository | None = None,
):
    """
    Build the study-agent LangGraph workflow.

    The MemoryRepository is injected into the graph so every
    memory operation uses the same repository/database as the
    application.
    """

    agent = agent or StudyAgent()

    if memory_repository is None:
        memory_repository = MemoryRepository()

    builder = StateGraph(
        StudyState
    )

    # ========================================================
    # MEMORY
    # ========================================================

    builder.add_node(
        "load_memory",
        lambda state: load_memory(
            state,
            agent,
            memory_repository,
        ),
    )

    builder.add_node(
        "save_interaction",
        lambda state: save_interaction(
            state,
            agent,
            memory_repository,
        ),
    )

    # ========================================================
    # CORE WORKFLOW
    # ========================================================

    builder.add_node(
        "route_request",
        route_request,
    )

    builder.add_node(
        "retrieve_evidence",
        lambda state: retrieve_evidence(
            state,
            agent,
        ),
    )

    builder.add_node(
        "build_context",
        lambda state: build_context(
            state,
            agent,
        ),
    )

    # ========================================================
    # GENERATION NODES
    # ========================================================

    builder.add_node(
        "generate_answer",
        lambda state: generate_result(
            {
                **state,
                "mode": "answer",
            },
            agent,
        ),
    )

    builder.add_node(
        "generate_explain",
        lambda state: generate_result(
            {
                **state,
                "mode": "explain",
            },
            agent,
        ),
    )

    builder.add_node(
        "generate_summarize",
        lambda state: generate_result(
            {
                **state,
                "mode": "summarize",
            },
            agent,
        ),
    )

    builder.add_node(
        "generate_quiz",
        lambda state: generate_result(
            {
                **state,
                "mode": "quiz",
            },
            agent,
        ),
    )

    builder.add_node(
        "generate_flashcards",
        lambda state: generate_result(
            {
                **state,
                "mode": "flashcards",
            },
            agent,
        ),
    )

    builder.add_node(
        "generate_compare",
        lambda state: generate_result(
            {
                **state,
                "mode": "compare",
            },
            agent,
        ),
    )

    # ========================================================
    # VALIDATION
    # ========================================================

    builder.add_node(
        "validate_result",
        validate_result,
    )

    # ========================================================
    # START → MEMORY → ROUTING
    # ========================================================

    builder.add_edge(
        START,
        "load_memory",
    )

    builder.add_edge(
        "load_memory",
        "route_request",
    )

    builder.add_edge(
        "route_request",
        "retrieve_evidence",
    )

    builder.add_edge(
        "retrieve_evidence",
        "build_context",
    )

    # ========================================================
    # MODE ROUTING
    # ========================================================

    builder.add_conditional_edges(
        "build_context",
        lambda state: state["mode"],
        {
            "answer": "generate_answer",
            "explain": "generate_explain",
            "summarize": "generate_summarize",
            "quiz": "generate_quiz",
            "flashcards": "generate_flashcards",
            "compare": "generate_compare",
        },
    )

    # ========================================================
    # GENERATION → VALIDATION
    # ========================================================

    builder.add_edge(
        "generate_answer",
        "validate_result",
    )

    builder.add_edge(
        "generate_explain",
        "validate_result",
    )

    builder.add_edge(
        "generate_summarize",
        "validate_result",
    )

    builder.add_edge(
        "generate_quiz",
        "validate_result",
    )

    builder.add_edge(
        "generate_flashcards",
        "validate_result",
    )

    builder.add_edge(
        "generate_compare",
        "validate_result",
    )

    # ========================================================
    # VALIDATION → MEMORY → END
    # ========================================================

    builder.add_edge(
        "validate_result",
        "save_interaction",
    )

    builder.add_edge(
        "save_interaction",
        END,
    )

    return builder.compile()