import streamlit as st

from src.study_app import StudyApp


st.set_page_config(
    page_title="AI Study Agent",
    page_icon="📚",
    layout="wide",
)


# ---------------------------------------------------------
# Persistent application instance
# ---------------------------------------------------------

if "study_app" not in st.session_state:
    st.session_state.study_app = StudyApp()

study_app: StudyApp = st.session_state.study_app


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def display_sources(result) -> None:
    evidence = getattr(result, "evidence", [])

    if not evidence:
        return

    with st.expander("Sources"):
        for index, item in enumerate(evidence, start=1):
            citation = item.citation()

            st.markdown(f"**E{index} — {citation}**")
            st.write(item.content)


def display_text_result(result) -> None:
    st.markdown(result.answer)
    display_sources(result)


# ---------------------------------------------------------
# Header
# ---------------------------------------------------------

st.title("📚 AI Study Agent")
st.caption(
    "Source-grounded study assistant powered by RAG, LangGraph and SQLite."
)


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------

with st.sidebar:
    st.header("Study Session")

    if study_app.session_id is None:
        st.info("No active study session.")

        if st.button(
            "Start Session",
            use_container_width=True,
        ):
            try:
                study_app.start_session()
                st.rerun()
            except Exception as exc:
                st.error(str(exc))

    else:
        st.success(
            f"Active session: {study_app.session_id}"
        )

        if st.button(
            "End Session",
            use_container_width=True,
        ):
            try:
                study_app.end_session()
                st.rerun()
            except Exception as exc:
                st.error(str(exc))

    st.divider()

    mode = st.radio(
        "Study Mode",
        [
            "Ask",
            "Explain",
            "Summarize",
            "Quiz",
            "Flashcards",
            "Compare",
            "Progress",
        ],
    )


# ---------------------------------------------------------
# Require active session for interactive study modes
# ---------------------------------------------------------

if mode != "Progress" and study_app.session_id is None:
    st.info(
        "Start a study session from the sidebar before using "
        "the study modes."
    )
    st.stop()


# ---------------------------------------------------------
# ASK
# ---------------------------------------------------------

if mode == "Ask":
    st.header("Ask")

    question = st.text_area(
        "What would you like to know?",
        placeholder="What is Retrieval-Augmented Generation?",
        height=120,
    )

    if st.button("Ask", type="primary"):
        try:
            with st.spinner("Searching your study material..."):
                result = study_app.ask(question)

            display_text_result(result)

        except Exception as exc:
            st.error(str(exc))


# ---------------------------------------------------------
# EXPLAIN
# ---------------------------------------------------------

elif mode == "Explain":
    st.header("Explain")

    topic = st.text_input(
        "Topic",
        placeholder="Explain embeddings",
    )

    if st.button("Explain", type="primary"):
        try:
            with st.spinner("Generating explanation..."):
                result = study_app.explain(topic)

            display_text_result(result)

        except Exception as exc:
            st.error(str(exc))


# ---------------------------------------------------------
# SUMMARIZE
# ---------------------------------------------------------

elif mode == "Summarize":
    st.header("Summarize")

    topic = st.text_input(
        "Topic",
        placeholder="Summarize Retrieval-Augmented Generation",
    )

    if st.button("Summarize", type="primary"):
        try:
            with st.spinner("Generating summary..."):
                result = study_app.summarize(topic)

            display_text_result(result)

        except Exception as exc:
            st.error(str(exc))


# ---------------------------------------------------------
# QUIZ
# ---------------------------------------------------------

elif mode == "Quiz":
    st.header("Quiz")

    topic = st.text_input(
        "Topic",
        placeholder="Machine Learning",
    )

    number_of_questions = st.number_input(
        "Number of questions",
        min_value=1,
        max_value=20,
        value=5,
        step=1,
    )

    if st.button("Generate Quiz", type="primary"):
        try:
            with st.spinner("Generating quiz..."):
                questions = study_app.quiz(
                    topic,
                    int(number_of_questions),
                )

            st.session_state.quiz_questions = questions

        except Exception as exc:
            st.error(str(exc))

    questions = st.session_state.get(
        "quiz_questions",
        [],
    )

    if questions:
        for index, question in enumerate(
            questions,
            start=1,
        ):
            st.subheader(
                f"Question {index}"
            )

            selected = st.radio(
                question.question,
                question.options,
                key=f"quiz_answer_{index}",
            )

            if st.button(
                f"Check Answer {index}",
                key=f"quiz_check_{index}",
            ):
                if selected == question.answer:
                    st.success("Correct!")

                else:
                    st.error(
                        f"Incorrect. Correct answer: "
                        f"{question.answer}"
                    )

                st.info(question.explanation)

        if questions:
            display_sources(questions[0])


# ---------------------------------------------------------
# FLASHCARDS
# ---------------------------------------------------------

elif mode == "Flashcards":
    st.header("Flashcards")

    topic = st.text_input(
        "Topic",
        placeholder="Embeddings",
    )

    number_of_cards = st.number_input(
        "Number of cards",
        min_value=1,
        max_value=20,
        value=5,
        step=1,
    )

    if st.button(
        "Generate Flashcards",
        type="primary",
    ):
        try:
            with st.spinner("Generating flashcards..."):
                cards = study_app.flashcards(
                    topic,
                    int(number_of_cards),
                )

            st.session_state.flashcards = cards

        except Exception as exc:
            st.error(str(exc))

    cards = st.session_state.get(
        "flashcards",
        [],
    )

    for index, card in enumerate(
        cards,
        start=1,
    ):
        st.subheader(f"Card {index}")

        st.markdown(
            f"**Front:** {card.front}"
        )

        if st.button(
            f"Show Answer {index}",
            key=f"flashcard_show_{index}",
        ):
            st.markdown(
                f"**Back:** {card.back}"
            )

        display_sources(card)


# ---------------------------------------------------------
# COMPARE
# ---------------------------------------------------------

elif mode == "Compare":
    st.header("Compare")

    topic_a = st.text_input(
        "Topic A",
        placeholder="RAG",
    )

    topic_b = st.text_input(
        "Topic B",
        placeholder="Fine-tuning",
    )

    if st.button(
        "Compare",
        type="primary",
    ):
        try:
            with st.spinner("Comparing topics..."):
                result = study_app.compare(
                    topic_a,
                    topic_b,
                )

            st.markdown(result.comparison)
            display_sources(result)

        except Exception as exc:
            st.error(str(exc))


# ---------------------------------------------------------
# PROGRESS
# ---------------------------------------------------------

elif mode == "Progress":
    st.header("Learning Progress")

    try:
        progress = study_app.progress()

        if not progress:
            st.info(
                "No learning progress has been recorded yet."
            )

        else:
            for item in progress:
                st.subheader(item.topic)

                col1, col2 = st.columns(2)

                with col1:
                    st.metric(
                        "Quiz Accuracy",
                        f"{item.quiz_accuracy:.0%}",
                    )

                with col2:
                    st.metric(
                        "Flashcard Retention",
                        f"{item.flashcard_retention:.0%}",
                    )

    except Exception as exc:
        st.error(str(exc))