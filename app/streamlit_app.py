import streamlit as st

from src.study_app import StudyApp


st.set_page_config(
    page_title="AI Study Agent",
    page_icon="📚",
    layout="wide",
)


@st.cache_resource
def get_app() -> StudyApp:
    return StudyApp()


app = get_app()


# ============================================================
# SESSION INITIALIZATION
# ============================================================

if "session_started" not in st.session_state:
    st.session_state.session_started = False

if "quiz_questions" not in st.session_state:
    st.session_state.quiz_questions = []

if "flashcards" not in st.session_state:
    st.session_state.flashcards = []

if "flashcard_index" not in st.session_state:
    st.session_state.flashcard_index = 0


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("📚 AI Study Agent")

mode = st.sidebar.radio(
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


# ============================================================
# SESSION CONTROLS
# ============================================================

st.sidebar.divider()
st.sidebar.subheader("Study Session")

if not st.session_state.session_started:

    if st.sidebar.button(
        "▶ Start Session",
        use_container_width=True,
    ):
        app.start_session()
        st.session_state.session_started = True
        st.rerun()

else:

    st.sidebar.success("Session active")

    if st.sidebar.button(
        "■ End Session",
        use_container_width=True,
    ):
        app.end_session()
        st.session_state.session_started = False
        st.rerun()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def show_sources(evidence):
    if not evidence:
        return

    with st.expander("📖 Sources"):
        for item in evidence:
            page = (
                f", page {item.page}"
                if item.page is not None
                else ""
            )

            st.markdown(
                f"**{item.source}{page}**"
            )

            st.caption(
                f"Distance: {item.distance:.4f}"
            )

            st.write(item.content)


def record_conversation(
    question: str,
    answer: str,
):
    if not st.session_state.session_started:
        return

    app.agent.record_user_message(question)
    app.agent.record_assistant_message(answer)


# ============================================================
# HEADER
# ============================================================

st.title("📚 AI Study Agent")

st.caption(
    "Local study documents → retrieval → grounded AI → citations"
)


# ============================================================
# ASK
# ============================================================

if mode == "Ask":

    st.header("Ask a Question")

    question = st.text_area(
        "What do you want to learn?",
        placeholder="Example: What is Retrieval-Augmented Generation?",
        height=120,
    )

    if st.button(
        "Ask",
        type="primary",
    ):

        if not question.strip():
            st.warning(
                "Please enter a question."
            )

        else:
            with st.spinner(
                "Searching your study material..."
            ):
                try:
                    result = app.ask(question)

                    record_conversation(
                        question,
                        result.answer,
                    )

                    st.markdown(
                        "### Answer"
                    )

                    st.write(
                        result.answer
                    )

                    show_sources(
                        result.evidence
                    )

                except Exception as error:
                    st.error(
                        f"Request failed: {error}"
                    )


# ============================================================
# EXPLAIN
# ============================================================

elif mode == "Explain":

    st.header("Explain a Topic")

    topic = st.text_input(
        "Topic",
        placeholder="Example: embeddings",
    )

    if st.button(
        "Explain",
        type="primary",
    ):

        if not topic.strip():
            st.warning(
                "Please enter a topic."
            )

        else:
            with st.spinner(
                "Preparing explanation..."
            ):
                try:
                    result = app.explain(topic)

                    record_conversation(
                        f"Explain {topic}",
                        result.answer,
                    )

                    st.markdown(
                        "### Explanation"
                    )

                    st.write(
                        result.answer
                    )

                    show_sources(
                        result.evidence
                    )

                except Exception as error:
                    st.error(
                        f"Request failed: {error}"
                    )


# ============================================================
# SUMMARIZE
# ============================================================

elif mode == "Summarize":

    st.header("Summarize")

    topic = st.text_input(
        "Topic or document subject",
        placeholder="Example: RAG",
    )

    if st.button(
        "Summarize",
        type="primary",
    ):

        if not topic.strip():
            st.warning(
                "Please enter a topic."
            )

        else:
            with st.spinner(
                "Creating summary..."
            ):
                try:
                    result = app.summarize(topic)

                    record_conversation(
                        f"Summarize {topic}",
                        result.answer,
                    )

                    st.markdown(
                        "### Summary"
                    )

                    st.write(
                        result.answer
                    )

                    show_sources(
                        result.evidence
                    )

                except Exception as error:
                    st.error(
                        f"Request failed: {error}"
                    )


# ============================================================
# QUIZ
# ============================================================

elif mode == "Quiz":

    st.header("🧠 Quiz")

    topic = st.text_input(
        "Quiz topic",
        placeholder="Example: RAG",
    )

    number = st.slider(
        "Number of questions",
        min_value=1,
        max_value=10,
        value=5,
    )

    if st.button(
        "Generate Quiz",
        type="primary",
    ):

        if not topic.strip():
            st.warning(
                "Please enter a topic."
            )

        else:
            with st.spinner(
                "Generating quiz..."
            ):
                try:
                    questions = app.quiz(
                        topic,
                        number,
                    )

                    st.session_state.quiz_questions = questions

                except Exception as error:
                    st.error(
                        f"Quiz generation failed: {error}"
                    )

    questions = st.session_state.quiz_questions

    if questions:

        st.divider()

        score = 0

        for index, question in enumerate(
            questions,
            start=1,
        ):

            st.markdown(
                f"### {index}. {question.question}"
            )

            selected = st.radio(
                "Choose an answer:",
                question.options,
                key=f"quiz_{index}",
            )

            if st.button(
                f"Check answer {index}",
                key=f"check_{index}",
            ):

                if selected == question.answer:

                    st.success(
                        "✅ Correct!"
                    )

                    score += 1

                    app.agent.record_quiz_attempt(
                        topic=topic,
                        question=question.question,
                        selected_answer=selected,
                        correct_answer=question.answer,
                    )

                else:

                    st.error(
                        f"❌ Incorrect. Correct answer: "
                        f"{question.answer}"
                    )

                    app.agent.record_quiz_attempt(
                        topic=topic,
                        question=question.question,
                        selected_answer=selected,
                        correct_answer=question.answer,
                    )

                st.info(
                    question.explanation
                )

                show_sources(
                    question.evidence
                )


# ============================================================
# FLASHCARDS
# ============================================================

elif mode == "Flashcards":

    st.header("🗂️ Flashcards")

    topic = st.text_input(
        "Flashcard topic",
        placeholder="Example: embeddings",
    )

    number = st.slider(
        "Number of cards",
        min_value=1,
        max_value=10,
        value=5,
    )

    if st.button(
        "Generate Flashcards",
        type="primary",
    ):

        if not topic.strip():
            st.warning(
                "Please enter a topic."
            )

        else:
            with st.spinner(
                "Generating flashcards..."
            ):
                try:
                    cards = app.flashcards(
                        topic,
                        number,
                    )

                    st.session_state.flashcards = cards
                    st.session_state.flashcard_index = 0

                except Exception as error:
                    st.error(
                        f"Flashcard generation failed: {error}"
                    )

    cards = st.session_state.flashcards

    if cards:

        index = st.session_state.flashcard_index
        card = cards[index]

        st.divider()

        st.caption(
            f"Card {index + 1} of {len(cards)}"
        )

        st.markdown(
            f"## {card.front}"
        )

        if st.button(
            "Show Answer",
            key=f"show_{index}",
        ):
            st.info(card.back)

            show_sources(
                card.evidence
            )

            col1, col2 = st.columns(2)

            with col1:

                if st.button(
                    "😕 Didn't remember",
                    key=f"forgot_{index}",
                ):
                    app.agent.record_flashcard_review(
                        topic=topic,
                        front=card.front,
                        remembered=False,
                    )

                    if index < len(cards) - 1:
                        st.session_state.flashcard_index += 1
                        st.rerun()

            with col2:

                if st.button(
                    "✅ Remembered",
                    key=f"remember_{index}",
                ):
                    app.agent.record_flashcard_review(
                        topic=topic,
                        front=card.front,
                        remembered=True,
                    )

                    if index < len(cards) - 1:
                        st.session_state.flashcard_index += 1
                        st.rerun()

        if index == len(cards) - 1:

            st.success(
                "🎉 You've reached the end of the flashcards."
            )


# ============================================================
# COMPARE
# ============================================================

elif mode == "Compare":

    st.header("⚖️ Compare Topics")

    col1, col2 = st.columns(2)

    with col1:
        topic_a = st.text_input(
            "Topic A",
            placeholder="RAG",
        )

    with col2:
        topic_b = st.text_input(
            "Topic B",
            placeholder="Fine-tuning",
        )

    if st.button(
        "Compare",
        type="primary",
    ):

        if not topic_a.strip() or not topic_b.strip():

            st.warning(
                "Please enter both topics."
            )

        else:

            with st.spinner(
                "Comparing topics..."
            ):

                try:

                    result = app.compare(
                        topic_a,
                        topic_b,
                    )

                    st.markdown(
                        f"### {result.topic_a} vs {result.topic_b}"
                    )

                    st.write(
                        result.comparison
                    )

                    show_sources(
                        result.evidence
                    )

                except Exception as error:

                    st.error(
                        f"Comparison failed: {error}"
                    )


# ============================================================
# PROGRESS
# ============================================================

elif mode == "Progress":

    st.header("📊 Learning Progress")

    try:

        progress = app.progress()

        if not progress:

            st.info(
                "No learning progress recorded yet."
            )

        else:

            for item in progress:

                st.subheader(
                    item.topic
                )

                col1, col2, col3, col4 = st.columns(4)

                with col1:

                    st.metric(
                        "Quiz Questions",
                        item.questions_answered,
                    )

                with col2:

                    st.metric(
                        "Quiz Accuracy",
                        f"{item.quiz_accuracy:.0%}",
                    )

                with col3:

                    st.metric(
                        "Cards Reviewed",
                        item.flashcards_reviewed,
                    )

                with col4:

                    st.metric(
                        "Retention",
                        f"{item.flashcard_retention:.0%}",
                    )

                st.divider()

    except Exception as error:

        st.error(
            f"Could not load progress: {error}"
        )