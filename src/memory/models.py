from dataclasses import dataclass


@dataclass
class StudySession:
    id: int
    created_at: str
    ended_at: str | None = None


@dataclass
class Message:
    id: int
    session_id: int
    role: str
    content: str
    created_at: str


@dataclass
class QuizAttempt:
    id: int
    session_id: int | None
    topic: str
    question: str
    selected_answer: str
    correct_answer: str
    is_correct: bool
    created_at: str


@dataclass
class FlashcardReview:
    id: int
    session_id: int | None
    topic: str
    front: str
    remembered: bool
    created_at: str


@dataclass
class TopicProgress:
    topic: str
    quiz_correct: int
    quiz_total: int
    flashcard_remembered: int
    flashcard_total: int
    updated_at: str | None

    @property
    def quiz_accuracy(self) -> float:
        if self.quiz_total == 0:
            return 0.0

        return self.quiz_correct / self.quiz_total

    @property
    def flashcard_retention(self) -> float:
        if self.flashcard_total == 0:
            return 0.0

        return (
            self.flashcard_remembered
            / self.flashcard_total
        )