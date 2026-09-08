from dataclasses import dataclass
from typing import Any


class ValidationError(ValueError):
    """Raised when generated study content is invalid."""


@dataclass
class ValidationResult:
    valid: bool
    errors: list[str]


def validate_quiz_item(
    item: Any,
) -> ValidationResult:

    errors = []

    if not isinstance(item, dict):
        return ValidationResult(
            valid=False,
            errors=["Quiz item must be an object."],
        )

    required_fields = [
        "question",
        "options",
        "answer",
        "explanation",
        "evidence",
    ]

    for field in required_fields:
        if field not in item:
            errors.append(
                f"Missing field: {field}"
            )

    if errors:
        return ValidationResult(
            valid=False,
            errors=errors,
        )

    if not isinstance(item["question"], str):
        errors.append(
            "question must be a string."
        )

    options = item["options"]

    if not isinstance(options, list):
        errors.append(
            "options must be a list."
        )
    elif len(options) != 4:
        errors.append(
            "options must contain exactly four items."
        )
    elif not all(
        isinstance(option, str)
        for option in options
    ):
        errors.append(
            "all options must be strings."
        )

    if isinstance(options, list):
        if item["answer"] not in options:
            errors.append(
                "answer must exactly match one option."
            )

    if not isinstance(
        item["explanation"],
        str,
    ):
        errors.append(
            "explanation must be a string."
        )

    if not isinstance(
        item["evidence"],
        list,
    ):
        errors.append(
            "evidence must be a list."
        )

    return ValidationResult(
        valid=not errors,
        errors=errors,
    )


def validate_flashcard_item(
    item: Any,
) -> ValidationResult:

    errors = []

    if not isinstance(item, dict):
        return ValidationResult(
            valid=False,
            errors=["Flashcard must be an object."],
        )

    required_fields = [
        "front",
        "back",
        "evidence",
    ]

    for field in required_fields:
        if field not in item:
            errors.append(
                f"Missing field: {field}"
            )

    if errors:
        return ValidationResult(
            valid=False,
            errors=errors,
        )

    if not isinstance(item["front"], str):
        errors.append(
            "front must be a string."
        )

    if not isinstance(item["back"], str):
        errors.append(
            "back must be a string."
        )

    if not isinstance(item["evidence"], list):
        errors.append(
            "evidence must be a list."
        )

    return ValidationResult(
        valid=not errors,
        errors=errors,
    )


def validate_comparison(
    data: Any,
) -> ValidationResult:

    errors = []

    if not isinstance(data, dict):
        return ValidationResult(
            valid=False,
            errors=["Comparison must be an object."],
        )

    required_fields = [
        "topic_a",
        "topic_b",
        "comparison",
        "evidence",
    ]

    for field in required_fields:
        if field not in data:
            errors.append(
                f"Missing field: {field}"
            )

    if errors:
        return ValidationResult(
            valid=False,
            errors=errors,
        )

    for field in [
        "topic_a",
        "topic_b",
        "comparison",
    ]:
        if not isinstance(data[field], str):
            errors.append(
                f"{field} must be a string."
            )

    if not isinstance(data["evidence"], list):
        errors.append(
            "evidence must be a list."
        )

    return ValidationResult(
        valid=not errors,
        errors=errors,
    )


def require_valid(
    result: ValidationResult,
) -> None:

    if result.valid:
        return

    raise ValidationError(
        "Generated content failed validation: "
        + "; ".join(result.errors)
    )