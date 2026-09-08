from langchain_core.documents import Document


def clean_documents(documents: list[Document]) -> list[Document]:
    """
    Perform conservative text cleanup.

    This function intentionally avoids aggressive content rewriting.
    """
    cleaned_documents = []

    for document in documents:
        text = document.page_content

        # Normalize excessive blank lines.
        lines = [line.rstrip() for line in text.splitlines()]

        cleaned_lines = []
        previous_blank = False

        for line in lines:
            is_blank = not line.strip()

            if is_blank and previous_blank:
                continue

            cleaned_lines.append(line)
            previous_blank = is_blank

        cleaned_text = "\n".join(cleaned_lines).strip()

        if not cleaned_text:
            continue

        cleaned_document = Document(
            page_content=cleaned_text,
            metadata=dict(document.metadata),
        )

        cleaned_documents.append(cleaned_document)

    return cleaned_documents