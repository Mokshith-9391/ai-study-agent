from pathlib import Path

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
)


SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".txt",
    ".md",
}


def discover_files(root_folder: str) -> list[Path]:
    """
    Recursively find supported study files.
    """
    root = Path(root_folder)

    if not root.exists():
        raise FileNotFoundError(
            f"Study folder does not exist: {root}"
        )

    files = []

    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            files.append(path)

    return sorted(files)


def load_file(path: Path, root_folder: Path) -> list[Document]:
    """
    Load one supported file and normalize its metadata.
    """

    suffix = path.suffix.lower()

    if suffix == ".pdf":
        loader = PyPDFLoader(str(path))

    elif suffix in {".txt", ".md"}:
        loader = TextLoader(
            str(path),
            encoding="utf-8",
        )

    else:
        raise ValueError(
            f"Unsupported file type: {path.suffix}"
        )

    documents = loader.load()

    relative_path = path.relative_to(root_folder)

    for document in documents:
        document.metadata["source"] = relative_path.as_posix()
        document.metadata["file_name"] = path.name
        document.metadata["file_type"] = suffix.lstrip(".")
        document.metadata["relative_path"] = relative_path.as_posix()

    return documents


def load_study_folder(root_folder: str) -> list[Document]:
    """
    Discover and load all supported study files.
    """

    root = Path(root_folder)

    files = discover_files(root)

    documents = []

    for file_path in files:
        print(f"Loading: {file_path}")

        file_documents = load_file(
            file_path,
            root,
        )

        documents.extend(file_documents)

    return documents