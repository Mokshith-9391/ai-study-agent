import os

from dotenv import load_dotenv
from google import genai
from google.genai import types


load_dotenv()


EMBEDDING_MODEL = "gemini-embedding-2"
EMBEDDING_DIMENSION = 768


class GeminiEmbedder:
    """
    Generates embeddings using Google's Gemini Embedding API.
    """

    def __init__(self) -> None:
        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY was not found."
            )

        self.client = genai.Client(api_key=api_key)

    def embed_text(
        self,
        text: str,
    ) -> list[float]:
        """
        Generate one embedding for one piece of text.
        """

        response = self.client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=text,
            config=types.EmbedContentConfig(
                output_dimensionality=EMBEDDING_DIMENSION,
            ),
        )

        if not response.embeddings:
            raise RuntimeError(
                "Embedding API returned no embedding."
            )

        vector = response.embeddings[0].values

        if len(vector) != EMBEDDING_DIMENSION:
            raise RuntimeError(
                f"Expected {EMBEDDING_DIMENSION} dimensions, "
                f"got {len(vector)}."
            )

        return vector

    def embed_query(
        self,
        text: str,
    ) -> list[float]:
        """
        Generate an embedding for a search query.
        """

        return self.embed_text(text)

    def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        """
        Generate separate embeddings for multiple documents.
        """

        if not texts:
            return []

        all_vectors = []

        for text in texts:
            vector = self.embed_text(text)
            all_vectors.append(vector)

        return all_vectors