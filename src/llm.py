import os
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai import errors

load_dotenv()

LLM_MODEL = "gemini-3.8-flash"

MAX_RETRIES = 3
INITIAL_RETRY_DELAY = 2


class GeminiLLM:
    """
    Gemini-based language model provider.
    """

    def __init__(
        self,
        model: str = LLM_MODEL,
    ) -> None:
        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY was not found."
            )

        self.client = genai.Client(
            api_key=api_key
        )

        self.model = model

    def generate(
        self,
        prompt: str,
    ) -> str:
        last_error = None

        for attempt in range(MAX_RETRIES):
            try:
                response = (
                    self.client.models.generate_content(
                        model=self.model,
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            thinking_config=(
                                types.ThinkingConfig(
                                    thinking_level="low"
                                )
                            ),
                            automatic_function_calling=(
                                types.AutomaticFunctionCallingConfig(
                                    disable=True
                                )
                            ),
                        ),
                    )
                )

                if not response.text:
                    raise RuntimeError(
                        "Gemini returned an empty response."
                    )

                return response.text

            except errors.ServerError as error:
                last_error = error

                if attempt == MAX_RETRIES - 1:
                    break

                delay = INITIAL_RETRY_DELAY * (
                    2 ** attempt
                )

                print(
                    f"Gemini server error. "
                    f"Retrying in {delay} seconds..."
                )

                time.sleep(delay)

            except errors.ClientError as error:
                raise RuntimeError(
                    f"Gemini client error: {error}"
                ) from error

            except errors.APIError as error:
                raise RuntimeError(
                    f"Gemini API error: {error}"
                ) from error

        raise RuntimeError(
            "Gemini server request failed after "
            f"{MAX_RETRIES} attempts."
        ) from last_error