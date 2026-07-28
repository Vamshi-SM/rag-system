from google import genai
from google.genai import types

from src.llm.base_llm import BaseLLM, ChatMessage, LLMError
from src.utils.profiler import measure


class GeminiChat(BaseLLM):
    def __init__(self, project_id: str, location: str = "us-central1"):
        self.client = genai.Client(
            vertexai=True,
            project=project_id,
            location=location,
        )

        self.model = "gemini-2.5-flash"

    def generate(self, prompt: str) -> str:
        try:
            # Stage 6 = the network round-trip to Gemini (model inference +
            # transport); stage 7 = extracting the text from the returned
            # response object. They're one logical call but split here so the
            # summary can show how much (if any) of LLM time is local parsing
            # vs. the actual remote request.
            with measure("6. LLM call (Gemini generate_content)"):
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                )
            with measure("7. receive/extract response"):
                return response.text
        except Exception as e:
            raise LLMError(str(e))

    def chat(self, messages: list[ChatMessage]) -> str:
        prompt = "\n".join(
            f"{m['role']}: {m['content']}" for m in messages
        )
        return self.generate(prompt)