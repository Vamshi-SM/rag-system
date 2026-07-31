from google import genai
from google.genai import types
import time

client = genai.Client(
    vertexai=True,
    project="rag-system-503214",
    location="us-central1",
)

for i in range(5):
    start = time.perf_counter()

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents="Say hello in one sentence.",
        config=types.GenerateContentConfig(
            temperature=0,
            max_output_tokens=20,
        ),
    )

    print(f"{i+1}: {time.perf_counter() - start:.2f}s")