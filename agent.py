# agent.py

import os
from pathlib import Path
from dotenv import load_dotenv
from vector_store import VectorStore
from google import genai

# Load environment variables
load_dotenv()

# Setup Gemini client
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("Missing GEMINI_API_KEY in .env")


client = genai.Client(api_key=api_key)

# Constants
PROMPT_DIR = Path("prompts")
CHUNK_TOP_K = 3

MODEL_NAME = "gemini-2.0-flash-001"  # You can change this to pro if needed

class SimpleStudyAgent:
    def __init__(self):
        self.vector_store = VectorStore()

    def _load_prompt(self, template_name: str) -> str:
        path = PROMPT_DIR / template_name
        if not path.exists():
            raise FileNotFoundError(f"Prompt template not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def _fill_prompt(self, template: str, topic: str, context: str) -> str:
        return template.replace("{{topic}}", topic).replace("{{context}}", context)

    
    def _generate(self, prompt: str) -> str:
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt
            )
            return response.text.strip() if response.text is not None else ""
        except Exception as e:
            return f"❌ Error generating response: {e}"

    def explain(self, topic: str) -> str:
        chunks = self.vector_store.retrieve(topic, k=CHUNK_TOP_K)
        if not chunks:
            return "⚠️ No relevant content found to explain the topic."

        context = "\n".join(chunk["text"] for chunk in chunks)
        template = self._load_prompt("tutor_prompt.txt")
        prompt = self._fill_prompt(template, topic, context)
        return self._generate(prompt)

    def quiz(self, topic: str) -> str:
        chunks = self.vector_store.retrieve(topic, k=CHUNK_TOP_K)
        if not chunks:
            return "⚠️ Not enough information found to generate a quiz."

        context = "\n".join(chunk["text"] for chunk in chunks)
        template = self._load_prompt("quiz_prompt.txt")
        prompt = self._fill_prompt(template, topic, context)
        return self._generate(prompt)

    def recommend(self) -> str:
        template = self._load_prompt("recommend_prompt.txt")
        prompt = template.replace("{{progress}}", "basic student performance data")
        return self._generate(prompt)
