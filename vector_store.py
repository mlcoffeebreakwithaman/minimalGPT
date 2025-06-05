# vector_store.py
import os
import json
import faiss
import numpy as np
from dotenv import load_dotenv
from google import genai

load_dotenv()

# ----- Embedding Batch Function -----
def embed_content_batch(chunks: list[str]) -> list[np.ndarray]:
    """
    Embeds a list of text chunks using Gemini API.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Missing GEMINI_API_KEY in .env")

    client = genai.Client(api_key=api_key)
    embeddings = []

    for chunk in chunks:
        response = client.models.embed_content(
            model="models/text-embedding-004",
            contents=chunk,
        )
        if not response.embeddings or not response.embeddings[0]:
            raise ValueError("Empty or invalid embedding response.")
        emb = np.array(response.embeddings[0].values, dtype=np.float32)
        embeddings.append(emb)

    return embeddings

# ----- FAISS Store Class (For Ingestion and Saving) -----
class FaissStore:
    def __init__(self, dim: int, index_path: str):
        self.index_path = index_path
        self.dim = dim
        self.index = faiss.IndexFlatL2(dim)

    def save(self, embeddings: list[np.ndarray]):
        matrix = np.vstack(embeddings).astype(np.float32)
        self.index.add(matrix)
        faiss.write_index(self.index, self.index_path)

    def load(self):
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
        else:
            raise FileNotFoundError(f"Index not found at {self.index_path}")

# ----- VectorStore Class (For Retrieval) -----
class VectorStore:
    def __init__(self):
        self.index_path = "data/faiss_index.bin"
        self.chunks_path = "data/chunks.json"
        self.api_key = os.getenv("GEMINI_API_KEY")

        if not self.api_key:
            raise ValueError("Missing GEMINI_API_KEY in .env")

        self.client = genai.Client(api_key=self.api_key)
        self.index = self._load_index()
        self.chunks = self._load_chunks()

    def _load_index(self):
        if not os.path.exists(self.index_path):
            raise FileNotFoundError(f"Missing FAISS index: {self.index_path}")
        return faiss.read_index(self.index_path)

    def _load_chunks(self):
        if not os.path.exists(self.chunks_path):
            raise FileNotFoundError(f"Missing chunks: {self.chunks_path}")
        with open(self.chunks_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _embed(self, text: str) -> np.ndarray:
        response = self.client.models.embed_content(
            model="models/text-embedding-004",
            contents=text,
        )
        if not response.embeddings or not response.embeddings[0]:
            raise ValueError("Empty or invalid embedding response.")
        return np.array(response.embeddings[0].values, dtype=np.float32).reshape(1, -1)

    def retrieve(self, query: str, k: int = 3):
        try:
            query_vec = self._embed(query)
            distances, indices = self.index.search(query_vec, k)
            return [self.chunks[i] for i in indices[0] if i < len(self.chunks)]
        except Exception as e:
            print(f"VectorStore error: {str(e)}")
            return []
