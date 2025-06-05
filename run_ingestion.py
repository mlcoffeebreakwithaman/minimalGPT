# run_ingestion.py
import os
import json
from dotenv import load_dotenv
from pathlib import Path
from vector_store import FaissStore, embed_content_batch
import fitz  # PyMuPDF

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Constants
PDF_PATH = "data/grade7.pdf"
CHUNKS_PATH = "data/chunks.json"
INDEX_PATH = "data/faiss_index.bin"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

def extract_text_from_pdf(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    return "\n".join([page.get_text() for page in doc])

def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += size - overlap
    return chunks

if __name__ == "__main__":
    try:
        print("📖 Extracting text from PDF...")
        text = extract_text_from_pdf(PDF_PATH)

        print("✂️ Chunking text...")
        chunks = chunk_text(text)

        print("🔮 Getting embeddings from Gemini...")
        embeddings = embed_content_batch(chunks)

        print("📦 Creating FAISS store...")
        store = FaissStore(dim=len(embeddings[0]), index_path=INDEX_PATH)
        store.save(embeddings)

        print("📝 Saving chunks to JSON...")
        with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
            json.dump([{"id": i, "text": chunk} for i, chunk in enumerate(chunks)], f, indent=2)

        print(f"✅ Ingestion complete. Total chunks: {len(chunks)}")

    except Exception as e:
        print(f"❌ Ingestion failed: {e}")
