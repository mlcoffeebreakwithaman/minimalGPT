Basic documentation to onboard anyone:


# StudyGPT AI (Minimal RAG + Gemini API)

This is a simple educational AI assistant that explains topics, creates quizzes, and recommends next steps based on a textbook PDF.

## 🚀 How to Run

1. Install dependencies:


pip install -r requirements.txt
Add your API key to .env:


GEMINI_API_KEY=your_api_key_here
Ingest your textbook:


python run_ingestion.py
Start the server:


uvicorn api.main:app --reload
Visit Swagger UI:

http://localhost:8000/docs