Basic documentation to onboard anyone:

md
Copy
Edit
# StudyGPT AI (Minimal RAG + Gemini API)

This is a simple educational AI assistant that explains topics, creates quizzes, and recommends next steps based on a textbook PDF.

## 🚀 How to Run

1. Install dependencies:

```bash
pip install -r requirements.txt
Add your API key to .env:

ini
Copy
Edit
GEMINI_API_KEY=your_api_key_here
Ingest your textbook:

bash
Copy
Edit
python run_ingestion.py
Start the server:

bash
Copy
Edit
uvicorn api.main:app --reload
Visit Swagger UI:

bash
Copy
Edit
http://localhost:8000/docs