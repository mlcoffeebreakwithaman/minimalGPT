from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from agent import SimpleStudyAgent


# Request model
class AskRequest(BaseModel):
    topic: str

# Initialize FastAPI app
app = FastAPI(title="StudyGPT AI - Simple Agent API")

# Enable CORS for all origins (adjust for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Instantiate the agent (it internally loads VectorStore)
agent = SimpleStudyAgent()

# Health check route
@app.get("/")
def root():
    return {"message": "📚 StudyGPT API is running!"}

# Explanation endpoint
@app.post("/ask")
def ask(req: AskRequest):
    try:
        explanation = agent.explain(req.topic)
        return {"explanation": explanation}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Quiz generation endpoint
@app.post("/quiz")
def quiz(req: AskRequest):
    try:
        quiz_text = agent.quiz(req.topic)
        return {"quiz": quiz_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Recommendation endpoint
@app.post("/recommend")
def recommend():
    try:
        rec = agent.recommend()
        return {"recommendation": rec}
    except Exception as e:
        
        raise HTTPException(status_code=500, detail=str(e))
