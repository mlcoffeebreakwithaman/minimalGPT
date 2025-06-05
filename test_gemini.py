# test_gemini.py

import os
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# Configure with your API key
client = genai.Client(api_key=api_key)  
# Generate content
#response = client.models.generate_content(model='gemini-2.0-flash-001', contents='Why is the sky blue?')
#print(response.text)

response = client.models.embed_content(
    model='text-embedding-004',
    contents='why is the sky blue?',
)
if response.embeddings is not None:
    print(response.embeddings[0].values)
else:
    print("No embeddings found in the response:", response)
    