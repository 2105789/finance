import os
from langchain_google_genai import ChatGoogleGenerativeAI
from core.config import GEMINI_API_KEY

def get_llm():
    """Initializes and returns the Gemini LLM."""
    os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.3)
    return llm 