# ============================================================
# config.py — ALL your settings live here


import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# Default URL - used when no runtime URL is provided
TARGET_URL = "https://en.wikipedia.org/wiki/Web_scraping"




LLM_MODEL = "llama-3.3-70b-versatile"
LLM_TEMPERATURE = 0  # 0 = focused answers, 1 = more creative

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

CHUNK_SIZE    = 1000
CHUNK_OVERLAP = 100
