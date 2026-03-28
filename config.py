# ============================================================
# config.py — ALL your settings live here


import os
from dotenv import load_dotenv

# Load environment variables from .env file (local dev)
load_dotenv()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# Streamlit Cloud / `streamlit run`: use App secrets or .streamlit/secrets.toml
try:
    import streamlit as st

    _from_secrets = st.secrets.get("GROQ_API_KEY")
    if _from_secrets:
        GROQ_API_KEY = _from_secrets
except Exception:
    pass

# Default URL - used when no runtime URL is provided
TARGET_URL = "https://en.wikipedia.org/wiki/Web_scraping"




LLM_MODEL = "llama-3.3-70b-versatile"
LLM_TEMPERATURE = 0  # 0 = focused answers, 1 = more creative

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

CHUNK_SIZE    = 1000
CHUNK_OVERLAP = 100
