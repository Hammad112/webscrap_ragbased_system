# 🕷️ Web Scraping + RAG System

A beginner-friendly, modular Python project that:
1. **Scrapes** any webpage
2. **Cleans** the raw text
3. **Answers your questions** about the page using AI (RAG)

---

## 📁 Project Structure

```
webscraping_rag_project/
│
├── config.py              ← Settings and env loading (API key from `.env` or Streamlit secrets)
├── app.py                 ← Streamlit UI (main way to run the app)
├── pipeline.py            ← Same flow as a Python class (for scripts / learning)
├── storage.py             ← Saves FAISS index + metadata under `knowledge_bases/`
│
├── scraper/
│   ├── __init__.py
│   └── web_scraper.py     ← Downloads webpages
│
├── cleaner/
│   ├── __init__.py
│   └── text_cleaner.py    ← Cleans and splits text into chunks
│
├── rag/
│   ├── __init__.py
│   └── rag_system.py      ← Builds the AI Q&A system
│
├── requirements.txt       ← All packages needed
└── .env                   ← Create this locally: `GROQ_API_KEY=...` (not committed)
```

---

## 🚀 How to Run

### Step 1 — Install packages
```bash
pip install -r requirements.txt
```

### Step 2 — Add your Groq API key

Create a `.env` file in the project folder (same directory as `config.py`):

```env
GROQ_API_KEY=your_key_here
```

Get a **free** key at https://console.groq.com

On **Streamlit Cloud**, use App secrets or `.streamlit/secrets.toml` with `GROQ_API_KEY` instead.

### Step 3 — Run the Streamlit UI
```bash
streamlit run app.py
```

Opens at http://localhost:8501

### Optional — Use the pipeline in code

Import `Pipeline` from `pipeline.py` and call `run_all()` / `ask()` from your own script (see docstrings in that file).

---

## 🔑 Key Concepts

| Concept | What it means |
|---|---|
| **Scraping** | Downloading a webpage and pulling out the text |
| **Cleaning** | Removing junk HTML, menus, footers, extra whitespace |
| **Chunking** | Splitting long text into small pieces the AI can read |
| **Embeddings** | Converting text to numbers so AI can search it |
| **FAISS** | A fast database that stores and searches embeddings |
| **RAG** | AI reads your scraped page, then answers questions about it |

---

## 💡 How it flows

```
URL → scraper → raw text → cleaner → clean chunks → RAG → answers
```
