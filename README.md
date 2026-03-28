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
├── config.py              ← Your API key and settings (edit this first!)
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
├── utils/
│   ├── __init__.py
│   └── helpers.py         ← Small helper functions
│
├── main.py                ← Run this to start everything
└── requirements.txt       ← All packages needed
```

---

## 🚀 How to Run

### Step 1 — Install packages
```bash
pip install -r requirements.txt
```

### Step 2 — Add your Groq API key
Open `config.py` and paste your key:
```python
GROQ_API_KEY = "your_key_here"
```
Get a **free** key at 👉 https://console.groq.com

### Step 3a — Run via terminal
```bash
python main.py
```

### Step 3b — Run the Streamlit UI (testing)
```bash
streamlit run app.py
```
Opens at http://localhost:8501

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
