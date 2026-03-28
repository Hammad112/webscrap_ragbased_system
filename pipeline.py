# ============================================================
# pipeline.py — THE STEP-BY-STEP PIPELINE
# ============================================================
# PIPELINE FLOW:
#   Step 1 → scrape_page()
#   Step 2 → clean_and_chunk()
#   Step 3 → summarize()
#   Step 4 → build_rag()
#   Step 5 → ask() 
# ============================================================

import config

from scraper.web_scraper  import scrape_url
from cleaner.text_cleaner import clean_and_split
from rag.rag_system       import (
    build_vector_store,
    build_qa_chain,
    ask_question,
    summarize_page,
)


class Pipeline:
    """
    Wraps the full scrape → clean → RAG flow into one object.

    Attributes set after each step:
        .docs         → raw scraped Documents       (after step 1)
        .chunks       → cleaned + split chunks      (after step 2)
        .summary      → one-paragraph summary       (after step 3)
        .vector_store → FAISS vector database       (after step 4)
        .qa_chain     → ready-to-use Q&A chain      (after step 4)
        .status       → dict tracking which steps are done
    """

    def __init__(self, url: str, groq_api_key: str,
                 llm_model: str        = config.LLM_MODEL,
                 embedding_model: str  = config.EMBEDDING_MODEL,
                 chunk_size: int       = config.CHUNK_SIZE,
                 chunk_overlap: int    = config.CHUNK_OVERLAP,
                 temperature: float    = config.LLM_TEMPERATURE):
      
        self.url             = url
        self.groq_api_key    = groq_api_key
        self.llm_model       = llm_model
        self.embedding_model = embedding_model
        self.chunk_size      = chunk_size
        self.chunk_overlap   = chunk_overlap
        self.temperature     = temperature

        # These are populated as each step runs
        self.docs         = None
        self.chunks       = None
        self.summary      = None
        self.vector_store = None
        self.qa_chain     = None

        # Track which steps completed successfully
        self.status = {
            "step1_scrape":  False,
            "step2_clean":   False,
            "step3_summary": False,
            "step4_rag":     False,
        }

    # ----------------------------------------------------------
    # STEP 1 — Scrape
    # ----------------------------------------------------------
    def step1_scrape(self, log=print):
        """
        Download the webpage and store raw Documents in self.docs.

        Args:
            log: A function to print messages (default: print).
                 Streamlit passes st.write here so messages show in UI.
        """
        log(f"📥 Step 1 — Scraping: {self.url}")
        self.docs = scrape_url(self.url)
        chars = len(self.docs[0].page_content)
        log(f"   ✅ Got {chars:,} characters from the page")
        self.status["step1_scrape"] = True

    # ----------------------------------------------------------
    # STEP 2 — Clean & Chunk
    # ----------------------------------------------------------
    def step2_clean(self, log=print):
        """
        Clean the raw text and split it into chunks.
        Stores result in self.chunks.

        Requires: step1_scrape() must have run first.
        """
        if not self.status["step1_scrape"]:
            raise RuntimeError("Run step1_scrape() first!")

        log(f"🧹 Step 2 — Cleaning & splitting (chunk_size={self.chunk_size})")
        self.chunks = clean_and_split(
            self.docs,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        log(f"   ✅ Created {len(self.chunks)} chunks")
        self.status["step2_clean"] = True

    # ----------------------------------------------------------
    # STEP 3 — Summarize
    # ----------------------------------------------------------
    def step3_summarize(self, log=print):
        """
        Ask Groq to summarize the page in a few sentences.
        Stores result in self.summary.

        Requires: step1_scrape() must have run first.
        """
        if not self.status["step1_scrape"]:
            raise RuntimeError("Run step1_scrape() first!")

        log("📝 Step 3 — Summarizing page with AI...")
        self.summary = summarize_page(
            self.docs,
            groq_api_key=self.groq_api_key,
            model=self.llm_model,
        )
        log("   ✅ Summary ready")
        self.status["step3_summary"] = True

    # ----------------------------------------------------------
    # STEP 4 — Build RAG
    # ----------------------------------------------------------
    def step4_build_rag(self, log=print):
        """
        Build the vector store and Q&A chain.
        Stores results in self.vector_store and self.qa_chain.

        Requires: step2_clean() must have run first.
        """
        if not self.status["step2_clean"]:
            raise RuntimeError("Run step2_clean() first!")

        log(f"🔢 Step 4a — Building vector store ({self.embedding_model})...")
        log("   (First run downloads the model — ~1 minute)")
        self.vector_store = build_vector_store(
            self.chunks,
            embedding_model=self.embedding_model,
        )

        log(f"🤖 Step 4b — Connecting to Groq LLM ({self.llm_model})...")
        self.qa_chain = build_qa_chain(
            db=self.vector_store,
            groq_api_key=self.groq_api_key,
            model=self.llm_model,
            temperature=self.temperature,
        )

        log("   ✅ RAG system ready — you can now ask questions!")
        self.status["step4_rag"] = True

    # ----------------------------------------------------------
    # ASK — after step 4 is done
    # ----------------------------------------------------------
    def ask(self, question: str) -> str:
        """
        Ask any question about the scraped page.

        Args:
            question: Any question string.

        Returns:
            The AI's answer as a string.

        Requires: step4_build_rag() must have run first.

        Example:
            answer = pipe.ask("What is web scraping?")
            print(answer)
        """
        if not self.status["step4_rag"]:
            raise RuntimeError("Run step4_build_rag() (or run_all()) first!")
        return ask_question(self.qa_chain, question)

    # ----------------------------------------------------------
    # RUN ALL — convenience method
    # ----------------------------------------------------------
    def run_all(self, log=print):
        """
        Run all 4 steps in order.
        After this you can call .ask() as many times as you want.

        Args:
            log: Print function (default: print, or pass st.write for Streamlit).

        Example:
            pipe = Pipeline(url="...", groq_api_key="gsk_...")
            pipe.run_all()
            print(pipe.ask("What is this page about?"))
        """
        self.step1_scrape(log)
        self.step2_clean(log)
        self.step3_summarize(log)
        self.step4_build_rag(log)
        log("\n🎉 Pipeline complete! Call pipe.ask('your question') to query.")

    # ----------------------------------------------------------
    # INFO — show current state
    # ----------------------------------------------------------
    def info(self) -> dict:
        """
        Return a summary of the pipeline's current state.
        Useful for debugging or displaying in Streamlit.

        Returns:
            Dict with url, status flags, chunk count, and summary snippet.
        """
        return {
            "url":         self.url,
            "status":      self.status,
            "chunk_count": len(self.chunks) if self.chunks else 0,
            "summary":     self.summary or "Not generated yet",
        }
