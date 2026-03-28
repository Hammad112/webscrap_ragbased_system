# ============================================================
# app.py — Clean Step-by-Step Web Scraping RAG System
# ============================================================

import streamlit as st
import config
from scraper.web_scraper import scrape_url
from cleaner.text_cleaner import clean_and_split
from rag.rag_system import (
    build_vector_store,
    build_qa_chain,
    ask_question,
    summarize_page,
)
from utils.helpers import validate_api_key
from urllib.parse import urlparse
from storage import save_knowledge_base, load_knowledge_base, list_saved_knowledge_bases
import time
import os
import re

# Page configuration
st.set_page_config(
    page_title="Web Scraping RAG System",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark theme CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #2E7D32;
        margin-bottom: 2rem;
    }
    .step-header {
        font-size: 1.3rem;
        font-weight: 600;
        color: #2E7D32;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .success-box {
        background-color: #1B5E20;
        border: 1px solid #2E7D32;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
        color: white;
    }
    .error-box {
        background-color: #B71C1C;
        border: 1px solid #D32F2F;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
        color: white;
    }
    .info-box {
        background-color: #1A237E;
        border: 1px solid #283593;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
        color: white;
    }
    .step-complete {
        background-color: #1B5E20;
        border-left: 4px solid #4CAF50;
        padding: 1rem;
        margin: 1rem 0;
        color: white;
    }
    .stButton > button {
        background-color: #2E7D32;
        color: white;
        border: none;
        font-weight: bold;
    }
    .stButton > button:hover {
        background-color: #1B5E20;
    }
    .stTextArea > div > div > textarea {
        background-color: #1E1E1E;
        color: white;
    }
    .stTextInput > div > div > input {
        background-color: #1E1E1E;
        color: white;
    }
    .stSelectbox > div > div > select {
        background-color: #1E1E1E;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">🌐 Web Scraping RAG System</div>', unsafe_allow_html=True)
st.markdown("---")

# Helper functions
def validate_url(url):
    """Validate that the input is a proper URL."""
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except:
        return False

def create_safe_kb_name(url):
    """Create a safe folder name from URL."""
    # Parse URL to get domain
    parsed = urlparse(url)
    domain = parsed.netloc
    
    # Remove www. and TLD
    domain = domain.replace('www.', '').split('.')[0]
    
    # Clean special characters
    domain = re.sub(r'[^a-zA-Z0-9]', '', domain)
    
    # Add timestamp to make unique
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    return f"{domain}_{timestamp}"

# Initialize session state
if 'step' not in st.session_state:
    st.session_state.step = 1
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None
if 'retriever' not in st.session_state:
    st.session_state.retriever = None
if 'summary' not in st.session_state:
    st.session_state.summary = None
if 'chunks_count' not in st.session_state:
    st.session_state.chunks_count = 0
if 'processed_url' not in st.session_state:
    st.session_state.processed_url = None
if 'qa_history' not in st.session_state:
    st.session_state.qa_history = []
if 'current_kb_name' not in st.session_state:
    st.session_state.current_kb_name = None

# Simple sidebar
with st.sidebar:
     
    # Knowledge Base Management
    st.subheader("📁 Knowledge Base")
    
    # List saved knowledge bases
    saved_kbs = list_saved_knowledge_bases()
    if saved_kbs:
        selected_kb = st.selectbox("Saved Knowledge Bases", ["None"] + saved_kbs)
        
        if selected_kb != "None" and selected_kb != st.session_state.get('current_kb_name', 'None'):
            if st.button("📂 Load Selected KB", key="load_kb"):
                vector_store, metadata = load_knowledge_base(f"knowledge_bases/{selected_kb}")
                if vector_store:
                    # Build QA chain
                    qa_chain_result = build_qa_chain(
                        db=vector_store,
                        groq_api_key=config.GROQ_API_KEY,
                        model="llama-3.3-70b-versatile",
                        temperature=0.0,
                    )
                    
                    # Extract chain and retriever
                    qa_chain, retriever = qa_chain_result
                    
                    st.session_state.vector_store = vector_store
                    st.session_state.qa_chain = qa_chain
                    st.session_state.retriever = retriever
                    st.session_state.summary = f"Loaded knowledge base from: {metadata['url']}"
                    st.session_state.chunks_count = metadata['chunks_count']
                    st.session_state.processed_url = metadata['url']
                    st.session_state.step = 2
                    st.session_state.current_kb_name = selected_kb
                    st.success(f"✅ Loaded {selected_kb}")
                    st.rerun()
    else:
        st.info("No saved knowledge bases found")
    
    # Progress
    st.divider()
    st.subheader("🔄 Progress")
    if st.session_state.step == 1:
        st.info("🔄 Step 1: Create Knowledge Base")
        st.markdown("⬜ Step 2: Ask Questions")
    elif st.session_state.step == 2:
        st.success("✅ Step 1: Knowledge Base Ready")
        st.info("🔄 Step 2: Ask Questions")
    
    # Current KB info
    if st.session_state.current_kb_name:
        st.markdown(f"**Current:** {st.session_state.current_kb_name}")
        if st.button("🗑️ Clear KB", key="clear_kb"):
            for key in ['step', 'vector_store', 'qa_chain', 'retriever', 'summary', 'chunks_count', 'processed_url', 'qa_history', 'current_kb_name']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()

# Main content
if st.session_state.step == 1:
    st.markdown('<div class="step-header">📥 Step 1: Create Knowledge Base</div>', unsafe_allow_html=True)
    
    # URL input
    url = st.text_input(
        "Enter URL to scrape",
        value="https://en.wikipedia.org/wiki/Web_scraping",
        placeholder="https://example.com"
    )
    
    # Create button
    col1, col2 = st.columns([2, 1])
    with col1:
        if st.button("🚀 Create Knowledge Base", type="primary", use_container_width=True):
            if not config.GROQ_API_KEY:
                st.error("❌ Set API key in .env file")
                st.stop()
            
            if not validate_url(url):
                st.error("❌ Enter valid URL")
                st.stop()
            
            # Process
            with st.spinner("Creating Knowledge Base..."):
                try:
                    # Scrape
                    docs = scrape_url(url)
                    
                    # Clean and split
                    chunks = clean_and_split(docs, chunk_size=1000, chunk_overlap=100)
                    
                    # Build vector store
                    vector_store = build_vector_store(chunks, embedding_model=config.EMBEDDING_MODEL)
                    
                    # Build QA chain
                    qa_chain_result = build_qa_chain(
                        db=vector_store,
                        groq_api_key=config.GROQ_API_KEY,
                        model="llama-3.3-70b-versatile",
                        temperature=0.0,
                    )
                    
                    # Extract chain and retriever from result
                    qa_chain, retriever = qa_chain_result
                    
                    # Summary
                    summary = summarize_page(docs, groq_api_key=config.GROQ_API_KEY, model="llama-3.3-70b-versatile")
                    
                    # Save to disk with safe name
                    kb_name = create_safe_kb_name(url)
                    save_path = save_knowledge_base(vector_store, chunks, url, f"knowledge_bases/{kb_name}")
                    
                    # Store in session
                    st.session_state.vector_store = vector_store
                    st.session_state.qa_chain = qa_chain
                    st.session_state.retriever = retriever
                    st.session_state.summary = summary
                    st.session_state.chunks_count = len(chunks)
                    st.session_state.processed_url = url
                    st.session_state.step = 2
                    st.session_state.current_kb_name = kb_name
                    
                    st.success(f"✅ Knowledge Base Created! {len(chunks)} chunks saved to disk")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
    
    with col2:
        st.markdown("**💾 Auto-save enabled**")
        st.markdown("Knowledge bases are saved to `knowledge_bases/` folder")

elif st.session_state.step == 2:
    st.markdown('<div class="step-header">💬 Step 2: Ask Questions</div>', unsafe_allow_html=True)
    
    # Show knowledge base status
    st.markdown(f"""
    <div class="step-complete">
        <strong>✅ Knowledge Base Ready!</strong><br>
        📄 URL: {st.session_state.processed_url}<br>
        🧩 {st.session_state.chunks_count} chunks indexed
    </div>
    """, unsafe_allow_html=True)
    
    # Question input
    question = st.text_input(
        "Ask a question",
        placeholder="What is web scraping used for?"
    )
    
    # Ask button
    if st.button("💬 Ask Question", type="primary", use_container_width=True):
        if not question.strip():
            st.error("❌ Enter a question")
            st.stop()
        
        with st.spinner("Getting answer..."):
            try:
                result = ask_question((st.session_state.qa_chain, st.session_state.retriever), question)
                answer = result['answer']
                relevant_chunks = result['relevant_chunks']
                
                # Add to history
                st.session_state.qa_history.append({
                    'question': question,
                    'answer': answer,
                    'relevant_chunks': relevant_chunks,
                    'timestamp': time.strftime("%H:%M:%S")
                })
                
                st.success("✅ Answer ready!")
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    
    # Show history
    if st.session_state.qa_history:
        st.markdown('<div class="step-header">📊 Question History</div>', unsafe_allow_html=True)
        
        for i, qa in enumerate(reversed(st.session_state.qa_history), 1):
            with st.expander(f"Q{i}: {qa['question']} ({qa['timestamp']})", expanded=False):
                st.write(f"**Answer:** {qa['answer']}")
                
                # Show relevant chunks
                if 'relevant_chunks' in qa and qa['relevant_chunks']:
                    st.markdown("**📚 Relevant Chunks Used:**")
                    for j, chunk in enumerate(qa['relevant_chunks'], 1):
                        with st.expander(f"📄 Chunk {j}", expanded=False):
                            # Show chunk content (truncated)
                            content = chunk.page_content
                            if len(content) > 500:
                                content = content[:500] + "..."
                            st.write(f"**Content:** {content}")
                            st.write(f"**Source:** {chunk.metadata.get('source', 'Unknown')}")
    
    # Clear history button
    if st.session_state.qa_history:
        if st.button("🗑️ Clear History"):
            st.session_state.qa_history = []
            st.rerun()
    
    # Show summary
    if st.session_state.summary:
        st.markdown('<div class="step-header">📝 Page Summary</div>', unsafe_allow_html=True)
        with st.expander("📄 View Summary", expanded=False):
            st.write(st.session_state.summary)
    
    # Start over button
    if st.button("🔄 Start Over (New URL)"):
        for key in ['step', 'vector_store', 'qa_chain', 'summary', 'chunks_count', 'processed_url', 'qa_history']:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div class="info-box">
    <strong>💡 How to use:</strong><br>
    1. Set your API key in .env file<br>
    2. Enter URL and create knowledge base<br>
    3. Ask questions and see answers<br>
    4. View question history below
</div>
""", unsafe_allow_html=True)
