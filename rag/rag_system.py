# ============================================================
# rag/rag_system.py
# ============================================================
#  3 STEPS:
#   1. EMBED  — Convert each chunk of text into a vector (a list
#               of numbers). Similar chunks get similar vectors.
#
#   2. STORE  — Put all vectors in FAISS, a fast search database.
#               Think of it as a library where similar books sit
#               next to each other.
#
#   3. RETRIEVE & ANSWER — When you ask a question:
#               a) Convert the question to a vector
#               b) Find the closest chunks in FAISS
#               c) Give those chunks + your question to the LLM
#               d) LLM reads them and writes an answer
# ============================================================

import os
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


def build_vector_store(chunks: list, embedding_model: str = "all-MiniLM-L6-v2") -> FAISS:
    """
    Convert text chunks into vectors and store them in FAISS.

    all-MiniLM-L6-v2 is a small, fast model that:
      - Runs on your computer (no API key needed, it's FREE)
      - Turns a sentence into a 384-number vector
      - Is good enough for most Q&A tasks

    Args:
        chunks:          List of Document chunks from the cleaner.
        embedding_model: Name of the HuggingFace model to use.

    Returns:
        A FAISS vector store you can search.

    """
    

    # HuggingFaceEmbeddings downloads and runs the model locally
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

    # FAISS.from_documents embeds every chunk and stores them
    db = FAISS.from_documents(chunks, embeddings)

    print(f"Vector store ready! Indexed {len(chunks)} chunks")
    return db


def build_qa_chain(db: FAISS, groq_api_key: str,
                   model: str = "llama-3.3-70b-versatile",
                   temperature: float = 0.0):
    """
    Build a Q&A chain that connects FAISS to the Groq LLM using modern LCEL.

    Args:
        db:           The FAISS vector store from build_vector_store().
        groq_api_key: Your Groq API key (from config.py).
        model:        Which Groq model to use.
        temperature:  0 = focused, 1 = creative.

    Returns:
        A Q&A chain ready to answer questions.

    """
    print(f"🤖 Loading LLM: {model}")

    # Set the API key so LangChain can find it
    os.environ["GROQ_API_KEY"] = groq_api_key

    # ChatGroq connects to the Groq API (fast, free tier available)
    llm = ChatGroq(model=model, temperature=temperature)

    # Create a prompt template for Q&A
    template = """Use following pieces of context to answer question at the end. 
    If you don't know the answer from the context, just say that you don't know. 
    Keep the answer concise and based on the provided context.

    Context: {context}

    Question: {question}

    Helpful Answer:"""

    prompt = PromptTemplate.from_template(template)

    # Create retriever
    retriever = db.as_retriever(search_kwargs={"k": 3})  # Get top 3 chunks

    # Build RAG chain using LCEL
    qa_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # Return both chain and retriever
    return qa_chain, retriever


def ask_question(qa_chain_with_retriever, question: str) -> str:
    """
    Ask a single question and get back the AI's answer as a string.

    Args:
        qa_chain_with_retriever: Tuple of (qa_chain, retriever) from build_qa_chain().
        question: Any question about the scraped page.

    Returns:
        The AI's answer as a plain string.

    """
    qa_chain, retriever = qa_chain_with_retriever
    
    # Retrievers expose the Runnable API: use invoke(query), not get_relevant_documents
    relevant_docs = retriever.invoke(question)

    
    # Get the answer
    result = qa_chain.invoke(question)
    
    # Return both answer and relevant docs
    return {
        'answer': result,
        'relevant_chunks': relevant_docs
    }


def ask_multiple_questions(qa_chain, questions: list) -> dict:
    """
    Ask several questions at once. Prints them nicely and returns
    a dictionary of {question: answer}.

    Args:
        qa_chain:  The Q&A chain from build_qa_chain().
        questions: A list of question strings.

    Returns:
        Dict mapping each question to its answer.

    """
    print(f"\n💬 Asking {len(questions)} question(s)...\n")
    print("=" * 60)

    results = {}
    for i, question in enumerate(questions, start=1):
        print(f"\n❓ Question {i}: {question}")
        answer = ask_question(qa_chain, question)
        results[question] = answer
        print(f"💬 Answer:\n{answer}")
        print("-" * 60)

    return results


def summarize_page(docs: list, groq_api_key: str,
                   model: str = "llama-3.3-70b-versatile") -> str:
    """
    Summarize the scraped page in a few sentences using Groq.

    This does NOT use RAG — it just sends the first chunk to the LLM
    and asks for a summary. Simple and fast.

    Args:
        docs:         List of Document objects from the scraper.
        groq_api_key: Your Groq API key.
        model:        Which Groq model to use.

    Returns:
        A summary string.

    """
    os.environ["GROQ_API_KEY"] = groq_api_key
    llm = ChatGroq(model=model, temperature=0.3)

    # Use the first 3000 characters of the page as context
    context = docs[0].page_content[:3000]

    prompt = (
        "Please summarize the following text in 3-5 sentences. "
        "Be clear and concise.\n\n"
        f"Text:\n{context}"
    )

    print("📝 Summarizing page...")
    response = llm.invoke(prompt)
    summary = response.content

    print("✅ Summary ready!")
    return summary
