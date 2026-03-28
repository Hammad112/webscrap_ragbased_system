# ============================================================
# rag/rag_system.py
# ============================================================
#  3 STEPS:
#   1. EMBED  
#   2. STORE  
#   3. RETRIEVE & ANSWER 
# ============================================================

import os
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


def build_vector_store(chunks: list, embedding_model: str = "all-MiniLM-L6-v2") -> FAISS:
  
    # HuggingFaceEmbeddings downloads and runs the model locally
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

    # FAISS.from_documents embeds every chunk and stores them
    db = FAISS.from_documents(chunks, embeddings)

    return db


def build_qa_chain(db: FAISS, groq_api_key: str,
                   model: str = "llama-3.3-70b-versatile",
                   temperature: float = 0.0):

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


def summarize_page(docs: list, groq_api_key: str,
                   model: str = "llama-3.3-70b-versatile") -> str:

    os.environ["GROQ_API_KEY"] = groq_api_key
    llm = ChatGroq(model=model, temperature=0.3)

    # Use the first 3000 characters of the page as context
    context = docs[0].page_content[:3000]

    prompt = (
        "Please summarize the following text in 3-5 sentences. "
        "Be clear and concise.\n\n"
        f"Text:\n{context}"
    )

    response = llm.invoke(prompt)
    summary = response.content

    return summary
