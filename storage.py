# ============================================================
# storage.py — Optional persistent storage for chunks
# ============================================================

import os
import pickle
from langchain_community.vectorstores import FAISS

def save_knowledge_base(vector_store: FAISS, chunks: list, url: str, save_dir: str = "knowledge_bases"):
    """
    Save knowledge base to disk for persistence.
    
    Args:
        vector_store: FAISS vector store
        chunks: List of Document chunks
        url: Source URL
        save_dir: Directory to save files
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Save vector store
    vector_store.save_local(f"{save_dir}/faiss_index")
    
    # Save metadata
    metadata = {
        'url': url,
        'chunks_count': len(chunks),
        'chunk_texts': [doc.page_content for doc in chunks]
    }
    
    with open(f"{save_dir}/metadata.pkl", 'wb') as f:
        pickle.dump(metadata, f)
    
    print(f"✅ Knowledge base saved to {save_dir}")
    return save_dir
