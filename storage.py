# ============================================================
# storage.py — Optional persistent storage for chunks
# ============================================================

import os
import pickle
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

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

def load_knowledge_base(save_dir: str = "knowledge_bases"):
    """
    Load knowledge base from disk.
    
    Args:
        save_dir: Directory containing saved knowledge base
        
    Returns:
        tuple: (vector_store, metadata) or (None, None) if not found
    """
    if not os.path.exists(f"{save_dir}/faiss_index"):
        return None, None
    
    try:
        # Load embeddings first
        from langchain_community.embeddings import HuggingFaceEmbeddings
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # Load vector store with embeddings
        vector_store = FAISS.load_local(f"{save_dir}/faiss_index", embeddings=embeddings)
        
        # Load metadata
        with open(f"{save_dir}/metadata.pkl", 'rb') as f:
            metadata = pickle.load(f)
        
        print(f"✅ Knowledge base loaded from {save_dir}")
        return vector_store, metadata
        
    except Exception as e:
        print(f"❌ Error loading knowledge base: {e}")
        return None, None

def list_saved_knowledge_bases(save_dir: str = "knowledge_bases"):
    """
    List all saved knowledge bases.
    
    Args:
        save_dir: Directory containing knowledge bases
        
    Returns:
        List of saved knowledge base names
    """
    if not os.path.exists(save_dir):
        return []
    
    return [d for d in os.listdir(save_dir) 
            if os.path.isdir(os.path.join(save_dir, d))]
