
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter


def clean_text(text: str) -> str:

    # Replace tabs with a space
    text = text.replace("\t", " ")

    # Collapse 3+ blank lines into just 2
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove lines that are only spaces
    lines = [line for line in text.split("\n") if line.strip()]
    text = "\n".join(lines)

    return text.strip()


def split_documents(docs: list, chunk_size: int = 1000, chunk_overlap: int = 100) -> list:
 
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    chunks = splitter.split_documents(docs)
    print(f"✅ Created {len(chunks)} chunks from {len(docs)} document(s)")
    return chunks


def clean_and_split(docs: list, chunk_size: int = 1000, chunk_overlap: int = 100) -> list:

    # Clean the text inside each Document object
    for doc in docs:
        doc.page_content = clean_text(doc.page_content)

    print(f"✅ Cleaned {len(docs)} document(s)")

    # Now split into chunks
    return split_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)


