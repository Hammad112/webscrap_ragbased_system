
from langchain_community.document_loaders import WebBaseLoader


def scrape_url(url: str) -> list:
    """
    Scrape a single URL and return the raw text as a list of Documents.

    A 'Document' is a LangChain object with two parts:
      - .page_content  → the text of the page
      - .metadata      → info like title and source URL

    Args:
        url: The webpage address to scrape.

    Returns:
        A list containing one Document object.

    """
    print(f" Scraping: {url}")

    loader = WebBaseLoader(url)
    docs = loader.load()

    print(f" Done! Got {len(docs[0].page_content):,} characters")
    return docs
