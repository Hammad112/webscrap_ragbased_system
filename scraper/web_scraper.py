
import requests
from bs4 import BeautifulSoup
from langchain_community.document_loaders import WebBaseLoader
import bs4


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


def scrape_url_filtered(url: str, tag: str = "div", attrs: dict = None) -> list:
    """
    Scrape a URL but only keep the main content (ignore menus, footers, etc.).

    SoupStrainer tells BeautifulSoup: 'only parse THIS part of the page'.
    This is useful for Wikipedia where most of the HTML is navigation junk.

    Args:
        url:   The webpage address to scrape.
        tag:   The HTML tag to keep (default: 'div').
        attrs: A dictionary of HTML attributes to match.
               Example: {'id': 'mw-content-text'} targets Wikipedia's main content.
    Returns:
        A list containing one filtered Document object.

    """
    if attrs is None:
        attrs = {}

    print(f"Scraping (filtered): {url}")

    # SoupStrainer: only grab the HTML that matches our tag + attrs
    strainer = bs4.SoupStrainer(name=tag, attrs=attrs)

    loader = WebBaseLoader(
        web_paths=(url,),
        bs_kwargs={"parse_only": strainer}
    )
    docs = loader.load()

    print(f"Done! Got {len(docs[0].page_content):,} characters (filtered)")
    return docs


def extract_links(url: str) -> list:
    """
    Get all the hyperlinks from a webpage.

    Args:
        url: The webpage address.

    Returns:
        A list of dicts, each with 'text' and 'href' keys.
        Example: [{'text': 'Main page', 'href': '/wiki/Main_Page'}, ...]

    """
    print(f"Extracting links from: {url}")

    # requests.get downloads the raw HTML
    response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})

    # BeautifulSoup parses the HTML into a tree we can search
    soup = BeautifulSoup(response.text, "lxml")

    links = []
    for tag in soup.find_all("a", href=True):
        text = tag.get_text(strip=True)
        href = tag["href"]
        if text:  # skip links with no visible text
            links.append({"text": text, "href": href})

    print(f"Found {len(links)} links")
    return links


def extract_headings(url: str) -> list:
    """
    Get all the headings (h1, h2, h3) from a webpage.
    This shows you the page's structure like a table of contents.

    Args:
        url: The webpage address.

    Returns:
        A list of dicts with 'level' and 'text' keys.
        Example: [{'level': 'h1', 'text': 'Python (programming language)'}, ...]
    """
    print(f"Extracting headings from: {url}")

    response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    soup = BeautifulSoup(response.text, "lxml")

    headings = []
    for tag in soup.find_all(["h1", "h2", "h3"]):
        text = tag.get_text(strip=True)
        if text:
            headings.append({"level": tag.name, "text": text})

    print(f"Found {len(headings)} headings")
    return headings
