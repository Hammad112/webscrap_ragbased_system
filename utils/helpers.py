# ============================================================
# utils/helpers.py
# ============================================================


def print_header(title: str) -> None:
    """
    Print a nice section header to the console.

    Args:
        title: The title text to display.

    """
    print("\n" + "=" * 60)
    print(f"{title}")
    print("=" * 60)


def print_doc_info(docs: list) -> None:
    """
    Print basic info about a list of scraped Documents.

    Args:
        docs: List of Document objects from the scraper.

    Example:
        print_doc_info(docs)
        # Page info:
        #   Title  : Python (programming language) - Wikipedia
        #   Source : https://en.wikipedia.org/wiki/...
        #   Length : 90,568 characters
    """
    print("\n📄 Page info:")
    for i, doc in enumerate(docs):
        title  = doc.metadata.get("title",  "Unknown")
        source = doc.metadata.get("source", "Unknown")
        length = len(doc.page_content)
        print(f"  Doc {i + 1}:")
        print(f"    Title  : {title}")
        print(f"    Source : {source}")
        print(f"    Length : {length:,} characters")


def validate_api_key(key: str) -> bool:
    """
    Check that the API key looks like it has been filled in.
    Returns True if it looks valid, False if it's the placeholder.

    Args:
        key: The API key string from config.py.

    Returns:
        True if the key looks real, False otherwise.

    Example:
        if not validate_api_key(config.GROQ_API_KEY):
            print("Please update your API key in config.py!")
    """
    if not key or key == "paste_your_key_here":
        return False
    if len(key) < 10:
        return False
    return True


def show_links_table(links: list, n: int = 10) -> None:
    """
    Print the first n links in a neat table.

    Args:
        links: List of {'text': ..., 'href': ...} dicts from extract_links().
        n:     How many links to show (default: 10).

    Example:
        show_links_table(links, n=5)
    """
    print(f"\n🔗 First {n} links:")
    print(f"  {'Text':<40} → Href")
    print("  " + "-" * 70)
    for link in links[:n]:
        text = link["text"][:40]
        href = link["href"][:60]
        print(f"  {text:<40} → {href}")


def show_headings(headings: list, n: int = 15) -> None:
    """
    Print headings with indentation to show page structure.

    Args:
        headings: List of {'level': 'h1'/'h2'/'h3', 'text': ...} dicts.
        n:        How many headings to show (default: 15).

    Example:
        show_headings(headings)
    """
    indent = {"h1": "", "h2": "  ", "h3": "    "}
    print(f"\n📋 Page structure (first {n} headings):")
    for h in headings[:n]:
        pad = indent.get(h["level"], "")
        print(f"  {pad}[{h['level']}] {h['text']}")
