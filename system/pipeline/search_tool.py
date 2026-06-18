"""
ArtDossier -- Web Search Tool (Sprint 3.1)
==========================================
DuckDuckGo for URL discovery, then httpx to fetch actual page content.
This gives the model real article text rather than ~200-char snippets.

Results are marked doc_type: "web_search" (low epistemic authority).
The model is instructed to treat web content as supplementary and lower
authority than the curated corpus.

Epistemic design decision:
    Web search supplements the corpus for facts not in the corpus
    (recent scholarship, auction records, exhibition history, niche details).
    It is always labelled lower authority than museum/scholarly catalogues.
"""

import re
import time
import httpx
from ddgs import DDGS

FETCH_TIMEOUT   = 8.0      # seconds per page fetch
MAX_PAGE_CHARS  = 2000     # chars extracted per page (reduced from 3000 — web is supplementary, not primary evidence)
FETCH_TOP_N     = 3        # how many pages to actually fetch (DDG can return more)
USER_AGENT      = "Mozilla/5.0 (compatible; ArtDossier/1.0)"


def _strip_html(html: str) -> str:
    """Extract readable text from HTML. No external dependency needed."""
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "nav", "header", "footer", "aside", "form"]):
            tag.decompose()
        text = soup.get_text(separator=" ", strip=True)
    except ImportError:
        # Fallback: crude regex strip if beautifulsoup4 not installed
        text = re.sub(r'<[^>]+>', ' ', html)

    text = re.sub(r'\s+', ' ', text).strip()
    return text


def _fetch_page(url: str) -> str:
    """Fetch a URL and return cleaned text, truncated to MAX_PAGE_CHARS."""
    try:
        resp = httpx.get(
            url,
            timeout=FETCH_TIMEOUT,
            headers={"User-Agent": USER_AGENT},
            follow_redirects=True,
        )
        if not resp.is_success:
            return ""
        content_type = resp.headers.get("content-type", "")
        if "html" not in content_type and "text" not in content_type:
            return ""   # skip PDFs, images, etc.
        return _strip_html(resp.text)[:MAX_PAGE_CHARS]
    except Exception:
        return ""


def search_web(query: str, max_results: int = 5) -> list[dict]:
    """
    Search DuckDuckGo, then fetch full page content for the top results.

    Returns list of dicts:
        text      -- extracted page text (up to MAX_PAGE_CHARS)
        title     -- page title from DDG
        url       -- source URL
        doc_type  -- always "web_search"
        authority -- always "low"
    """
    raw_results = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                raw_results.append({
                    "title": r.get("title", ""),
                    "url":   r.get("href", "") or r.get("link", ""),
                    "snippet": r.get("body", "") or r.get("snippet", ""),
                })
        time.sleep(0.4)
    except Exception as e:
        return [{
            "text":      f"Web search failed: {e}",
            "title":     "Search error",
            "url":       "",
            "doc_type":  "web_search",
            "authority": "low",
            "source":    "web_search",
            "section":   "web_result",
        }]

    results = []
    for i, r in enumerate(raw_results):
        if i < FETCH_TOP_N and r["url"]:
            # Fetch full content for top N results
            page_text = _fetch_page(r["url"])
            text = page_text if page_text else r["snippet"]
        else:
            # For the rest, use DDG snippet only
            text = r["snippet"]

        if not text:
            continue

        results.append({
            "text":      text,
            "title":     r["title"],
            "url":       r["url"],
            "doc_type":  "web_search",
            "authority": "low",
            "source":    "web_search",
            "section":   "web_result",
        })

    return results


def format_web_results(results: list[dict]) -> str:
    """Format web results for injection into the VLM prompt."""
    if not results:
        return "No web results."

    lines = ["WEB SEARCH RESULTS [epistemic authority: LOW]\n"]
    for i, r in enumerate(results, 1):
        lines.append(f"[WEB {i}] {r['title']}")
        if r["url"]:
            lines.append(f"  URL: {r['url']}")
        lines.append(f"  {r['text'][:MAX_PAGE_CHARS]}")
        lines.append("")
    return "\n".join(lines)


# -- Standalone test ------------------------------------------------------------
if __name__ == "__main__":
    print("Web search tool test — full page fetch")
    print("Query: 'Rembrandt Study of a Woman in a White Cap Leiden Collection'")
    results = search_web("Rembrandt Study of a Woman in a White Cap Leiden Collection", max_results=3)
    print(format_web_results(results))
    print(f"\nCh