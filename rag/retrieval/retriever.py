"""
ArtDossier -- RAG Retriever + Reranker (v3)
============================================
v3 changes over v2:
  - Hybrid retrieval: vector search + spaCy noun-phrase keyword search on the
    iconographic section of Panel 1, merged into a single candidate pool
  - RRF overlap bonus: chunks matched by multiple keyword queries get a mild
    score boost after cross-encoder reranking
  - Source diversity quota: min 2 per source type (Leiden / Rijksmuseum / WGA)
    if available above MIN_DIVERSITY_SCORE, then fill to TOTAL_CHUNKS
  - top_k_per_query raised 7 → 15; total final chunks raised 7 → 12

Setup (one-time, non-destructive):
  pip install spacy
  python -m spacy download en_core_web_sm
  python setup_qdrant_index.py   <- creates text payload indexes on Qdrant

All keyword / spaCy steps fail silently if not set up — pipeline degrades to
pure vector search, not a crash.

NOTE: ⚠️ Do NOT write artist names, painting titles, or iconographic answers
into Panel 1 or Panel 3 prompts. Keyword extraction reads what the VLM
actually produced — it does NOT give the model any hints. See MEMORY.md.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

_hf_token = os.getenv("HF_TOKEN", "").strip()
if _hf_token:
    os.environ["HF_TOKEN"] = _hf_token
    os.environ["HUGGING_FACE_HUB_TOKEN"] = _hf_token

QDRANT_URL      = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY  = os.getenv("QDRANT_API_KEY", None)
COLLECTION_NAME = "artdossier"
EMBED_MODEL     = "BAAI/bge-m3"
RERANK_MODEL    = "BAAI/bge-reranker-v2-m3"
_MODELS_DIR     = str(Path(__file__).resolve().parent.parent.parent / "models")

# ── Retrieval settings ────────────────────────────────────────────────────────
TOP_K_PER_QUERY     = 15    # vector candidates per query (3 queries → up to 45)
TOP_K_KW_PER_TERM   = 15    # keyword candidates per noun phrase (up to 8 phrases → ~30 unique new)
TOTAL_CHUNKS        = 12    # final chunks passed to Panel 2+3
MIN_PER_SOURCE      = 2     # guaranteed minimum per source type
MIN_DIVERSITY_SCORE = 0.30  # don't force low-quality chunks to fill quota
KW_BONUS_WEIGHT     = 0.08  # RRF bonus per keyword hit: score × (1 + 0.08 × min(hits,3))
KW_BONUS_CAP        = 3     # cap at 3 hits so one frequent term can't dominate

DOC_TYPE_LABELS = {
    "scholarly_catalogue":     "Leiden Collection [scholarly catalogue]",
    "institutional_catalogue": "Rijksmuseum [institutional catalogue]",
    "institutional_data":      "Metropolitan Museum of Art [institutional data]",
    "web_catalogue":           "Web Gallery of Art [web catalogue]",
    "web_search":              "Web search [unverified]",
}

DOC_TYPE_AUTHORITY = {
    "scholarly_catalogue":     "high",
    "institutional_catalogue": "high",
    "institutional_data":      "medium",
    "web_catalogue":           "medium",
    "web_search":              "low",
}

RETRIEVAL_DOC_TYPES = ["scholarly_catalogue", "institutional_catalogue", "web_catalogue"]

_embedder = None
_reranker = None
_qdrant   = None
_nlp      = None   # spaCy model, loaded once


# ── Model loading ─────────────────────────────────────────────────────────────

def _get_embedder():
    global _embedder
    if _embedder is None:
        from transformers import AutoTokenizer, AutoModel
        tok   = AutoTokenizer.from_pretrained(EMBED_MODEL, cache_dir=_MODELS_DIR)
        model = AutoModel.from_pretrained(EMBED_MODEL, cache_dir=_MODELS_DIR)
        model.eval()
        _embedder = (tok, model)
    return _embedder


def _get_reranker():
    """
    Transformers-native cross-encoder. Do NOT use sentence_transformers —
    CrossEncoder triggers os._exit() on this stack (torch 2.5.1+cu121 / transformers 4.50.0).
    sigmoid(logit) reproduces CrossEncoder scores exactly.
    """
    global _reranker
    if _reranker is None:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        tok   = AutoTokenizer.from_pretrained(RERANK_MODEL, cache_dir=_MODELS_DIR)
        model = AutoModelForSequenceClassification.from_pretrained(RERANK_MODEL, cache_dir=_MODELS_DIR)
        model.eval()
        _reranker = (tok, model)
    return _reranker


def _get_nlp():
    """Load spaCy en_core_web_sm once. Returns None silently if not installed."""
    global _nlp
    if _nlp is None:
        try:
            import spacy
            _nlp = spacy.load("en_core_web_sm")
        except (ImportError, OSError):
            _nlp = False   # sentinel: tried and failed, don't retry
    return _nlp if _nlp is not False else None


def _get_qdrant():
    global _qdrant
    if _qdrant is None:
        from qdrant_client import QdrantClient
        kwargs = {"url": QDRANT_URL}
        if QDRANT_API_KEY:
            kwargs["api_key"] = QDRANT_API_KEY
        _qdrant = QdrantClient(**kwargs)
    return _qdrant


def _rerank(query: str, texts: list[str], batch_size: int = 16) -> list[float]:
    """Score [query, text] pairs with cross-encoder. Returns sigmoid relevance scores."""
    import torch
    tok, model = _get_reranker()
    scores: list[float] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = tok(
            [query] * len(batch), batch,
            padding=True, truncation=True, max_length=512, return_tensors="pt",
        )
        with torch.no_grad():
            logits = model(**inputs).logits.view(-1).float()
            scores.extend(torch.sigmoid(logits).tolist())
    return scores


# ── Keyword extraction ─────────────────────────────────────────────────────────

# Words too generic to be useful keywords for corpus matching
_KW_STOPWORDS = {
    "painting", "scene", "figure", "figures", "image", "work", "artist",
    "style", "period", "century", "composition", "background", "foreground",
    "light", "dark", "color", "colour", "form", "lines", "surface", "technique",
    "panel", "canvas", "oil", "medium", "overall", "likely", "possibly",
    "however", "therefore", "although", "perhaps", "rather", "overall",
    "genre", "subject", "theme", "themes", "element", "elements", "type",
}


def _extract_keywords(iconographic_text: str) -> list[str]:
    """
    Extract distinctive noun phrases and proper nouns from the iconographic
    section of Panel 1 for keyword corpus search.

    Uses spaCy en_core_web_sm if installed. Falls back to regex for
    capitalized phrases (proper nouns / iconographic labels).

    Returns up to 8 phrases, sorted by specificity (longer = more specific first).
    Does NOT read any ground-truth about the painting — only what the VLM produced.
    """
    text = iconographic_text[:1200]   # cap to keep it fast

    nlp = _get_nlp()

    if nlp is not None:
        doc = nlp(text)

        candidates = []

        # Noun chunks: prefer multi-word phrases
        for chunk in doc.noun_chunks:
            phrase = chunk.text.strip().lower()
            words  = phrase.split()
            # Keep if: multi-word and not all stopwords, OR single specific word
            if len(words) >= 2 and not all(w in _KW_STOPWORDS for w in words):
                candidates.append(phrase)
            elif len(words) == 1 and len(phrase) > 5 and phrase not in _KW_STOPWORDS:
                candidates.append(phrase)

        # Proper nouns separately (capitalized names / terms in original)
        for token in doc:
            if token.pos_ == "PROPN" and len(token.text) > 3:
                candidates.append(token.text.lower())

    else:
        # Regex fallback: capitalized multi-word phrases + longer capitalized singles
        import re
        multi  = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b', text)
        single = re.findall(r'\b[A-Z][a-z]{5,}\b', text)
        candidates = [p.lower() for p in multi + single]

    # Deduplicate, filter stopwords, sort longest first (more specific)
    seen, unique = set(), []
    for kw in candidates:
        kw = kw.strip()
        if kw and kw not in seen and kw not in _KW_STOPWORDS:
            seen.add(kw)
            unique.append(kw)

    unique.sort(key=len, reverse=True)
    return unique[:8]


# ── Search ─────────────────────────────────────────────────────────────────────

def _vec_search(query: str, top_k: int, allowed_doc_types: list = None) -> list:
    """Embed query and search Qdrant. MET (institutional_data) excluded by default.
    Pass allowed_doc_types to restrict further (e.g. expert mode: Leiden + Rijks only)."""
    import torch
    import torch.nn.functional as F
    tokenizer, model = _get_embedder()
    inputs = tokenizer(query[:512], return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        out = model(**inputs)
    vec = F.normalize(out.last_hidden_state[:, 0, :], p=2, dim=1)
    q_vec = vec[0].tolist()
    qc = _get_qdrant()
    doc_types = allowed_doc_types if allowed_doc_types is not None else RETRIEVAL_DOC_TYPES
    from qdrant_client.models import Filter, FieldCondition, MatchAny
    return qc.query_points(
        collection_name=COLLECTION_NAME,
        query=q_vec,
        limit=top_k,
        with_payload=True,
        score_threshold=0.0,
        query_filter=Filter(must=[FieldCondition(
            key="doc_type",
            match=MatchAny(any=doc_types),
        )]),
    ).points


def _keyword_pool(keywords: list[str], top_k_per_kw: int = TOP_K_KW_PER_TERM,
                  allowed_doc_types: list = None) -> dict:
    """
    For each keyword, scroll Qdrant for matching chunks (text or title fields).
    Returns {point_id: (point, hit_count)} — hit_count = how many distinct keywords
    matched this chunk. Used later for the RRF bonus.
    Falls back silently if text indexes not created (run setup_qdrant_index.py).
    Pass allowed_doc_types to restrict sources (e.g. expert mode).
    """
    if not keywords:
        return {}

    qc = _get_qdrant()
    doc_types = allowed_doc_types if allowed_doc_types is not None else RETRIEVAL_DOC_TYPES
    from qdrant_client.models import Filter, FieldCondition, MatchText, MatchAny

    hits: dict = {}   # id -> (point, count)

    for kw in keywords:
        if not kw or len(kw) < 3:
            continue
        try:
            pts, _ = qc.scroll(
                collection_name=COLLECTION_NAME,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(key="doc_type", match=MatchAny(any=doc_types)),
                    ],
                    should=[
                        FieldCondition(key="text",  match=MatchText(text=kw)),
                        FieldCondition(key="title", match=MatchText(text=kw)),
                    ],
                ),
                with_payload=True,
                limit=top_k_per_kw,
            )
            for p in pts:
                if p.id in hits:
                    hits[p.id] = (hits[p.id][0], hits[p.id][1] + 1)
                else:
                    hits[p.id] = (p, 1)
        except Exception:
            pass   # index not created — silent fallback to vector-only

    return hits


# ── Post-processing ─────────────────────────────────────────────────────────────

def _apply_rrf_bonus(
    ranked: list[tuple],   # (rerank_score, point)
    kw_hits: dict,         # {id: (point, hit_count)}
) -> list[tuple]:
    """Apply keyword overlap bonus: score × (1 + 0.08 × min(hits, 3))."""
    result = []
    for score, p in ranked:
        hit_count = kw_hits.get(p.id, (None, 0))[1]
        bonus     = 1.0 + KW_BONUS_WEIGHT * min(hit_count, KW_BONUS_CAP)
        result.append((score * bonus, p))
    result.sort(key=lambda x: x[0], reverse=True)
    return result


def _apply_diversity(
    ranked: list[tuple],
    min_per_source: int = MIN_PER_SOURCE,
    total: int = TOTAL_CHUNKS,
    score_floor: float = MIN_DIVERSITY_SCORE,
) -> list[tuple]:
    """
    Guarantee min_per_source chunks from each source if score >= score_floor,
    then fill remaining slots with best-ranked regardless of source.
    """
    from collections import defaultdict

    by_type: dict = defaultdict(list)
    for s, p in ranked:
        dt = p.payload.get("doc_type", "")
        by_type[dt].append((s, p))

    chosen: list[tuple] = []
    chosen_ids: set = set()

    # Quota pass
    for dt in ["scholarly_catalogue", "institutional_catalogue", "web_catalogue"]:
        quota = 0
        for s, p in by_type[dt]:
            if quota >= min_per_source:
                break
            if p.id not in chosen_ids and s >= score_floor:
                chosen.append((s, p))
                chosen_ids.add(p.id)
                quota += 1

    # Fill pass — best remaining
    for s, p in ranked:
        if len(chosen) >= total:
            break
        if p.id not in chosen_ids:
            chosen.append((s, p))
            chosen_ids.add(p.id)

    chosen.sort(key=lambda x: x[0], reverse=True)
    return chosen[:total]


def _fetch_full_doc(doc_id: str) -> list[dict]:
    """Fetch all chunks from one doc_id, ordered by chunk_index."""
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    qc = _get_qdrant()
    points, _ = qc.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=Filter(must=[
            FieldCondition(key="doc_id", match=MatchValue(value=doc_id))
        ]),
        with_payload=True,
        limit=200,
    )
    points.sort(key=lambda p: p.payload.get("chunk_index", 0))
    return [_point_to_chunk(p, score=1.0) for p in points]


def _point_to_chunk(p, score: float) -> dict:
    return {
        "text":       p.payload.get("text", ""),
        "score":      score,
        "doc_id":     p.payload.get("doc_id", ""),
        "source":     p.payload.get("source", ""),
        "doc_type":   p.payload.get("doc_type", "institutional_data"),
        "authority":  DOC_TYPE_AUTHORITY.get(p.payload.get("doc_type", ""), "medium"),
        "section":    p.payload.get("section", ""),
        "title":      p.payload.get("title", ""),
        "artist":     p.payload.get("artist", ""),
        "date":       p.payload.get("date", ""),
        "object_url": p.payload.get("object_url", ""),
    }


def _apply_expansion(top_chunks: list[dict]) -> list[dict]:
    """Full-doc expansion if top chunk clears score threshold."""
    if not top_chunks:
        return top_chunks

    top_score  = top_chunks[0]["score"]
    top_doc_id = top_chunks[0]["doc_id"]

    EXPAND_MIN_SCORE = 0.65
    EXPAND_MIN_GAP   = 0.15

    second   = top_chunks[1] if len(top_chunks) > 1 else None
    same_doc = second and second["doc_id"] == top_doc_id
    gap_ok   = same_doc or (second is None) or (top_score - second["score"] > EXPAND_MIN_GAP)

    if top_score > EXPAND_MIN_SCORE and gap_ok and top_doc_id:
        full_doc   = _fetch_full_doc(top_doc_id)
        seen_texts = {c["text"] for c in full_doc}
        extras     = [c for c in top_chunks[1:] if c["text"] not in seen_texts]
        return full_doc + extras

    return top_chunks


# ── Public API ────────────────────────────────────────────────────────────────

def retrieve_multi(
    queries: list[str],
    top_k_per_query: int = TOP_K_PER_QUERY,
    top_k_rerank: int = TOTAL_CHUNKS,
    expand_top_doc: bool = True,
    allowed_doc_types: list = None,
) -> list[dict]:
    """allowed_doc_types: restrict retrieval to specific sources.
    None (default) = Leiden + Rijksmuseum + WGA (MET always excluded).
    Expert mode: pass ["scholarly_catalogue", "institutional_catalogue"] to exclude WGA."""
    """
    Hybrid multi-query retrieval with RRF keyword bonus and source diversity.

    queries[0] = visual section
    queries[1] = iconographic section  ← keyword extraction runs here
    queries[2] = synthesis section     ← cross-encoder reranking anchor

    Steps:
      1. Vector search: 3 queries × 15 = up to 45 unique candidates
      2. Keyword extraction: spaCy noun phrases from iconographic section (queries[1])
      3. Keyword pool: up to 8 phrases × 15 Qdrant text matches = ~30 new candidates
      4. Merge everything, deduplicate
      5. Cross-encoder reranks full pool using synthesis as anchor
      6. RRF bonus: chunks matching multiple keywords get score × (1 + 0.08 × hits)
      7. Source diversity: min 2 per source type, fill to 12 total
      8. Full-doc expansion on top chunk if score > 0.65
    """
    if not queries:
        return []

    seen_ids: set  = set()
    all_candidates: list = []

    # Step 1 — vector search
    for q in queries:
        if not q.strip():
            continue
        pts = _vec_search(q, top_k_per_query, allowed_doc_types=allowed_doc_types)
        for p in pts:
            if p.id not in seen_ids:
                seen_ids.add(p.id)
                all_candidates.append(p)

    # Steps 2+3 — keyword search on iconographic section
    iconographic_q = queries[1] if len(queries) > 1 else ""
    keywords  = _extract_keywords(iconographic_q) if iconographic_q.strip() else []
    kw_hits   = _keyword_pool(keywords, allowed_doc_types=allowed_doc_types) if keywords else {}

    # Merge keyword-only candidates into pool
    for pid, (p, _count) in kw_hits.items():
        if pid not in seen_ids:
            seen_ids.add(pid)
            all_candidates.append(p)

    if not all_candidates:
        return []

    # Step 5 — rerank full merged pool (synthesis as anchor)
    anchor_query = queries[-1] if queries else ""

    # Rerank the full merged pool for relevance scores
    scores = _rerank(anchor_query, [p.payload.get("text", "") for p in all_candidates])
    ranked = sorted(zip(scores, all_candidates), key=lambda x: x[0], reverse=True)

    # Step 6 — Separate lanes: relevance top-N + keyword rescue slots
    # Keyword candidates can only ADD, never displace a high-relevance chunk.
    KEYWORD_RESCUE_SLOTS = 3   # guaranteed spots for keyword-found entries
    relevance_n = top_k_rerank - KEYWORD_RESCUE_SLOTS   # e.g. 12-3=9

    chosen_ids: set = set()
    final_ranked: list = []

    # Lane 1: top relevance_n by cross-encoder score (source diversity applied)
    diverse_9 = _apply_diversity(ranked, total=relevance_n)
    for s, p in diverse_9:
        final_ranked.append((s, p))
        chosen_ids.add(p.id)

    # Lane 2: up to KEYWORD_RESCUE_SLOTS from keyword hits, ordered by hit count
    # then cross-encoder score — picks the most keyword-supported chunks not already chosen
    kw_scored = sorted(
        [(kw_hits[p.id][1], score, p)
         for score, p in ranked if p.id in kw_hits and p.id not in chosen_ids],
        key=lambda x: (x[0], x[1]), reverse=True,
    )
    for _hits, score, p in kw_scored[:KEYWORD_RESCUE_SLOTS]:
        final_ranked.append((score, p))
        chosen_ids.add(p.id)

    final_ranked.sort(key=lambda x: x[0], reverse=True)
    top_chunks = [_point_to_chunk(p, float(s)) for s, p in final_ranked]

    # Step 8 — full-doc expansion
    if not expand_top_doc:
        return top_chunks
    return _apply_expansion(top_chunks)


def retrieve(
    query: str,
    top_k_retrieve: int = 20,
    top_k_rerank: int = TOTAL_CHUNKS,
    expand_top_doc: bool = True,
) -> list[dict]:
    """Single-query retrieval (used for direct calls / testing)."""
    results = _vec_search(query, top_k_retrieve)
    if not results:
        return []
    scores  = _rerank(query, [r.payload.get("text", "") for r in results])
    ranked  = sorted(zip(scores, results), key=lambda x: x[0], reverse=True)
    diverse = _apply_diversity(ranked, total=top_k_rerank)
    top_chunks = [_point_to_chunk(p, float(s)) for s, p in diverse]
    if not expand_top_doc:
        return top_chunks
    return _apply_expansion(top_chunks)


def format_for_prompt(chunks: list[dict], include_authority: bool = True) -> str:
    if not chunks:
        return "No relevant sources retrieved."

    parts = []
    for i, c in enumerate(chunks, 1):
        label     = DOC_TYPE_LABELS.get(c["doc_type"], c["doc_type"])
        authority = c.get("authority", "medium")
        header    = f"[SOURCE {i}] {label}"
        if c["title"]:    header += f" -- {c['title']}"
        if c["artist"]:   header += f" ({c