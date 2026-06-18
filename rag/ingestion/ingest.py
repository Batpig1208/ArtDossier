"""
ArtDossier — RAG Ingestion (Sprint 2.2)
========================================
Chunks all 489 normalized documents, embeds with BGE-M3,
and stores in Qdrant.

Prerequisites:
    1. Qdrant running:
       docker run -p 6333:6333 -v qdrant_storage:/qdrant/storage qdrant/qdrant
    2. pip install -r requirements.txt  (already done)

Run:
    cd "/path/to/artdossier"
    python rag/ingestion/ingest.py

Re-running is safe — collection is recreated from scratch each time.
To add documents only (skip existing), pass --incremental flag.
"""

import json
import os
import sys
import glob
import argparse
import time

# ── Config ─────────────────────────────────────────────────────────────────────
PROCESSED_DIR   = os.path.join(os.path.dirname(__file__), "..", "..", "corpus", "processed")
COLLECTION_NAME = "artdossier"
EMBED_MODEL     = "BAAI/bge-m3"          # 1024-dim, multilingual, best for art text
EMBED_DIM       = 1024
CHUNK_SIZE      = 512                    # tokens
CHUNK_OVERLAP   = 64                     # tokens
MIN_CHUNK_CHARS = 50                     # skip chunks shorter than this
BATCH_SIZE      = 32                     # embedding batch size

from dotenv import load_dotenv
load_dotenv()

# ── GPU detection ──────────────────────────────────────────────────────────────
def check_gpu():
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"  GPU detected: {name} ({vram:.1f} GB VRAM)")
            print(f"  BGE-M3 will use CUDA automatically via use_fp16=True")
            return True
        else:
            print("  No CUDA GPU detected — running on CPU (~60-70 min)")
            print("  To enable GPU: ensure CUDA toolkit + torch[cuda] are installed")
            print("  Quick check: python -c \"import torch; print(torch.cuda.is_available())\"")
            return False
    except ImportError:
        print("  torch not installed — GPU check skipped")
        return False

QDRANT_URL     = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)


# ── Chunking ───────────────────────────────────────────────────────────────────
def chunk_document(doc: dict) -> list[dict]:
    """
    Split a document into chunks by section.

    Chunking strategy:
    - paragraph_separator="\n\n" means the splitter tries paragraph boundaries FIRST.
      Only if a paragraph exceeds chunk_size does it fall back to sentence boundaries,
      then word boundaries. This preserves academic paragraph arguments intact.
    - Relevant mainly for Leiden essays (long, multi-paragraph academic text).
    - Rijksmuseum descriptions are short enough to always be a single chunk.
    - MET entries are always single chunks.
    """
    from llama_index.core.node_parser import SentenceSplitter

    splitter = SentenceSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        paragraph_separator="\n\n",          # respect paragraph boundaries first
        secondary_chunking_regex="[.!?]",    # fallback: split at sentence endings
    )
    chunks = []

    sections = doc.get("text_sections", {})
    if not sections:
        return []

    # Process each section independently
    for section_name, section_text in sections.items():
        if not section_text or len(section_text.strip()) < MIN_CHUNK_CHARS:
            continue

        # Split this section
        raw_chunks = splitter.split_text(section_text)

        for i, chunk_text in enumerate(raw_chunks):
            if len(chunk_text.strip()) < MIN_CHUNK_CHARS:
                continue

            chunks.append({
                "text": chunk_text.strip(),
                "metadata": {
                    # Retrieval identity
                    "doc_id":       doc["doc_id"],
                    "source":       doc["source"],
                    "doc_type":     doc.get("doc_type", "institutional_data"),
                    "section":      section_name,
                    "chunk_index":  i,
                    "total_chunks": len(raw_chunks),
                    # Display / filtering
                    "title":        doc.get("title", ""),
                    "artist":       doc.get("artist", ""),
                    "date":         doc.get("date", ""),
                    "medium":       doc.get("medium", ""),
                    "object_url":   doc.get("object_url", ""),
                    "inventory_id": doc.get("inventory_id", ""),
                    # Flags
                    "is_leiden":    doc["source"] == "leiden_collection",
                    "is_rijksmuseum": doc["source"] == "rijksmuseum",
                    "is_met":       doc["source"] == "met",
                }
            })

    return chunks


# ── Embedding ──────────────────────────────────────────────────────────────────
def load_embedder():
    print(f"  Loading BGE-M3 embedder ({EMBED_MODEL})...")
    import torch
    from sentence_transformers import SentenceTransformer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")
    model = SentenceTransformer(EMBED_MODEL, device=device)
    print("  Embedder ready")
    return model


def embed_batch(model, texts: list[str]) -> list[list[float]]:
    vecs = model.encode(texts, batch_size=BATCH_SIZE,
                        normalize_embeddings=True, show_progress_bar=False)
    return vecs.tolist()


# ── Qdrant setup ───────────────────────────────────────────────────────────────
def setup_qdrant(recreate: bool = True):
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams

    print(f"  Connecting to Qdrant at {QDRANT_URL}...")
    kwargs = {"url": QDRANT_URL}
    if QDRANT_API_KEY:
        kwargs["api_key"] = QDRANT_API_KEY
    qc = QdrantClient(**kwargs)

    # Verify connection
    try:
        qc.get_collections()
        print("  ✓ Qdrant connected")
    except Exception as e:
        print(f"  ✗ Cannot reach Qdrant: {e}")
        print("    Start it with: docker run -p 6333:6333 qdrant/qdrant")
        sys.exit(1)

    if recreate:
        # Drop + recreate collection
        existing = [c.name for c in qc.get_collections().collections]
        if COLLECTION_NAME in existing:
            qc.delete_collection(COLLECTION_NAME)
            print(f"  Dropped existing collection '{COLLECTION_NAME}'")

        qc.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
        )
        print(f"  ✓ Created collection '{COLLECTION_NAME}' (dim={EMBED_DIM}, cosine)")
    else:
        existing = [c.name for c in qc.get_collections().collections]
        if COLLECTION_NAME not in existing:
            qc.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
            )
            print(f"  ✓ Created collection '{COLLECTION_NAME}'")
        else:
            print(f"  ✓ Using existing collection '{COLLECTION_NAME}'")

    return qc


def upsert_chunks(qc, all_chunks: list[dict], vectors: list[list[float]]):
    """Upsert all chunks + vectors into Qdrant in batches."""
    from qdrant_client.models import PointStruct

    UPSERT_BATCH = 256
    total = len(all_chunks)

    for start in range(0, total, UPSERT_BATCH):
        end   = min(start + UPSERT_BATCH, total)
        batch_chunks  = all_chunks[start:end]
        batch_vectors = vectors[start:end]

        points = [
            PointStruct(
                id=start + i,
                vector=vec,
                payload={
                    "text": chunk["text"],
                    **chunk["metadata"],
                }
            )
            for i, (chunk, vec) in enumerate(zip(batch_chunks, batch_vectors))
        ]
        qc.upsert(collection_name=COLLECTION_NAME, points=points)

    print(f"  ✓ Upserted {total} chunks into Qdrant")


# ── Main ────────────────────────────────────────────────────────────────────────
def main(incremental: bool = False):
    print("=" * 60)
    print("ArtDossier — RAG Ingestion")
    print("=" * 60)

    # ── GPU check ─────────────────────────────────────────────────────────────
    print("\n[0/4] GPU check...")
    check_gpu()

    # ── Load processed docs ───────────────────────────────────────────────────
    print("\n[1/4] Loading processed documents...")
    doc_files = sorted(glob.glob(os.path.join(PROCESSED_DIR, "*.json")))
    doc_files = [f for f in doc_files if not os.path.basename(f).startswith("_")]
    docs = [json.load(open(f, encoding="utf-8")) for f in doc_files]
    print(f"  Loaded {len(docs)} documents")

    # ── Chunk ─────────────────────────────────────────────────────────────────
    print("\n[2/4] Chunking by section...")
    all_chunks = []
    source_stats = {}

    for doc in docs:
        chunks = chunk_document(doc)
        all_chunks.extend(chunks)
        src = doc["source"]
        source_stats[src] = source_stats.get(src, {"docs": 0, "chunks": 0})
        source_stats[src]["docs"]   += 1
        source_stats[src]["chunks"] += len(chunks)

    print(f"  Total chunks: {len(all_chunks)}")
    for src, s in source_stats.items():
        avg = s["chunks"] // max(s["docs"], 1)
        print(f"    {src:<30} {s['chunks']:>4} chunks from {s['docs']:>3} docs (avg {avg}/doc)")

    if not all_chunks:
        print("  No chunks produced — check corpus/processed/ files.")
        sys.exit(1)

    # ── Embed ─────────────────────────────────────────────────────────────────
    print(f"\n[3/4] Embedding {len(all_chunks)} chunks with BGE-M3...")
    embedder = load_embedder()

    texts = [c["text"] for c in all_chunks]
    vectors = []
    t0 = time.time()

    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start:start + BATCH_SIZE]
        batch_vecs = embed_batch(embedder, batch)
        vectors.extend(batch_vecs)
        done = min(start + BATCH_SIZE, len(texts))
        elapsed = time.time() - t0
        rate = done / elapsed if elapsed > 0 else 0
        eta  = (len(texts) - done) / rate if rate > 0 else 0
        print(f"  {done:>4}/{len(texts)}  ({rate:.0f} chunks/s, ETA {eta:.0f}s)", end="\r")

    print(f"  ✓ Embedded {len(vectors)} chunks in {time.time()-t0:.1f}s        ")

    # ── Ingest into Qdrant ────────────────────────────────────────────────────
    print(f"\n[4/4] Ingesting into Qdrant...")
    qc = setup_qdrant(recreate=not incremental)
    upsert_chunks(qc, all_chunks, vectors)

    # Verify
    info = qc.get_collection(COLLECTION_NAME)
    count = info.points_count
    print(f"  ✓ Collection '{COLLECTION_NAME}': {count} vectors stored")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Ingestion complete.")
    print(f"  {len(docs)} documents → {len(all_chunks)} chunks → Qdrant")
    print(f"  Collection: {COLLECTION_NAME}  |  Dim: {EMBED_DIM}  |  Metric: cosine")
    print(f"\nReady for Sprint 2.3 — retriever + reranker.")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--incremental", action="store_true",
                        help="Add to existing collection instead of recreating")
    args = parser.parse_args()
    main(incremental=args.incremental)
