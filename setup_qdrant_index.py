"""
One-time setup: create full-text payload indexes on Qdrant for hybrid search.
Run this ONCE before using retriever v3. Safe to re-run (idempotent).

Usage:
    python setup_qdrant_index.py
"""
import os
from dotenv import load_dotenv
load_dotenv()

from qdrant_client import QdrantClient
from qdrant_client.models import PayloadSchemaType

QDRANT_URL      = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY  = os.getenv("QDRANT_API_KEY", None)
COLLECTION_NAME = "artdossier"

kwargs = {"url": QDRANT_URL}
if QDRANT_API_KEY:
    kwargs["api_key"] = QDRANT_API_KEY
qc = QdrantClient(**kwargs)

fields = ["text", "title", "artist"]
for field in fields:
    try:
        qc.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name=field,
            field_schema=PayloadSchemaType.TEXT,
        )
        print(f"  ✓ text index created: {field}")
    except Exception as e:
        if "already exists" in str(e).lower() or "conflict" in str(e).lower():
            print(f"  ✓ text index already exists: {field}")
        else:
            print(f"  ✗ {field}: {e}")

print("\nDone. Keyword search is now active in retriever v3.")
