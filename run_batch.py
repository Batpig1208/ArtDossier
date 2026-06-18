"""
ArtDossier -- Batch pipeline runner
====================================
Runs the full three-panel pipeline on every image in a folder.
Models load once; each painting reuses the same pipeline instance.

Usage:
    python run_batch.py                          # runs eval_images/expert/
    python run_batch.py eval_images/expert       # explicit folder
    python run_batch.py eval_images/broad --redo # re-run even if output exists

Outputs saved to: experiments/ai/<image_stem>_<timestamp>.txt
Progress saved to: experiments/ai/_batch_log.txt  (resume on crash)
"""

import os
import sys
import datetime
import glob

# ── Config ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR  = os.path.join(BASE_DIR, "eval_images", "expert")
OUT_DIR    = os.path.join(BASE_DIR, "experiments", "expert")
EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")

if len(sys.argv) > 1 and not sys.argv[1].startswith("--"):
    IMAGE_DIR = os.path.join(BASE_DIR, sys.argv[1]) if not os.path.isabs(sys.argv[1]) else sys.argv[1]
    OUT_DIR   = IMAGE_DIR.replace("eval_images", "experiments")

LOG_PATH = os.path.join(OUT_DIR, "_batch_log.txt")   # must be after OUT_DIR override
REDO  = "--redo" in sys.argv
LIMIT = next((int(sys.argv[sys.argv.index("--limit")+1]) for _ in ["x"] if "--limit" in sys.argv), None)

# ── Setup ─────────────────────────────────────────────────────────────────────
os.makedirs(OUT_DIR, exist_ok=True)
sys.path.insert(0, BASE_DIR)

from system.pipeline.pipeline import ArtDossierPipeline

# Expert mode: only Leiden Collection + Rijksmuseum sources retrieved.
# WGA excluded at RAG level — its descriptions carry editorial bias that
# is not solely reflective of the painting (eurocentric, sexist framing).
# MET is always excluded regardless of mode.
_expert_mode = "expert" in IMAGE_DIR
_allowed_doc_types = (
    ["scholarly_catalogue", "institutional_catalogue"]
    if _expert_mode else None
)
if _expert_mode:
    print("Expert mode: RAG restricted to Leiden Collection + Rijksmuseum only")

pipeline = ArtDossierPipeline(
    top_k_retrieve=20,
    top_k_rerank=12,
    web_search=True,
    web_max_results=3,
    allowed_doc_types=_allowed_doc_types,
)

# Load already-done stems from log
done_stems = set()
if os.path.exists(LOG_PATH):
    with open(LOG_PATH, encoding="utf-8") as f:
        for line in f:
            if line.startswith("OK "):
                done_stems.add(line.split("OK ")[1].strip())

# Collect images
images = sorted([
    p for ext in EXTENSIONS
    for p in glob.glob(os.path.join(IMAGE_DIR, f"*{ext}"))
])

if not images:
    print(f"No images found in: {IMAGE_DIR}")
    sys.exit(1)

if LIMIT:
    images = images[:LIMIT]

print(f"ArtDossier Batch Runner")
print(f"Folder : {IMAGE_DIR}")
print(f"Output : {OUT_DIR}")
print(f"Images : {len(images)}  |  Already done: {len(done_stems)}{f'  |  Limit: {LIMIT}' if LIMIT else ''}")
print("=" * 60)

ok_count = skip_count = fail_count = 0
log_file = open(LOG_PATH, "a", encoding="utf-8")

for i, image_path in enumerate(images, 1):
    stem  = os.path.splitext(os.path.basename(image_path))[0]
    label = stem.replace("_", " ")

    if stem in done_stems and not REDO:
        # Also verify output file still exists — user may have deleted it manually
        output_exists = bool(glob.glob(os.path.join(OUT_DIR, f"{glob.escape(stem)}_*.txt")))
        if not output_exists:
            print(f"[{i}/{len(images)}] Re-running (output deleted): {stem}")
            done_stems.discard(stem)
        else:
            print(f"[{i}/{len(images)}] SKIP (done)  {stem}")
            skip_count += 1
        continue

    print(f"[{i}/{len(images)}] Running: {stem}")
    t0 = datetime.datetime.now()

    try:
        result = pipeline.run(
            image_path=image_path,
            user_query="Dutch Golden Age painting",
            verbose=True,
        )

        ts       = t0.strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(OUT_DIR, f"{stem}_{ts}.txt")

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(f"PAINTING : {label}\n")
            f.write(f"FILE     : {os.path.basename(image_path)}\n\n")
            f.write("PANEL 1 -- RAW CV CAPTION\n" + "=" * 60 + "\n")
            f.write(result.panel1 + "\n\n")
            f.write("PANEL 2 -- REASONING TRACE\n" + "=" * 60 + "\n")
            f.write(result.panel2_thinking + "\n\n")
            f.write("PANEL 3 -- EIGHT-LAYER DOSSIER\n" + "=" * 60 + "\n")
            f.write(result.panel3 + "\n\n")
            f.write("RAG CHUNKS USED:\n")
            for j, c in enumerate(result.rag_chunks, 1):
                f.write(f"  [{j}] score={c.get('score', 0):.4f} [{c['doc_type']}] "