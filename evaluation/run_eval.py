"""
ArtDossier Evaluation Script
Runs quantitative metrics (F1, METEOR, BERTScore, CLIPScore) + retrieval analysis.

Sections:
  Part A  — text parsing + F1 (runs anywhere, no ML dependencies)
  Part B  — METEOR (requires nltk)
  Part C  — BERTScore (requires bert-score + transformers)
  Part D  — CLIPScore (requires open-clip-torch + PIL)
  Part E  — Retrieval quality analysis (no ML dependencies)
  Part F  — Save CSVs

Usage:
  # Full run (install: pip install bert-score nltk open-clip-torch torchvision):
  python evaluation/run_eval.py

  # Parts A + E only (no ML dependencies):
  python evaluation/run_eval.py --skip-ml

Outputs:
  evaluation/Quantitative Metrics Groundedness and Situatedness/quantitative_results.csv
  evaluation/Retrieval Quality and Citation Faithfulness Quantitative/retrieval_analysis.csv
"""

import argparse
import json
import re
import os
import glob
import difflib
from collections import defaultdict
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
AI_DIR = ROOT / "experiments" / "AI"
BROAD_DIR = ROOT / "experiments" / "broad"
EXPERT_DIR = ROOT / "experiments" / "expert"
CORPUS_DIR = ROOT / "corpus" / "processed"
IMAGES_BROAD = ROOT / "eval_images" / "broad"
IMAGES_EXPERT = ROOT / "eval_images" / "expert"
OUT_QUANT = ROOT / "evaluation" / "Quantitative Metrics Groundedness and Situatedness"
OUT_RETRIEVAL = ROOT / "evaluation" / "Retrieval Quality and Citation Faithfulness Quantitative"

# ── CLI ────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--skip-ml", action="store_true",
                    help="Skip METEOR, BERTScore, CLIPScore (no ML dependencies needed)")
parser.add_argument("--no-clip", action="store_true", help="Skip CLIPScore only")
args, _ = parser.parse_known_args()

# ── Part A: Parse output files ─────────────────────────────────────────────────

SECTION_SEP = "=" * 60

def parse_output_file(path):
    """Parse a pipeline output .txt file into components."""
    text = Path(path).read_text(encoding="utf-8", errors="replace")

    # Header
    painting_id = ""
    fname_match = re.search(r"^PAINTING\s*:\s*(.+)$", text, re.M)
    if fname_match:
        painting_id = fname_match.group(1).strip()

    # Split into named sections
    def extract_between(text, start_marker, end_markers):
        start = text.find(start_marker)
        if start == -1:
            return ""
        start = text.find("\n", start) + 1
        # find first end marker after start
        end = len(text)
        for em in end_markers:
            pos = text.find(em, start)
            if pos != -1 and pos < end:
                end = pos
        return text[start:end].strip()

    panel1 = extract_between(text, "PANEL 1 -- RAW CV CAPTION", ["PANEL 2 --", "PANEL 3 --", "RAG CHUNKS"])
    panel2 = extract_between(text, "PANEL 2 -- REASONING TRACE", ["PANEL 3 --", "RAG CHUNKS"])
    panel3 = extract_between(text, "PANEL 3 -- EIGHT-LAYER DOSSIER", ["RAG CHUNKS"])

    # RAG CHUNKS USED
    rag_section_match = re.search(r"RAG CHUNKS USED:\s*\n(.*?)(?:\nWEB RESULTS USED:|\Z)",
                                   text, re.S)
    rag_raw = rag_section_match.group(1).strip() if rag_section_match else ""

    # Web results
    web_section_match = re.search(r"WEB RESULTS USED:\s*\n(.*?)$", text, re.S)
    web_raw = web_section_match.group(1).strip() if web_section_match else ""

    return {
        "painting_id": painting_id,
        "panel1": panel1,
        "panel2": panel2,
        "panel3": panel3,
        "full_hypothesis": (panel1 + "\n\n" + panel2 + "\n\n" + panel3).strip(),
        "text_for_clip": (panel1 + "\n\n" + panel3).strip(),  # no reasoning trace
        "rag_raw": rag_raw,
        "web_raw": web_raw,
        "raw_text": text,
    }


def parse_rag_chunks(rag_raw):
    """Parse RAG CHUNKS USED section into list of dicts."""
    chunks = []
    for line in rag_raw.splitlines():
        line = line.strip()
        m = re.match(
            r"\[(\d+)\]\s+score=([\d.]+)\s+\[(\w+)\]\s+(.*?)\s+--\s+(\w+)$",
            line
        )
        if m:
            chunks.append({
                "rank": int(m.group(1)),
                "score": float(m.group(2)),
                "doc_type": m.group(3),
                "title_artist": m.group(4).strip(),
                "section": m.group(5),
            })
    return chunks


def parse_stem(filename):
    """
    Parse a filename like '1Christ on the Mount of OlivesLD_20260612_130329.txt'
    Returns: (number, title, source_code)
    source_code: LD | RK | WG
    """
    stem = Path(filename).stem  # strip .txt
    # Strip timestamp suffix _YYYYMMDD_HHMMSS
    stem = re.sub(r"_\d{8}_\d{6}$", "", stem)
    # Match source code at end
    m = re.match(r"^(\d+)(.*?)(LD|RK|WG)$", stem)
    if m:
        number = int(m.group(1))
        title = m.group(2).strip()
        source_code = m.group(3)
        # Determine eval_set: broad or expert
        return number, title, source_code
    return None, stem, "UNK"


SOURCE_CODE_MAP = {
    "LD": "leiden_collection",
    "RK": "rijksmuseum",
    "WG": "wga",
}

GENRE_MAP = {
    # Broad set
    "1Christ on the Mount of Olives": "religious",
    "2Stone Operation (Allegory of Touch)": "genre/allegory",
    "3Self-PortraitLievens": "portrait",
    "4Still Life with Peaches, Grapes, and Melon and a Butterfly on a Stone Plate": "still life",
    "5Hunter Getting Dressed after Bathing": "genre",
    "6Still Life with Cheeses": "still life",
    "7Enjoying the Ice": "landscape/genre",
    "8View of Batavia": "landscape/cityscape",
    "9Peasants in an Interior": "genre",
    "10Isaac Blessing Jacob": "religious",
    "11Portrait of Maritge Claesdr Vooght (1577-1644)": "portrait",
    "12Satire on the Trial of Johan van Oldenbarnevelt": "history/allegory",
    "13The Company of Captain Allaert Cloeck and Lieutenant Lucas Jacobsz Rotgans, Amsterdam, 1632 (former title)": "group portrait",
    "14Portrait of Margaretha van Raephorst (1625-90)": "portrait",
    "15Bacchanal": "history/allegory",
    "16Vase of Flowers with Pocket Watch": "still life",
    "17The Tomb of William the Silent in an Imaginary Church": "architectural",
    "18Diana and Her Nymphs": "history/allegory",
    "19Officers and Sergeants of the St George Civic Guard Company": "group portrait",
    "20Woman and Maid in a Courtyard": "genre",
    "21Brothel Scene": "genre",
    "22Portrait of the Artist with His Wife Isabella de Wolff in a Tavern": "genre/portrait",
    "23TheSeduction": "genre",
    "24The Jewish Bride": "religious/portrait",
    "25Young Girl with a Flute": "portrait",
    # Expert set
    "1Woman with Carrots and Haddock": "genre/still life",
    "2Portrait of_Dina Margareta de Bye": "portrait",
    "3Elisha Refusing Naaman's Gifts": "religious",
    "4Self-Portrait with Shaded Eyes": "portrait",
    "5Boy in a Cape and Turban (Portrait of Prince Rupert of the Palatinate)": "portrait",
    "6Self-Portrait with Magic Scene": "portrait",
    "7Parable of the Lost Piece of Silver": "religious",
    "8Cat Crouching on the Ledge of an Artist's Atelier": "genre",
    "9Elderly Couple in an Interior": "genre",
    "10Lucretia": "history/allegory",
    "11Old Woman Reading": "genre",
    "12Still Life with Golden Goblet": "still life",
    "13Battle of Gibraltar in 1607": "history",
    "14Een marktstal in Batavia": "genre/landscape",
    "15Portrait of Maurits (1567-1625), Prince of Orange": "portrait",
    "16Elisabeth van Kessel (1640-1717)": "portrait",
    "17Jacob Wrestling with the Angel": "religious",
    "18St_Peter's_Denial": "religious",
    "19Portrait_of_Jacoba_van_Orliens_(164": "portrait",
    "20Elegant_Man": "portrait",
    "21Man_Tuning_a_Violin": "genre",
    "22Banquet_of_Anthony_and_Cleopatra": "history/allegory",
    "23Fisherman_and_His_Wife_in_an_Interi": "genre",
    "24Still_Life_with_a_Broken_Glass": "still life",
    "25Italian_Landscape_with_Mule_Driver": "landscape",
}


# ── Load corpus index ──────────────────────────────────────────────────────────
print("Loading corpus index...")
index_entries = json.load(open(CORPUS_DIR / "_index.json"))["entries"]
# Build lookup: (title_lower, source) → doc_id
corpus_by_title = defaultdict(list)
for e in index_entries:
    corpus_by_title[e["title"].lower()].append(e)

def find_corpus_entries(title, source_code):
    """Fuzzy-match title to corpus entries. Returns list of matching entry dicts."""
    source = SOURCE_CODE_MAP.get(source_code, None)
    title_lower = title.lower().strip()

    # 1. Exact match in correct source
    candidates = corpus_by_title.get(title_lower, [])
    exact = [e for e in candidates if source and e["source"] == source]
    if exact:
        return exact

    # 2. Exact match in any source
    if candidates:
        return candidates

    # 3. Fuzzy match across all titles
    all_titles = list(corpus_by_title.keys())
    close = difflib.get_close_matches(title_lower, all_titles, n=3, cutoff=0.6)
    results = []
    for t in close:
        for e in corpus_by_title[t]:
            if source and e["source"] == source:
                results.append(e)
    if results:
        return results

    # 4. Substring match
    for t, entries in corpus_by_title.items():
        if title_lower[:20] in t or t[:20] in title_lower:
            for e in entries:
                if source and e["source"] == source:
                    results.append(e)
    return results[:3]


def get_reference_text(doc_id):
    """Load full_text from processed corpus file."""
    path = CORPUS_DIR / f"{doc_id}.json"
    if not path.exists():
        return ""
    d = json.load(open(path, encoding="utf-8", errors="replace"))
    return d.get("full_text", "") or " ".join(d.get("text_sections", {}).values())


# ── Part A: F1 (token overlap) ─────────────────────────────────────────────────

def tokenize(text):
    """Simple word tokenizer, lowercase, alphanumeric only."""
    return re.findall(r"[a-z0-9]+", text.lower())

def compute_f1(hypothesis, reference):
    """Token-level F1 between hypothesis and reference strings."""
    hyp_tokens = set(tokenize(hypothesis))
    ref_tokens = set(tokenize(reference))
    if not hyp_tokens or not ref_tokens:
        return 0.0, 0.0, 0.0
    common = hyp_tokens & ref_tokens
    precision = len(common) / len(hyp_tokens)
    recall = len(common) / len(ref_tokens)
    if precision + recall == 0:
        return 0.0, 0.0, 0.0
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


# ── Part B: METEOR ─────────────────────────────────────────────────────────────
METEOR_AVAILABLE = False
if not args.skip_ml:
    try:
        import nltk
        try:
            nltk.data.find("corpora/wordnet")
        except LookupError:
            nltk.download("wordnet", quiet=True)
            nltk.download("punkt", quiet=True)
            nltk.download("punkt_tab", quiet=True)
        from nltk.translate.meteor_score import meteor_score as _meteor
        METEOR_AVAILABLE = True
        print("METEOR: available")
    except Exception as e:
        print(f"METEOR: not available ({e})")

def compute_meteor(hypothesis, reference):
    if not METEOR_AVAILABLE:
        return None
    try:
        hyp_tokens = tokenize(hypothesis)
        ref_tokens = tokenize(reference)
        return _meteor([ref_tokens], hyp_tokens)
    except Exception:
        return None


# ── Part C: BERTScore ──────────────────────────────────────────────────────────
BERTSCORE_AVAILABLE = False
bertscore_fn = None
if not args.skip_ml:
    try:
        from bert_score import score as _bertscore
        BERTSCORE_AVAILABLE = True
        print("BERTScore: available")
    except Exception as e:
        print(f"BERTScore: not available ({e})")

def compute_bertscore_batch(hypotheses, references):
    """Returns list of (P, R, F1) tuples."""
    if not BERTSCORE_AVAILABLE:
        return [(None, None, None)] * len(hypotheses)
    try:
        # Truncate to 512 tokens approx (BERTScore handles internally but faster if pre-truncated)
        P, R, F = _bertscore(hypotheses, references, model_type="roberta-large",
                              lang="en", verbose=False, batch_size=4)
        return list(zip(P.tolist(), R.tolist(), F.tolist()))
    except Exception as e:
        print(f"BERTScore error: {e}")
        return [(None, None, None)] * len(hypotheses)


# ── Part D: CLIPScore ──────────────────────────────────────────────────────────
CLIP_AVAILABLE = False
clip_model = clip_preprocess = clip_tokenizer = None
if not args.skip_ml and not args.no_clip:
    try:
        import open_clip
        import torch
        from PIL import Image

        clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai"
        )
        clip_tokenizer = open_clip.get_tokenizer("ViT-B-32")
        clip_model.eval()
        CLIP_AVAILABLE = True
        print("CLIPScore: available")
    except Exception as e:
        print(f"CLIPScore: not available ({e})")


def compute_clip_score(image_path, text):
    """Compute CLIPScore = 2.5 × 100 × cosine(image_embed, text_embed)."""
    if not CLIP_AVAILABLE:
        return None
    try:
        import torch
        from PIL import Image

        img = clip_preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0)
        # Truncate text to 77 tokens (CLIP limit)
        tokens = clip_tokenizer([text[:300]])
        with torch.no_grad():
            img_feat = clip_model.encode_image(img)
            txt_feat = clip_model.encode_text(tokens)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
            cosine = (img_feat * txt_feat).sum().item()
        return max(cosine, 0) * 2.5 * 100
    except Exception as e:
        print(f"CLIPScore error for {image_path}: {e}")
        return None


# ── Main loop ──────────────────────────────────────────────────────────────────

print(f"\nProcessing {AI_DIR}...")
ai_files = sorted(AI_DIR.glob("*.txt"))
print(f"Found {len(ai_files)} output files.\n")

quant_rows = []
retrieval_rows = []

# Collect all hypotheses and references for batch BERTScore
all_hyp = []
all_ref = []
file_indices = []  # which quant_row index each (hyp, ref) pair corresponds to

for fpath in ai_files:
    if fpath.name == "_batch_log.txt":
        continue

    fname = fpath.name
    number, title, source_code = parse_stem(fname)

    # Determine eval_set
    # Check if this file exists in broad or expert
    broad_match = (BROAD_DIR / fname).exists()
    expert_match = (EXPERT_DIR / fname).exists()
    eval_set = "broad" if broad_match else ("expert" if expert_match else "unknown")

    print(f"[{number}] {title} ({source_code}) [{eval_set}]")

    parsed = parse_output_file(fpath)
    chunks = parse_rag_chunks(parsed["rag_raw"])

    # ── Retrieval stats ──────────────────────────────────────────────────────
    chunk_count = len(chunks)
    top_score = chunks[0]["score"] if chunks else 0.0
    scores = [c["score"] for c in chunks]
    mean_score = sum(scores) / len(scores) if scores else 0.0

    by_type = defaultdict(int)
    for c in chunks:
        by_type[c["doc_type"]] += 1

    scholarly_count = by_type.get("scholarly_catalogue", 0)
    institutional_count = by_type.get("institutional_catalogue", 0)
    web_count = by_type.get("web_catalogue", 0)
    wga_in_rag = web_count > 0

    # Correct title in RAG
    title_clean = re.sub(r"[^\w\s]", "", title).lower().strip()
    correct_title_in_rag = False
    correct_title_rag_score = None
    for c in chunks:
        chunk_title = re.sub(r"[^\w\s]", "", c["title_artist"].split("(")[0]).lower().strip()
        ratio = difflib.SequenceMatcher(None, title_clean[:30], chunk_title[:30]).ratio()
        if ratio > 0.6:
            correct_title_in_rag = True
            correct_title_rag_score = c["score"]
            break

    # IDENTIFICATION fired? (look for "Confidence: High" in Panel 3)
    id_fired = "Confidence: High" in parsed["panel3"] or "IDENTIFICATION" in parsed["panel3"]

    retrieval_rows.append({
        "painting_id": fname.replace(".txt", ""),
        "title": title,
        "source_code": source_code,
        "eval_set": eval_set,
        "genre": GENRE_MAP.get(f"{number}{title}", GENRE_MAP.get(title, "unknown")),
        "rag_chunks_used": chunk_count,
        "top_rag_score": round(top_score, 4),
        "mean_rag_score": round(mean_score, 4),
        "scholarly_catalogue_chunks": scholarly_count,
        "institutional_catalogue_chunks": institutional_count,
        "web_catalogue_chunks": web_count,
        "wga_in_rag": wga_in_rag,
        "correct_title_in_rag": correct_title_in_rag,
        "correct_title_rag_score": round(correct_title_rag_score, 4) if correct_title_rag_score else None,
        "identification_fired": id_fired,
        "panel1_chars": len(parsed["panel1"]),
        "panel3_chars": len(parsed["panel3"]),
    })

    # ── Corpus matching ──────────────────────────────────────────────────────
    corpus_entries = find_corpus_entries(title, source_code)
    if not corpus_entries:
        # Try all sources
        corpus_entries = find_corpus_entries(title, "UNK")

    if not corpus_entries:
        print(f"  WARNING: no corpus match for '{title}' [{source_code}]")
        quant_rows.append({
            "painting_id": fname.replace(".txt", ""),
            "title": title,
            "artist": "",
            "eval_set": eval_set,
            "source_code": source_code,
            "ref_source": "none",
            "ref_doc_id": "none",
            "F1_precision": None,
            "F1_recall": None,
            "F1": None,
            "METEOR": None,
            "BERTScore_P": None,
            "BERTScore_R": None,
            "BERTScore_F1": None,
            "CLIPScore": None,
            "rag_chunks_used": chunk_count,
            "wga_in_rag": wga_in_rag,
            "top_rag_score": round(top_score, 4),
            "correct_title_in_rag": correct_title_in_rag,
            "correct_title_rag_score": None,
        })
        continue

    # Image path for CLIPScore
    img_dir = IMAGES_BROAD if eval_set == "broad" else IMAGES_EXPERT
    img_candidates = list(img_dir.glob(f"{number}*"))
    img_path = img_candidates[0] if img_candidates else None

    # One row per (painting × ref_source)
    for entry in corpus_entries[:2]:  # max 2 ref sources per painting
        ref_text = get_reference_text(entry["doc_id"])
        if not ref_text:
            continue

        hyp = parsed["full_hypothesis"]
        prec, rec, f1 = compute_f1(hyp, ref_text)
        meteor = compute_meteor(hyp, ref_text)

        clip_score = None
        if img_path and CLIP_AVAILABLE:
            clip_score = compute_clip_score(img_path, parsed["text_for_clip"][:1000])

        row = {
            "painting_id": fname.replace(".txt", ""),
            "title": title,
            "artist": entry.get("artist", ""),
            "eval_set": eval_set,
            "source_code": source_code,
            "genre": GENRE_MAP.get(f"{number}{title}", GENRE_MAP.get(title, "unknown")),
            "ref_source": entry["source"],
            "ref_doc_id": entry["doc_id"],
            "F1_precision": round(prec, 4),
            "F1_recall": round(rec, 4),
            "F1": round(f1, 4),
            "METEOR": round(meteor, 4) if meteor is not None else None,
            "BERTScore_P": None,  # filled in batch below
            "BERTScore_R": None,
            "BERTScore_F1": None,
            "CLIPScore": round(clip_score, 4) if clip_score is not None else None,
            "rag_chunks_used": chunk_count,
            "wga_in_rag": wga_in_rag,
            "top_rag_score": round(top_score, 4),
            "correct_title_in_rag": correct_title_in_rag,
            "correct_title_rag_score": round(correct_title_rag_score, 4) if correct_title_rag_score else None,
        }
        quant_rows.append(row)
        all_hyp.append(hyp)
        all_ref.append(ref_text)
        file_indices.append(len(quant_rows) - 1)

# ── Batch BERTScore ────────────────────────────────────────────────────────────
if BERTSCORE_AVAILABLE and all_hyp:
    print(f"\nRunning BERTScore on {len(all_hyp)} pairs...")
    bert_results = compute_bertscore_batch(all_hyp, all_ref)
    for i, (P, R, F) in zip(file_indices, bert_results):
        quant_rows[i]["BERTScore_P"] = round(P, 4) if P is not None else None
        quant_rows[i]["BERTScore_R"] = round(R, 4) if R is not None else None
        quant_rows[i]["BERTScore_F1"] = round(F, 4) if F is not None else None

# ── Part F: Save CSVs ──────────────────────────────────────────────────────────
import csv

def write_csv(rows, path):
    if not rows:
        print(f"No rows to write for {path}")
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {path} ({len(rows)} rows)")

write_csv(quant_rows,
    OUT_QUANT / "quantitative_results.csv")
write_csv(retrieval_rows,
    OUT_RETRIEVAL / "retrieval_analysis.csv")

print("\nDone.")
