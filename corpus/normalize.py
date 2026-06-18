"""
ArtDossier -- Corpus Normalization (Sprint 2.1)
===============================================
Reads raw JSON from all 3 sources, applies a unified schema,
cleans text, and writes one JSON per painting to corpus/processed/.

Run:
    cd "/path/to/artdossier"
    python corpus/normalize.py

Output: corpus/processed/<doc_id>.json  (489 files total)
        corpus/processed/_index.json     (master index)
"""

import json
import re
import os
import glob

BASE    = os.path.dirname(os.path.abspath(__file__))
RAW     = os.path.join(BASE, "raw")
OUT_DIR = os.path.join(BASE, "processed")
os.makedirs(OUT_DIR, exist_ok=True)


# -- Text cleaning --------------------------------------------------------------
def _fix_entities(text):
    text = re.sub(r'<[^>]+>', ' ', text)
    text = text.replace('‘', "'").replace('’', "'")
    text = text.replace('“', '"').replace('”', '"')
    text = text.replace('–', '-').replace('—', '-')
    text = text.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
    text = text.replace('&nbsp;', ' ').replace('&#8217;', "'").replace('&#8216;', "'")
    return text


def clean(text):
    """For metadata fields (title, artist, etc.) -- collapses all whitespace."""
    if not text:
        return ""
    if not isinstance(text, str):
        text = str(text)
    text = _fix_entities(text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def clean_text(text):
    """
    For body text (essays, descriptions, provenance).
    PRESERVES paragraph breaks (double newline) so the chunker
    can split at paragraph boundaries. Only normalises whitespace
    within each paragraph.
    """
    if not text:
        return ""
    if not isinstance(text, str):
        text = str(text)
    text = _fix_entities(text)
    paragraphs = text.split('\n\n')
    cleaned = []
    for p in paragraphs:
        p = re.sub(r'\s+', ' ', p).strip()
        if p:
            cleaned.append(p)
    return '\n\n'.join(cleaned)


def join_list(lst, sep="\n\n"):
    """Join a list of strings as body text, skipping empties."""
    if not lst:
        return ""
    if isinstance(lst, str):
        return clean_text(lst)
    return sep.join(clean_text(s) for s in lst if s and clean_text(s))


# -- Unified schema builder -----------------------------------------------------
def make_doc(
    doc_id, source, title, artist, artist_bio, date,
    medium, dimensions, object_url,
    iconclass_terms, tags, text_sections, authority_links,
    extra_meta=None,
):
    # Store sections with paragraphs preserved
    clean_sections = {k: clean_text(v) for k, v in text_sections.items() if clean_text(v)}

    # full_text: sections joined with headers, paragraph breaks preserved
    full_text_parts = []
    for section_name, section_text in clean_sections.items():
        full_text_parts.append(f"[{section_name.upper()}]\n{section_text}")
    full_text = "\n\n".join(full_text_parts)

    # Epistemic doc_type — drives source citation and Layer 7 transparency
    DOC_TYPE_MAP = {
        "leiden_collection":  "scholarly_catalogue",
        "rijksmuseum":        "institutional_catalogue",
        "met":                "institutional_data",
        "wga":                "web_catalogue",
    }
    doc_type = DOC_TYPE_MAP.get(source, "institutional_data")

    return {
        "doc_id":          doc_id,
        "source":          source,
        "doc_type":        doc_type,
        "title":           clean(title),
        "artist":          clean(artist),
        "artist_bio":      clean(artist_bio),
        "date":            clean(date),
        "medium":          clean(medium),
        "dimensions":      clean(dimensions),
        "object_url":      object_url or "",
        "iconclass_terms": iconclass_terms or [],
        "tags":            tags or [],
        "text_sections":   clean_sections,
        "full_text":       full_text,
        "has_text":        bool(full_text.strip()),
        "authority_links": {k: v for k, v in (authority_links or {}).items() if v},
        **(extra_meta or {}),
    }


# -- Source 1: Leiden Collection ------------------------------------------------
def process_leiden():
    files = glob.glob(os.path.join(RAW, "leiden_collection", "[0-9]*.json"))
    docs  = []
    print(f"  Leiden: {len(files)} files")

    for fpath in files:
        raw  = json.load(open(fpath, encoding="utf-8"))
        meta = raw.get("metadata", {})
        cat  = raw.get("catalogue", {})
        lid  = raw.get("leiden_id", os.path.basename(fpath).replace(".json", ""))

        iconclass = meta.get("iconclass_terms", [])
        ic_list = [{"code": t, "label": ""} for t in iconclass] if isinstance(iconclass, list) else []

        essay    = join_list(cat.get("essay", ""))
        prov     = join_list(cat.get("provenance", []))
        tech     = join_list(cat.get("technical_summary", ""))
        refs     = join_list(cat.get("references", []))
        exh      = join_list(cat.get("exhibition_history", []))
        endnotes = join_list(cat.get("endnotes", []))

        text_sections = {}
        if essay:    text_sections["essay"]       = essay
        if prov:     text_sections["provenance"]  = prov
        if tech:     text_sections["technical"]   = tech
        if refs:     text_sections["references"]  = refs
        if exh:      text_sections["exhibition"]  = exh
        if endnotes: text_sections["endnotes"]    = endnotes

        doc = make_doc(
            doc_id          = f"leiden_{lid}",
            source          = "leiden_collection",
            title           = meta.get("title", ""),
            artist          = meta.get("artist", ""),
            artist_bio      = "",
            date            = meta.get("date", ""),
            medium          = meta.get("medium", ""),
            dimensions      = meta.get("dimensions", ""),
            object_url      = meta.get("catalogue_url", ""),
            iconclass_terms = ic_list,
            tags            = [],
            text_sections   = text_sections,
            authority_links = {},
            extra_meta      = {
                "inventory_id":     meta.get("inventory_id", ""),
                "leiden_id":        lid,
                "catalogue_author": clean(cat.get("catalogue_author", "")),
                "citation":         clean(cat.get("citation", "")),
            }
        )
        docs.append(doc)

    return docs


# -- Source 2: Rijksmuseum ------------------------------------------------------
def process_rijksmuseum():
    files  = glob.glob(os.path.join(RAW, "rijksmuseum", "all_paintings", "*.json"))
    docs   = []
    print(f"  Rijksmuseum: {len(files)} files")

    for fpath in files:
        raw     = json.load(open(fpath, encoding="utf-8"))
        obj_num = raw.get("object_number", os.path.basename(fpath).replace(".json", ""))

        creator_infos = raw.get("creator_info", [])
        artist        = creator_infos[0].get("name", "") if creator_infos else raw.get("creator_name") or ""
        bio_parts     = []
        if creator_infos:
            ci = creator_infos[0]
            if ci.get("birth"): bio_parts.append(f"b. {ci['birth']}")
            if ci.get("death"): bio_parts.append(f"d. {ci['death']}")
        artist_bio = ", ".join(bio_parts)

        auth = {}
        if creator_infos:
            ci = creator_infos[0]
            if ci.get("viaf"):     auth["viaf"]    = ci["viaf"]
            if ci.get("rkd"):      auth["rkd"]     = ci["rkd"]
            if ci.get("wikidata"): auth["wikidata"] = ci["wikidata"]
            if ci.get("getty"):    auth["ulan"]     = ci["getty"]

        dims_list = raw.get("dimensions", [])
        dims      = "; ".join(dims_list) if isinstance(dims_list, list) else str(dims_list)

        # Rijksmuseum descriptions are short (rarely multi-paragraph)
        # but clean_text is still correct to use here
        desc_en = clean_text(raw.get("description_en", ""))
        desc_nl = clean_text(raw.get("description_nl", ""))
        prov    = join_list(raw.get("provenance", []))
        biblio  = join_list(raw.get("bibliography", []))

        text_sections = {}
        if desc_en: text_sections["description_en"] = desc_en
        if desc_nl: text_sections["description_nl"] = desc_nl
        if prov:    text_sections["provenance"]     = prov
        if biblio:  text_sections["bibliography"]   = biblio

        doc = make_doc(
            doc_id          = f"rijksmuseum_{obj_num.replace('-', '_')}",
            source          = "rijksmuseum",
            title           = raw.get("title", ""),
            artist          = artist,
            artist_bio      = artist_bio,
            date            = raw.get("date", ""),
            medium          = "; ".join(raw.get("medium_labels", [])) or "",
            dimensions      = dims,
            object_url      = raw.get("rijksmuseum_url", ""),
            iconclass_terms = raw.get("iconclass", []),
            tags            = [],
            text_sections   = text_sections,
            authority_links = auth,
            extra_meta      = {
                "object_number": obj_num,
                "lod_id":        raw.get("lod_id", ""),
            }
        )
        docs.append(doc)

    return docs


# -- Source 3: MET --------------------------------------------------------------
def process_met():
    files = glob.glob(os.path.join(RAW, "met", "[0-9]*.json"))
    docs  = []
    print(f"  MET: {len(files)} files")

    for fpath in files:
        raw  = json.load(open(fpath, encoding="utf-8"))
        oid  = str(raw.get("object_id", ""))
        tags = raw.get("tags", [])

        parts = []
        if raw.get("title"):
            parts.append(raw["title"])
        if raw.get("artist_display_name"):
            bio    = raw.get("artist_display_bio", "")
            prefix = clean(raw.get("artist_prefix", ""))
            astr   = f"{prefix} {raw['artist_display_name']}".strip()
            if bio: astr += f" ({bio})"
            parts.append(astr)
        if raw.get("object_date"):
            parts.append(f"Date: {raw['object_date']}")
        if raw.get("medium"):
            parts.append(f"Medium: {raw['medium']}")
        if raw.get("dimensions"):
            parts.append(f"Dimensions: {raw['dimensions']}")
        if raw.get("credit_line"):
            parts.append(f"Credit: {raw['credit_line']}")
        if raw.get("culture"):
            parts.append(f"Culture: {raw['culture']}")
        if raw.get("department"):
            parts.append(f"Department: {raw['department']}")

        tag_text = ""
        if tags:
            terms = [t["term"] for t in tags if t.get("term")]
            if terms:
                tag_text = "Subject tags: " + ", ".join(terms)

        text_sections = {}
        if parts:    text_sections["catalogue_entry"] = ". ".join(parts) + "."
        if tag_text: text_sections["subject_tags"]    = tag_text

        doc = make_doc(
            doc_id          = f"met_{oid}",
            source          = "met",
            title           = raw.get("title", ""),
            artist          = raw.get("artist_display_name", ""),
            artist_bio      = raw.get("artist_display_bio", ""),
            date            = raw.get("object_date", ""),
            medium          = raw.get("medium", ""),
            dimensions      = raw.get("dimensions", ""),
            object_url      = raw.get("object_url", ""),
            iconclass_terms = [],
            tags            = tags,
            text_sections   = text_sections,
            authority_links = {
                "ulan":     raw.get("artist_ulan_url", ""),
                "wikidata": raw.get("object_wikidata_url", "") or raw.get("artist_wikidata_url", ""),
            },
            extra_meta      = {
                "object_id":          oid,
                "accession_number":   raw.get("object_number", ""),
                "is_highlight":       raw.get("is_highlight", False),
                "is_public_domain":   raw.get("is_public_domain", True),
                "department":         raw.get("department", ""),
                "culture":            raw.get("culture", ""),
                "artist_nationality": raw.get("artist_nationality", ""),
                "artist_prefix":      clean(raw.get("artist_prefix", "")),
                "gallery_number":     raw.get("gallery_number", ""),
            }
        )
        docs.append(doc)

    return docs


# -- Source 4: WGA --------------------------------------------------------------
def process_wga():
    files = glob.glob(os.path.join(RAW, "wga", "paintings", "*.json"))
    docs  = []
    print(f"  WGA: {len(files)} files")

    for fpath in files:
        raw   = json.load(open(fpath, encoding="utf-8"))
        stem  = os.path.basename(fpath).replace(".json", "")

        desc = clean_text(raw.get("description", ""))
        text_sections = {}
        if desc:
            text_sections["description"] = desc

        doc = make_doc(
            doc_id          = f"wga_{stem}",
            source          = "wga",
            title           = raw.get("title", ""),
            artist          = raw.get("author", ""),
            artist_bio      = raw.get("born_died", ""),
            date            = raw.get("date", ""),
            medium          = raw.get("technique", ""),
            dimensions      = "",
            object_url      = raw.get("wga_url", ""),
            iconclass_terms = [],
            tags            = [{"term": raw.get("type", "")}] if raw.get("type") else [],
            text_sections   = text_sections,
            authority_links = {},
            extra_meta      = {
                "wga_type":      raw.get("type", ""),
                "wga_location":  raw.get("location", ""),
                "wga_timeframe": raw.get("timeframe", ""),
            }
        )
        docs.append(doc)

    return docs


# -- Main -----------------------------------------------------------------------
def main():
    print("=" * 60)
    print("ArtDossier -- Corpus Normalization")
    print("=" * 60)
    print()

    all_docs = []
    print("Processing sources:")
    all_docs += process_leiden()
    all_docs += process_rijksmuseum()
    all_docs += process_met()
    all_docs += process_wga()
    print(f"\nTotal documents: {len(all_docs)}")

    print("\nSaving processed files...")
    for doc in all_docs:
        out_path = os.path.join(OUT_DIR, f"{doc['doc_id']}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False, indent=2)

    has_text      = sum(1 for d in all_docs if d["has_text"])
    has_essay     = sum(1 for d in all_docs if "essay" in d["text_sections"])
    has_desc      = sum(1 for d in all_docs if any(k.startswith("description") for k in d["text_sections"]))
    has_prov      = sum(1 for d in all_docs if "provenance" in d["text_sections"])
    has_iconclass = sum(1 for d in all_docs if d["iconclass_terms"])
    has_tags      = sum(1 for d in all_docs if d["tags"])
    has_auth      = sum(1 for d in all_docs if d["authority_links"])
    total_chars   = sum(len(d["full_text"]) for d in all_docs)

    by_source = {}
    for d in all_docs:
        by_source[d["source"]] = by_source.get(d["source"], 0) + 1

    print(f"\n{'─'*60}")
    for src, cnt in by_source.items():
        print(f"  {src:<30} {cnt} docs")
    print(f"{'─'*60}")
    print(f"  Has any text:         {has_text}/{len(all_docs)}")
    print(f"  Has essay (Leiden):   {has_essay}")
    print(f"  Has description:      {has_desc}")
    print(f"  Has provenance:       {has_prov}")
    print(f"  Has iconclass:        {has_iconclass}")
    print(f"  Has AAT tags (MET):   {has_tags}")
    print(f"  Has authority links:  {has_auth}")
    print(f"  Total text chars:     {total_chars:,}")
    print(f"  Avg chars/doc:        {total_chars // max(len(all_docs), 1):,}")

    index = [
        {
            "doc_id":     d["doc_id"],
            "source":     d["source"],
            "title":      d["title"],
            "artist":     d["artist"],
            "date":       d["date"],
            "has_text":   d["has_text"],
            "text_chars": len(d["full_text"]),
            "sections":   list(d["text_sections"].keys()),
            "object_url": d["object_url"],
        }
        for d in all_docs
    ]

    idx_path = os.path.join(OUT_DIR, "_index.json")
    with open(idx_path, "w", encoding="utf-8") as f:
        json.dump({"total": len(all_docs), "by_source": by_source, "entries": index},
                  f, ensure_ascii=False, indent=2)

    print(f"\n  Files -> corpus/processed/")
    print(f"  Index -> corpus/processed/_index.json")
    print(f"\nReady for Sprint 2.2 -- chunking + Qdrant ingestion.")


if __name__ == "__main__":
    main()
