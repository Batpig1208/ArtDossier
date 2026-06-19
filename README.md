# ArtDossier

A VLM-based pipeline for generating structured eight-layer art-historical dossiers for 17th-century Dutch paintings, combining visual language models, hybrid RAG retrieval, and critical aesthetics theory.

Developed as an MA thesis project.

---

## What it does

Given a painting image, ArtDossier produces a three-panel output:

- **Panel 1** — Raw visual description by the VLM (no external knowledge)
- **Panel 2** — Internal reasoning trace (thinking mode)
- **Panel 3** — Eight-layer structured dossier synthesising visual analysis, retrieved scholarly sources, and critical reflection

The eight layers are: Socio-Historical Background · Artist's Background · Materiality & Techniques · Immediate Appearance · Situated Symbols & Metaphors · Artistic & Spiritual Traditions · Horizons of Meaning · Ambiguity

---

## Repository structure

```
artdossier/
├── system/
│   ├── pipeline/
│   │   ├── pipeline.py        # Main three-panel orchestrator
│   │   ├── vlm_client.py      # SiliconFlow / OpenAI-compatible VLM wrapper
│   │   └── search_tool.py     # DuckDuckGo + full-page web search
│   └── prompts/
│       ├── panel1_caption.txt # System prompt: raw visual description
│       ├── panel2_thinking.txt
│       └── panel3_dossier.txt # System prompt: eight-layer dossier
│
├── rag/
│   ├── ingestion/
│   │   └── ingest.py          # Chunk + embed corpus → Qdrant
│   └── retrieval/
│       └── retriever.py       # Hybrid vector + keyword retrieval + reranker
│
├── corpus/
│   └── normalize.py           # Normalise raw JSON from all corpus sources
│
├── evaluation/
│   ├── run_eval.py            # Quantitative metrics (F1, METEOR, BERTScore, CLIPScore)
│   ├── run_llm_judge.py       # LLM-as-judge qualitative evaluation
│   ├── llm_judge_prompt.txt   # Editable judge system prompt
│   ├── quantitative_analysis.md
│   └── graphs/                # All evaluation figures
│
├── run_batch.py               # Run pipeline on a folder of images
├── setup_qdrant_index.py      # One-time Qdrant text index setup
├── requirements.txt
├── .env.example               # Copy to .env and fill in your API keys
└── .gitignore
```

---

## Setup

### 1. Prerequisites

- Python 3.10+
- [Docker](https://docs.docker.com/get-docker/) (for Qdrant)
- A [SiliconFlow](https://siliconflow.cn) account (or any OpenAI-compatible API endpoint)
- GPU recommended for embedding + reranking (CPU works, ~60 min for full corpus ingest)

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

For evaluation metrics:
```bash
pip install bert-score nltk open-clip-torch torchvision
python -m spacy download en_core_web_sm
```

### 3. Configure environment

```bash
cp .env.example .env
# Edit .env and add your API keys
```

### 4. Start Qdrant

```bash
docker run -p 6333:6333 -v qdrant_storage:/qdrant/storage qdrant/qdrant
```

### 5. Prepare the corpus

Place raw corpus JSON files in `corpus/raw/` following the structure:
- `corpus/raw/leiden_collection/*.json`
- `corpus/raw/rijksmuseum/all_paintings/*.json`
- `corpus/raw/wga/paintings/*.json`

Then normalise:
```bash
python corpus/normalize.py
```

### 6. Ingest into Qdrant

```bash
python rag/ingestion/ingest.py
python setup_qdrant_index.py   # creates text indexes for hybrid search
```

---

## Running the pipeline

### Single image

```python
from system.pipeline.pipeline import ArtDossierPipeline

pipeline = ArtDossierPipeline()
result = pipeline.run(image_path="painting.jpg")

print(result.panel1)           # raw visual description
print(result.panel2_thinking)  # reasoning trace
print(result.panel3)           # eight-layer dossier
```

**Expert mode** (Leiden Collection + Rijksmuseum only, WGA excluded):

```python
pipeline = ArtDossierPipeline(
    allowed_doc_types=["scholarly_catalogue", "institutional_catalogue"]
)
```

### Batch run

```bash
python run_batch.py eval_images/broad    # broad mode
python run_batch.py eval_images/expert   # expert mode
python run_batch.py eval_images/broad --redo   # re-run all
```

Outputs saved to `experiments/AI/` as `.txt` files with all three panels + RAG chunks used.

---

## Evaluation

### Quantitative metrics (F1, METEOR, BERTScore, CLIPScore)

```bash
python evaluation/run_eval.py
```

Outputs CSVs to `evaluation/Quantitative Metrics Groundedness and Situatedness/`.

### LLM qualitative judge

```bash
python evaluation/run_llm_judge.py
```

Requires `SILICONFLOW_API_KEY` or `ANTHROPIC_API_KEY` in `.env`. Resume-safe (skips already-scored paintings).

---

## Models used

| Role | Model |
|---|---|
| Panel 1 (visual description) | Qwen/Qwen3-VL-8B-Instruct |
| Panel 2+3 (reasoning + dossier) | Qwen/Qwen3-VL-32B-Thinking |
| Embedding | BAAI/bge-m3 |
| Reranker | BAAI/bge-reranker-v2-m3 |
| LLM judge | deepseek-ai/DeepSeek-V4-Flash |
| BERTScore | roberta-large |
| CLIPScore | ViT-B/32 (OpenAI) |

Models are downloaded automatically on first use to the `models/` directory (not committed to git).

---

## Corpus

The pipeline was evaluated on three corpus sources:

| Source | Entries | Doc type |
|---|---|---|
| [Leiden Collection](https://www.theleidencollection.com) | 195 | scholarly_catalogue |
| [Rijksmuseum](https://www.rijksmuseum.nl/en/collection) | 739 | institutional_catalogue |
| [Web Gallery of Art](https://www.wga.hu) | 2,724 | web_catalogue |

Raw corpus data is not redistributed here. See each institution's data access terms.

---

## Key design decisions

- **No ground truth at inference time** — the pipeline never receives the painting's title or artist as input. Retrieval is driven entirely by Panel 1's visual description.
- **Epistemic transparency** — sources are labelled by authority level (scholarly / institutional / web) and injected into the prompt so the model can distinguish what it knows vs. what it retrieved.
- **Expert mode** — excludes WGA, which carries editorial bias not solely reflective of the painting (eurocentric framing, selective coverage).
- **Hybrid retrieval** — vector search (BGE-M3) + spaCy noun-phrase keyword search + cross-encoder reranking (bge-reranker-v2-m3) + source diversity quota.

---

## Citation

If you use this code, please cite the accompanying thesis (details to be added on paper).

---

## License

MIT License. See `LICENSE` for details.
Corpus data is subject to each institution's own terms of use and is not included in this repository.
