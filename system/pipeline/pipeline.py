"""
ArtDossier -- Three-Panel Pipeline (Sprint 3)
=============================================
Orchestrates the full analysis of a painting through three panels:

  Panel 1: Raw CV caption (image only, no RAG, no think mode)
           Prompt: describe visible objects, composition, materials, zero external knowledge
           Model:  Qwen3-VL-32B-Thinking (think mode OFF)

  Panel 2 + 3 (ONE CALL):
           Inputs: image + Panel 1 + RAG chunks + web results
           System: panel3_dossier.txt (critical aesthetics 8-layer prompt)
           Model:  Qwen3-VL-32B-Thinking
           Output: <think>...</think> = Panel 2 (reasoning trace)
                   response text      = Panel 3 (8-layer dossier)

  Qwen3-VL-32B-Thinking always produces <think> blocks before responding.
  We extract both from the same call — saving ~50% tokens vs two separate calls.

Retrieval is managed by the pipeline, not by the model via tool calls.
This keeps the epistemic chain explicit and auditable.

Usage:
    from system.pipeline.pipeline import ArtDossierPipeline
    pipeline = ArtDossierPipeline()
    result = pipeline.run("path/to/painting.jpg", user_query="Who painted this?")
    print(result.panel1)
    print(result.panel2_thinking)
    print(result.panel3)
"""

import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

# Prompt files
PROMPTS_DIR = Path(__file__).parent.parent / "prompts"

def _load_prompt(filename: str) -> str:
    path = PROMPTS_DIR / filename
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    raise FileNotFoundError(f"Prompt file not found: {path}")


# ── Response dataclass ─────────────────────────────────────────────────────────
@dataclass
class DossierResponse:
    # Panel outputs
    panel1:           str = ""   # raw CV caption
    panel2_thinking:  str = ""   # <think> content only
    panel3:           str = ""   # 8-layer dossier

    # Retrieved context (for transparency / logging)
    rag_chunks:       list = field(default_factory=list)
    web_results:      list = field(default_factory=list)

    # Token usage
    tokens_panel1:    dict = field(default_factory=dict)
    tokens_panel2:    dict = field(default_factory=dict)
    tokens_panel3:    dict = field(default_factory=dict)

    # Timing
    elapsed_panel1:   float = 0.0
    elapsed_panel2:   float = 0.0
    elapsed_panel3:   float = 0.0

    @property
    def total_elapsed(self) -> float:
        return self.elapsed_panel1 + self.elapsed_panel2 + self.elapsed_panel3

    def summary(self) -> str:
        return (
            f"Panel 1: {len(self.panel1)} chars\n"
            f"Panel 2 (thinking): {len(self.panel2_thinking)} chars\n"
            f"Panel 3 (dossier): {len(self.panel3)} chars\n"
            f"RAG chunks: {len(self.rag_chunks)}\n"
            f"Web results: {len(self.web_results)}\n"
            f"Total time: {self.total_elapsed:.1f}s"
        )


# ── Pipeline ───────────────────────────────────────────────────────────────────
class ArtDossierPipeline:
    def __init__(
        self,
        top_k_retrieve: int = 20,
        top_k_rerank:   int = 5,
        web_search:     bool = True,
        web_max_results: int = 4,
        allowed_doc_types: list = None,
    ):
        self.top_k_retrieve    = top_k_retrieve
        self.top_k_rerank      = top_k_rerank
        self.web_search        = web_search
        self.web_max_results   = web_max_results
        # None = all sources (Leiden + Rijksmuseum + WGA); MET always excluded.
        # Expert mode: ["scholarly_catalogue", "institutional_catalogue"] — WGA also excluded.
        self.allowed_doc_types = allowed_doc_types

        # Lazy-load heavy modules
        self._client    = None
        self._retriever = None
        self._searcher  = None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            import httpx
            self._client = OpenAI(
                api_key=os.getenv("SILICONFLOW_API_KEY"),
                base_url=os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1"),
                timeout=httpx.Timeout(600.0, connect=10.0),  # 10min total (Panel2+3 needs it)
            )
        return self._client

    def _get_model(self) -> str:
        """Panel 2+3 model — large thinking model."""
        return os.getenv("VLM_MODEL", "Qwen/Qwen3-VL-32B-Thinking")

    def _get_panel1_model(self) -> str:
        """Panel 1 model — small fast non-thinking model."""
        return os.getenv("PANEL1_MODEL", "Qwen/Qwen3-VL-8B-Instruct")

    def _image_block(self, image_path: str = None, image_url: str = None) -> dict:
        if image_url:
            return {"type": "image_url", "image_url": {"url": image_url}}
        if image_path:
            from system.pipeline.vlm_client import encode_image
            b64, mime = encode_image(image_path)
            return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}
        raise ValueError("Provide either image_path or image_url")

    @staticmethod
    def _extract_thinking(text: str) -> tuple[str, str]:
        """Split <think>...</think> from response. Returns (thinking, response)."""
        m = re.match(r"<think>(.*?)</think>(.*)", text, re.DOTALL)
        if m:
            return m.group(1).strip(), m.group(2).strip()
        # Some models put thinking in reasoning_content field (handled in run())
        return "", text

    def _call(
        self,
        messages: list,
        model: str = None,
        max_tokens: int = 1500,
        temperature: float = 0.3,
        top_p: float = 0.9,
        frequency_penalty: float = 0.0,
        thinking_budget: int = 0,
        top_k: int = -1,
        min_p: float = 0.0,
    ) -> tuple[str, str, dict]:
        """
        Make a VLM call. Returns (thinking_text, response_text, usage_dict).
        model: override which model to use (defaults to VLM_MODEL env var)
        """
        client       = self._get_client()
        model        = model or self._get_model()

        # Build extra_body for non-standard params SiliconFlow may support
        extra_body = {}
        if thinking_budget > 0:
            extra_body["thinking_budget"] = thinking_budget
        if top_k != -1:
            extra_body["top_k"] = top_k
        if min_p > 0:
            extra_body["min_p"] = min_p

        kwargs = dict(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
        )
        if extra_body:
            kwargs["extra_body"] = extra_body

        t0 = time.time()
        # Retry up to 3 times on SiliconFlow 500 errors (transient server failures)
        for _attempt in range(3):
            try:
                resp = client.chat.completions.create(**kwargs)
                break
            except Exception as e:
                err = str(e).lower()
                retryable = (
                    "500" in err or "502" in err or "503" in err
                    or "timeout" in err or "timed out" in err
                    or "connection error" in err or "connecterror" in err
                    or "getaddrinfo" in err or "network" in err
                )
                if _attempt < 2 and retryable:
                    wait = 30 * (_attempt + 1)   # 30s / 60s — network issues need cooldown
                    print(f"      API error ({e}) — retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    raise
        elapsed = time.time() - t0

        msg     = resp.choices[0].message
        content = msg.content or ""

        # Extract thinking
        thinking = ""
        if hasattr(msg, "reasoning_content") and msg.reasoning_content:
            thinking = msg.reasoning_content
        elif "<think>" in content:
            thinking, content = self._extract_thinking(content)

        usage = {
            "prompt_tokens":     resp.usage.prompt_tokens     if resp.usage else 0,
            "completion_tokens": resp.usage.completion_tokens if resp.usage else 0,
            "elapsed_s":         round(elapsed, 2),
            "thinking_chars":    len(thinking),
        }
        return thinking, content, usage

    # ── Panel 1 ─────────────────────────────────────────────────────────────────
    def _panel1(self, img_block: dict) -> tuple[str, dict]:
        """
        Raw CV caption — same model, thinking DISABLED.
        enable_thinking=False skips <think> blocks entirely → much faster.
        Parameters chosen for factual, grounded visual description.
        """
        system = _load_prompt("panel1_caption.txt")
        messages = [
            {"role": "system", "content": system},
            {"role": "user",   "content": [
                img_block,
                {"type": "text", "text": "What is present in this painting?"}
            ]},
        ]
        _, response, usage = self._call(
            messages,
            model=self._get_panel1_model(),   # 32B instruct, thinking OFF
            max_tokens=1800,
            temperature=0.4,
            top_p=0.9,
            frequency_penalty=0.1,
        )
        return response, usage

    # ── Retrieve ─────────────────────────────────────────────────────────────────
    def _retrieve(self, query: str) -> tuple[list, list]:
        """
        3-query RAG retrieval.

        Panel 1 has three sections (visual, iconographic, synthesis).
        Each is searched separately — 7 candidates each = 21 total before reranking.
        The reranker uses the synthesis section (most condensed) as its anchor.
        Falls back to character-position split if section markers aren't found.
        """
        from rag.retrieval.retriever import retrieve_multi
        from system.pipeline.search_tool import search_web

        def _find_pos(text: str, markers: list) -> int:
            """Case-insensitive search for any marker, returns char position or -1."""
            lower = text.lower()
            for m in markers:
                idx = lower.find(m.lower())
                if idx != -1:
                    return idx
            return -1

        ICON_MARKERS = [
            "iconographic identification",
            "iconographic identification:",
            "**iconographic",
            "## iconographic",
            "iconographic:",
        ]
        SYN_MARKERS = [
            "synthesis:",
            "synthesis\n",
            "**synthesis",
            "## synthesis",
            "synthesi",   # catches "Synthesising" etc.
        ]

        idx_b = _find_pos(query, ICON_MARKERS)
        idx_c = _find_pos(query, SYN_MARKERS)

        n = len(query)
        if idx_b > 100 and idx_c > idx_b + 100:
            # Clean three-way split from markers
            q_a = query[:idx_b].strip()
            q_b = query[idx_b:idx_c].strip()
            q_c = query[idx_c:].strip()
        elif idx_b > 100:
            # Found visual/iconographic split but no synthesis marker
            q_a = query[:idx_b].strip()
            q_b = query[idx_b:].strip()
            q_c = query[max(0, n - 400):].strip()   # last ~400 chars as synthesis proxy
        else:
            # Fallback: positional split (60% visual, 30% iconographic, 10% synthesis)
            cut1 = int(n * 0.60)
            cut2 = int(n * 0.90)
            q_a = query[:cut1].strip()
            q_b = query[cut1:cut2].strip()
            q_c = query[cut2:].strip()

        queries = [q for q in [q_a, q_b, q_c] if len(q) > 80]

        rag_chunks = retrieve_multi(
            queries=queries,
            top_k_per_query=15,
            top_k_rerank=self.top_k_rerank,
            allowed_doc_types=self.allowed_doc_types,
        )

        web_results = []
        if self.web_search and query.strip():
            # Query A: synthesis section — broad context
            web_results = search_web(
                q_c[:400] if q_c else query[:400],
                max_results=self.web_max_results,
            )
            # Query B: iconographic keywords — specific anchors
            # Reuse the same spaCy extraction already running for corpus RAG
            try:
                from rag.retrieval.retriever import _extract_keywords
                kw_list = _extract_keywords(q_b[:1200]) if q_b.strip() else []
                if kw_list:
                    kw_query = " ".join(kw_list[:4]) + " Dutch Golden Age painting"
                    web_results += search_web(kw_query, max_results=self.web_max_results)
            except Exception:
                pass   # keyword query fails silently — synthesis results still returned

        return rag_chunks, web_results

    # ── Panel 2 + 3 (one call) ───────────────────────────────────────────────────
    def _panel2_and_3(self, img_block: dict, panel1_text: str,
                      rag_chunks: list, web_results: list) -> tuple[str, str, dict]:
        """
        Single call that produces both Panel 2 and Panel 3.

        Qwen3-VL-32B-Thinking always emits <think>...</think> before responding.
        We use the Panel 3 system prompt (8-layer dossier) as the task —
        the model's reasoning while writing the dossier IS Panel 2.

        Returns: (panel2_thinking, panel3_dossier, usage)
        """
        from rag.retrieval.retriever import format_for_prompt
        from system.pipeline.search_tool import format_web_results

        system  = _load_prompt("panel3_dossier.txt")
        rag_text = format_for_prompt(rag_chunks) if rag_chunks else "No corpus sources retrieved."
        web_text = format_web_results(web_results) if web_results else "No web results."

        context = (
            f"PANEL 1 — RAW VISUAL DESCRIPTION:\n{panel1_text}\n\n"
            f"RETRIEVED CORPUS SOURCES:\n{rag_text}\n\n"
            f"WEB SEARCH RESULTS:\n{web_text}\n\n"
            f"Write the eight-layer dossier."
        )

        messages = [
            {"role": "system", "content": system},
            {"role": "user",   "content": [
                img_block,
                {"type": "text", "text": context},
            ]},
        ]

        # thinking_budget=3000 + dossier=3000 → max_tokens=6000
        # Reduced from 4000 thinking tokens to 3000 — saves ~1 min per run on SiliconFlow
        # at 32B speed. Dossier quality preserved; thinking is Panel 2 evidence not primary output.
        thinking, dossier, usage = self._call(
            messages,
            model=self._get_model(),   # Qwen3-VL-32B-Thinking
            max_tokens=6000,
            temperature=0.6,
            top_p=0.9,
            top_k=-1,
            min_p=0.05,
            frequency_penalty=0.3,
            thinking_budget=3000,
        )
        usage["dossier_chars"] = len(dossier)
        return thinking, dossier, usage

    # ── Main entry ───────────────────────────────────────────────────────────────
    def run(
        self,
        image_path: str = None,
        image_url:  str = None,
        user_query: str = "",
        verbose:    bool = True,
    ) -> DossierResponse:
        """
        Run the full three-panel pipeline on a painting.

        Args:
            image_path: Local path to painting image
            image_url:  Public URL to painting image (no download needed)
            user_query: Optional user question (enriches RAG query)
            verbose:    Print progress to stdout
        """
        result    = DossierResponse()
        img_block = self._image_block(image_path=image_path, image_url=image_url)

        # ── Panel 1 ────────────────────────────────────────────────────────────
        if verbose: print("[1/3] Panel 1 — raw CV caption (thinking OFF)...")
        t0 = time.time()
        result.panel1, result.tokens_panel1 = self._panel1(img_block)
        result.elapsed_panel1 = time.time() - t0
        if verbose:
            print(f"      {len(result.panel1)} chars  ({result.elapsed_panel1:.1f}s)")

        # ── Retrieve ───────────────────────────────────────────────────────────
        if verbose: print("[  ] Retrieving from corpus + web (3 queries)...")
        rag_query = f"{result.panel1} {user_query}".strip()
        try:
            result.rag_chunks, result.web_results = self._retrieve(rag_query)
        except BaseException as e:
            import traceback
            print(f"      RETRIEVAL ERROR ({type(e).__name__}): {e}")
            traceback.print_exc()
            result.rag_chunks, result.web_results = [], []

        if verbose:
            print(f"      {len(result.rag_chunks)} RAG chunks, {len(result.web_results)} web results")
            for i, c in enumerate(result.rag_chunks[:3], 1):
                print(f"      [{i}] {c['score']:.3f}  {c['title'][:45]}")

        # ── Panel 2 + 3 (single call) ──────────────────────────────────────────
        if verbose: print("[2/3] Panel 2+3 —