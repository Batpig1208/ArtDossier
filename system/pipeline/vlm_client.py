"""
ArtDossier VLM Client — SiliconFlow / Qwen3-VL-32B
====================================================
Wraps the OpenAI-compatible SiliconFlow API.
Handles: image encoding, think mode, tool use, streaming, retry.
"""

import os
import base64
import time
import json
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ── Config ─────────────────────────────────────────────────────────────────────
API_KEY    = os.getenv("SILICONFLOW_API_KEY")
BASE_URL   = os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1")
VLM_MODEL  = os.getenv("VLM_MODEL", "Qwen/Qwen3-VL-32B")

if not API_KEY:
    raise EnvironmentError("SILICONFLOW_API_KEY not set. Copy .env.example → .env and add your key.")

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)


# ── Image helpers ───────────────────────────────────────────────────────────────
def encode_image(image_path: str, max_px: int = 1600) -> tuple[str, str]:
    """
    Load an image from disk, resize if needed, return (base64_str, mime_type).
    Keeps longest side ≤ max_px to stay within model limits.
    """
    from PIL import Image
    import io

    img = Image.open(image_path)

    # Resize if too large
    if max(img.size) > max_px:
        img.thumbnail((max_px, max_px), Image.LANCZOS)

    # Convert to RGB if needed (handles RGBA, P-mode etc.)
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return b64, "image/jpeg"


def image_content_block(image_path: str) -> dict:
    """Build an OpenAI-format image content block from a local file."""
    b64, mime = encode_image(image_path)
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{mime};base64,{b64}"}
    }


def url_content_block(url: str) -> dict:
    """Build an image content block from a public URL (no download needed)."""
    return {
        "type": "image_url",
        "image_url": {"url": url}
    }


# ── Core call ───────────────────────────────────────────────────────────────────
def call(
    messages: list[dict],
    model: str = VLM_MODEL,
    max_tokens: int = 4096,
    temperature: float = 0.3,
    think: bool = False,
    tools: list[dict] | None = None,
    stream: bool = False,
    retries: int = 3,
):
    """
    Call the SiliconFlow API with retry logic.

    Args:
        messages:    OpenAI-format messages list
        model:       Model string (default: VLM_MODEL from .env)
        max_tokens:  Max output tokens
        temperature: Sampling temperature
        think:       Enable Qwen3 think mode (try /think suffix or extra_body)
        tools:       OpenAI-format tool definitions for function calling
        stream:      Stream response (yields chunks if True)
        retries:     Max retry attempts on rate limit / server error
    """
    # Think mode: Qwen3-VL-32B-Thinking has thinking built in — no suffix needed
    # If using a non-thinking model variant, append /think suffix
    effective_model = model
    extra_body = {}
    if think and not model.endswith("Thinking") and not model.endswith("/think"):
        effective_model = model + "/think"

    kwargs = dict(
        model=effective_model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        stream=stream,
    )
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"
    if extra_body:
        kwargs["extra_body"] = extra_body

    for attempt in range(retries):
        try:
            return client.chat.completions.create(**kwargs)
        except Exception as e:
            err = str(e).lower()
            if "rate limit" in err or "429" in err:
                wait = 2 ** attempt
                print(f"  Rate limit, waiting {wait}s...")
                time.sleep(wait)
            elif "500" in err or "502" in err or "503" in err:
                wait = 2 ** attempt
                print(f"  Server error, retry {attempt+1}/{retries}...")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError(f"Max retries ({retries}) exceeded.")


# ── Convenience wrappers ────────────────────────────────────────────────────────
def caption_image(
    image_path: str | None = None,
    image_url: str | None = None,
    system_prompt: str = "You are an expert art historian.",
    user_prompt: str = "Describe this painting in detail.",
    **kwargs
) -> str:
    """
    Panel 1 — raw CV caption, no retrieval.
    Returns the response text.
    """
    img_block = (
        image_content_block(image_path) if image_path
        else url_content_block(image_url)
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": [img_block, {"type": "text", "text": user_prompt}]},
    ]
    resp = call(messages, **kwargs)
    return resp.choices[0].message.content


def think_and_retrieve(
    image_path: str | None = None,
    image_url: str | None = None,
    initial_caption: str = "",
    system_prompt: str = "",
    user_prompt: str = "",
    tools: list[dict] | None = None,
    **kwargs
) -> tuple[str, str, list]:
    """
    Panel 2 — think mode + tool use.
    Returns (thinking_text, response_text, tool_calls).
    """
    img_block = (
        image_content_block(image_path) if image_path
        else url_content_block(image_url)
    )
    user_content = [
        img_block,
        {"type": "text", "text": f"Initial caption:\n{initial_caption}\n\n{user_prompt}"}
    ]
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_content},
    ]
    resp = call(messages, think=True, tools=tools, **kwargs)
    msg = resp.choices[0].message

    # Extract thinking content (Qwen3 returns it in a separate block or prefixed)
    thinking = ""
    response_text = msg.content or ""

    # Some SiliconFlow versions return thinking in a separate field
    if hasattr(msg, "reasoning_content") and msg.reasoning_content:
        thinking = msg.reasoning_content
    elif response_text.startswith("<think>"):
        # Strip <think>...</think> block
        import re
        m = re.match(r"<think>(.*?)</think>(.*)", response_text, re.DOTALL)
        if m:
            thinking, response_text = m.group(1).strip(), m.group(2).strip()

    tool_calls = msg.tool_calls or []
    return thinking, response_text, tool_calls


def extract_thinking(response_text: str) -> tuple[str, str]:
    """
    Split <think>...</think> from response text.
    Returns (thinking, clean_response).
    """
    import re
    m = re.match(r"<think>(.*?)</think>(.*)", response_text, re.DOTALL)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return "", response_text
