"""Real dataset downloader for MC-LLM.

Downloads open conversational/text datasets from the Hugging Face Hub using
the public ``datasets-server`` API (Parquet format) and reads them streaming
with ``pyarrow`` — no ``datasets``/``dill`` dependency, so it works on any
Python version (including 3.14) and in Colab.

Each downloaded dataset is written to ``data/raw/<name>.jsonl`` as
``{"text": ...}`` lines, ready for the existing cleaning/sharding pipeline.

Conversational datasets (e.g. PersonaChat) are converted into the
``USER: ... / ASSISTANT: ...`` dialogue format.
"""
from __future__ import annotations

import json
import os
import urllib.request
from typing import Callable, List, Optional

import pyarrow.parquet as pq

_DATASETS_SERVER = "https://datasets-server.huggingface.co"


def _http_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "mc-llm/1.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def get_parquet_urls(repo: str, config: str, split: str) -> List[str]:
    url = f"{_DATASETS_SERVER}/parquet?dataset={repo}&config={config}&split={split}"
    data = _http_json(url)
    return [f["url"] for f in data.get("parquet_files", [])]


def download_file(url: str, dest: str) -> str:
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if not os.path.exists(dest):
        req = urllib.request.Request(url, headers={"User-Agent": "mc-llm/1.0"})
        with urllib.request.urlopen(req, timeout=300) as r, open(dest, "wb") as f:
            f.write(r.read())
    return dest


# ---------------------------------------------------------------------------
# text extractors
# ---------------------------------------------------------------------------
def wikipedia_extractor(row: dict) -> str:
    text = row.get("text", "")
    title = row.get("title", "")
    if title and text and not text.startswith(title):
        return f"{title}\n{text}"
    return text


def persona_chat_extractor(row: dict) -> str:
    """PersonaChat → USER/ASSISTANT dialogue.  Handles the standard schema
    (``utterances`` list with per-turn ``history``), plus a few variants."""
    utterances = row.get("utterances", row.get("dialog", row.get("turns")))
    if not utterances:
        return ""
    # full conversation = history of the last turn (alternating speakers)
    last = utterances[-1] if isinstance(utterances, list) and utterances else {}
    history = None
    if isinstance(last, dict):
        history = last.get("history")
    if history is None and isinstance(row, dict):
        history = row.get("history")
    if not history:
        return ""
    parts = []
    for i, msg in enumerate(history):
        role = "USER" if i % 2 == 0 else "ASSISTANT"
        parts.append(f"{role}: {msg}")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# main downloader
# ---------------------------------------------------------------------------
def download_dataset(
    repo: str,
    config: str,
    split: str,
    out_jsonl: str,
    extractor: Callable[[dict], str],
    cache_dir: Optional[str] = None,
    max_rows: int = 0,
    max_files: int = 0,
    min_chars: int = 30,
) -> int:
    """Download a dataset and write cleaned text to a JSONL file.

    Returns the number of documents written.
    """
    urls = get_parquet_urls(repo, config, split)
    if max_files:
        urls = urls[:max_files]

    cache_dir = cache_dir or os.path.join("data", "download")
    os.makedirs(os.path.dirname(out_jsonl), exist_ok=True)

    total = 0
    with open(out_jsonl, "w", encoding="utf-8") as out:
        for u in urls:
            name = u.rstrip("/").split("/")[-1]
            local = download_file(u, os.path.join(cache_dir, name))
            pf = pq.ParquetFile(local)
            for batch in pf.iter_batches(batch_size=512):
                for row in batch.to_pylist():
                    try:
                        text = extractor(row)
                    except Exception:
                        continue
                    if not text or len(text) < min_chars:
                        continue
                    out.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                    total += 1
                    if max_rows and total >= max_rows:
                        return total
    return total


# ---------------------------------------------------------------------------
# curated presets
# ---------------------------------------------------------------------------
PRESETS = {
    "wikipedia_ru": dict(repo="wikimedia/wikipedia", config="20231101.ru",
                         split="train", extractor=wikipedia_extractor),
    "wikipedia_en": dict(repo="wikimedia/wikipedia", config="20231101.en",
                         split="train", extractor=wikipedia_extractor),
    "persona_chat_ru": dict(repo="AlekseyKorshuk/persona-chat", config="default",
                            split="train", extractor=persona_chat_extractor),
}


def download_preset(name: str, out_dir: str = "data/raw", max_rows: int = 0,
                    max_files: int = 0, cache_dir: Optional[str] = None) -> int:
    if name not in PRESETS:
        raise ValueError(f"Unknown preset {name!r}; choose from {list(PRESETS)}")
    cfg = PRESETS[name]
    out_jsonl = os.path.join(out_dir, f"{name}.jsonl")
    return download_dataset(
        cfg["repo"], cfg["config"], cfg["split"], out_jsonl,
        extractor=cfg["extractor"], cache_dir=cache_dir,
        max_rows=max_rows, max_files=max_files,
    )
