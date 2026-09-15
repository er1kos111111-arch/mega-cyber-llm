"""Text cleaning for the MC-LLM data pipeline.

Pure, deterministic, streaming-friendly cleaning primitives:

* Unicode normalization (NFC / NFKC);
* control-character and zero-width stripping;
* whitespace collapsing;
* length limits;
* per-language hooks (later stages handle language detection).
"""
from __future__ import annotations

import re
import unicodedata
from typing import Iterable, Iterator, Optional

# Characters we always remove: C0/C1 controls except common whitespace, and
# zero-width/formatting characters that carry no meaning for LM pretraining.
_CONTROL_RE = re.compile(
    r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f\u200b\u200c\u200d\u200e\u200f\ufeff]"
)
_WHITESPACE_RE = re.compile(r"[ \t\r\f\v]+")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
# Unicode space variants (NBSP, thin/em/en spaces, ideographic space, etc.)
_SPACE_VARIANTS_RE = re.compile(r"[\u00a0\u1680\u2000-\u200a\u202f\u205f\u3000]")

# Placeholder tokens sometimes leaking from web crawls.
_URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")

MAX_DOC_CHARS = 1_000_000
MIN_DOC_CHARS = 1


def normalize_unicode(text: str, form: str = "NFC") -> str:
    return unicodedata.normalize(form, text)


def strip_control_chars(text: str) -> str:
    return _CONTROL_RE.sub("", text)


def collapse_whitespace(text: str) -> str:
    text = _WHITESPACE_RE.sub(" ", text)
    text = _MULTI_NEWLINE_RE.sub("\n\n", text)
    return text.strip()


def clean_text(text: str, *, unicode_form: str = "NFC",
               redact_urls: bool = False, redact_emails: bool = False) -> str:
    """Full cleaning pipeline for a single document string."""
    if text is None:
        return ""
    text = normalize_unicode(text, unicode_form)
    text = _SPACE_VARIANTS_RE.sub(" ", text)
    text = strip_control_chars(text)
    if redact_urls:
        text = _URL_RE.sub("<URL>", text)
    if redact_emails:
        text = _EMAIL_RE.sub("<EMAIL>", text)
    text = collapse_whitespace(text)
    return text


def clean_documents(docs: Iterable[str], **kwargs) -> Iterator[str]:
    """Streaming wrapper: yields cleaned documents one at a time."""
    for doc in docs:
        cleaned = clean_text(doc, **kwargs)
        if cleaned:
            yield cleaned
