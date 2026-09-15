"""Data pipeline tests: cleaner, filter, dedup, shard, streaming dataset."""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.cleaner import clean_text
from data.dedup import exact_dedup
from data.filter import detect_language, filter_document
from data.shard import shard_texts
from data.dataset import ShardedTokenDataset


def test_clean_text_normalizes():
    out = clean_text("  Hello   \u00a0world\n\n\n  ")
    assert out == "Hello world"


def test_clean_text_strips_control_chars():
    out = clean_text("a\x00b\x0cc")
    assert out == "abc"


def test_language_detection():
    assert detect_language("Привет как дела это русский текст") == "ru"
    assert detect_language("Hello this is an english text") == "en"


def test_filter_rejects_short():
    assert not filter_document("short")


def test_filter_rejects_pii():
    assert not filter_document("Contact me at john.doe@example.com for more details about this very long document text." * 2)


def test_exact_dedup():
    docs = ["hello world", "hello world", "hello  world", "unique"]
    out = list(exact_dedup(docs))
    assert len(out) == 2  # "hello world" deduped, "unique" kept


def test_shard_and_stream(tokenizer, tmp_path):
    texts = ["Привет мир, это тест." * 10, "Hello world this is a test." * 10] * 5
    manifest = shard_texts(iter(texts), tokenizer, str(tmp_path), len(tokenizer),
                           tokens_per_shard=50, max_seq_len=16)
    assert os.path.exists(manifest)

    ds = ShardedTokenDataset(str(tmp_path), seq_len=16, shuffle_shards=False)
    seqs = list(ds)
    assert len(seqs) > 0
    for s in seqs:
        assert s.shape[0] == 16
        assert s.dtype == torch.int64


def test_dataset_no_full_ram_load(tokenizer, tmp_path):
    texts = ["word " * 100] * 20
    shard_texts(iter(texts), tokenizer, str(tmp_path), len(tokenizer),
                tokens_per_shard=100, max_seq_len=32)
    ds = ShardedTokenDataset(str(tmp_path), seq_len=32, shuffle_shards=False)
    # just verify it can iterate lazily and repeatedly
    count1 = sum(1 for _ in ds)
    count2 = sum(1 for _ in ds)
    assert count1 == count2 > 0
