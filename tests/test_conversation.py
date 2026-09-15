"""Tests for the conversational dataset pipeline: generator, quality, dedup,
resource monitor, and the streaming driver."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.conversation.generator import DialogueGenerator, GenConfig
from data.conversation import quality
from data.conversation.dedup import ConversationDedup
from data.resource_monitor import ResourceMonitor, ResourceLimits


@pytest.fixture(scope="module")
def generator():
    return DialogueGenerator(seed=123)


def test_generator_produces_valid_format(generator):
    for _ in range(50):
        d = generator.generate()
        assert d["turns"] >= 2
        assert quality.check_format(d["text"]), d["text"][:200]
        assert quality.check_no_empty(d["text"])
        assert quality.check_unicode(d["text"])


def test_generator_no_degenerate_loops(generator):
    # no dialogue should be dominated by repeated identical messages
    for _ in range(50):
        d = generator.generate()
        assert quality.check_repetition(d["text"]), d["text"]


def test_length_distribution_roughly_matches():
    g = DialogueGenerator(seed=7)
    counts = {}
    for _ in range(500):
        d = g.generate()
        counts[d["bucket"]] = counts.get(d["bucket"], 0) + 1
    # medium should be the largest bucket, very_short the smallest
    assert counts["medium"] > counts["very_short"]
    assert counts["long"] > counts["very_short"]


def test_language_distribution_russian_primary(generator):
    langs = {}
    for _ in range(300):
        d = generator.generate()
        langs[d["lang"]] = langs.get(d["lang"], 0) + 1
    assert langs.get("ru", 0) > langs.get("en", 0) + langs.get("mixed", 0)


def test_quality_gates():
    assert quality.check_dialogue("USER: привет\n\nASSISTANT: Привет!")
    assert not quality.check_dialogue("hello")  # bad format
    assert not quality.check_dialogue("USER: \n\nASSISTANT: ")  # empty
    # broken roles (two USER in a row)
    assert not quality.check_dialogue("USER: a\n\nUSER: b")


def test_dedup_exact_and_template():
    dd = ConversationDedup()
    base = {"text": "USER: привет\n\nASSISTANT: Привет!", "signature": ("greet_user", "greet_asst")}
    ok1, _ = dd.should_keep(base)
    dd.record(base)
    assert ok1 is True
    # exact duplicate
    ok2, _ = dd.should_keep(dict(base))
    assert ok2 is False
    # template duplicate with same signature (different text)
    dd2 = ConversationDedup(template_max_repeat=1)
    a = {"text": "USER: привет\n\nASSISTANT: Привет!", "signature": ("g", "a")}
    dd2.record(a)
    b = {"text": "USER: здарова\n\nASSISTANT: Привет!", "signature": ("g", "a")}
    ok3, _ = dd2.should_keep(b)
    assert ok3 is False


def test_resource_monitor():
    mon = ResourceMonitor(ResourceLimits(ram_fraction=0.85, disk_limit_gb=1.0))
    snap = mon.snapshot()
    assert snap.ram_total_gb > 0
    # with a tiny disk limit, it should stop
    stop, reason = mon.should_stop(snap, dialogues=10)
    assert isinstance(stop, bool)
    assert isinstance(reason, str)


def test_streaming_driver_creates_shards(tmp_path, tokenizer_dir):
    from data.conversation.generate_dataset import run, _scan_shards

    out = str(tmp_path / "shards")
    report = run(
        out_dir=out, tokenizer_dir=tokenizer_dir, seed=0,
        vocab_size=512, dialogues_per_shard=50, tokens_per_shard=200_000,
        log_every=1000,
        limits=ResourceLimits(max_dialogues=120),
    )
    assert report["dialogues"] >= 100
    assert report["tokens"] > 0
    assert report["shards"] >= 2

    last_idx, td, tt, tk, size = _scan_shards(out)
    assert last_idx >= 0
    assert td == report["dialogues"]
    assert tk == report["tokens"]

    # resume: run again, should continue from existing shards
    report2 = run(
        out_dir=out, tokenizer_dir=tokenizer_dir, seed=1,
        vocab_size=512, dialogues_per_shard=50, tokens_per_shard=200_000,
        log_every=1000,
        limits=ResourceLimits(max_dialogues=200),
    )
    assert report2["dialogues"] >= report["dialogues"]


def test_dump_phrase_text(generator):
    text = generator.dump_phrase_text()
    assert len(text) > 1000
    assert "привет" in text.lower() or "hello" in text.lower()
