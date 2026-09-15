"""Tokenizer tests: roundtrip, languages, unicode, emoji, code, JSON, long text."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tokenizer.tokenizer import CyberTokenizer


def test_vocab_size(tokenizer):
    assert len(tokenizer) >= 256 + 12
    assert tokenizer.eos_token_id == 258
    assert tokenizer.bos_token_id == 257
    assert tokenizer.pad_token_id == 256


def test_special_tokens_present(tokenizer):
    for tok in ["<PAD>", "<BOS>", "<EOS>", "<UNK>", "<USER>", "<ASSISTANT>",
                "<SYSTEM>", "<TOOL>", "<THINK>", "<END_THINK>", "<CODE>", "<END_CODE>"]:
        assert tok in tokenizer.special_map


def test_roundtrip_russian(tokenizer):
    text = "Привет, как дела? Это проверка русского текста."
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_roundtrip_english(tokenizer):
    text = "Hello world, this is an English test string!"
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_roundtrip_mixed(tokenizer):
    text = "Смешанный mixed текст with English words and numbers 12345."
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_roundtrip_unicode(tokenizer):
    text = "Unicode: é è ü ñ ç ß Ω ∑ √ ≈"
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_roundtrip_emoji(tokenizer):
    text = "Emoji test 🔥 🚀 😀 🎉"
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_roundtrip_code(tokenizer):
    text = "def add(a, b):\n    return a + b\n"
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_roundtrip_json(tokenizer):
    text = '{"name": "test", "values": [1, 2, 3], "nested": {"ok": true}}'
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_roundtrip_long_text(tokenizer):
    text = ("Длинный текст для проверки токенизации. " * 50)
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_unknown_symbols_lossless(tokenizer):
    # byte-level BPE is lossless even for "unknown" symbols
    text = "Symbols: ☃ ☂ ✈ ♥ \x00\xff rare bytes"
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_encode_returns_ids(tokenizer):
    ids = tokenizer.encode("hello")
    assert isinstance(ids, list)
    assert all(isinstance(i, int) for i in ids)


def test_bpe_compresses_common_words(tokenizer):
    # a common repeated word should encode to fewer tokens than its length
    ids = tokenizer.encode("hello hello hello")
    assert len(ids) < len("hello hello hello")


def test_empty_string(tokenizer):
    assert tokenizer.encode("") == []
    assert tokenizer.decode([]) == ""


def test_tokenize_with_bos_eos(tokenizer):
    text = "hello"
    ids = tokenizer.tokenize(text, add_bos=True, add_eos=True)
    assert ids[0] == tokenizer.bos_token_id
    assert ids[-1] == tokenizer.eos_token_id


def test_pad_and_truncate(tokenizer):
    ids = tokenizer.encode("hello")
    padded = tokenizer.pad(ids, 20)
    assert len(padded) == 20
    assert padded[:len(ids)] == ids
    truncated = tokenizer.pad(ids, 3)
    assert len(truncated) == 3


def test_chat_template(tokenizer):
    s = tokenizer.apply_chat_template([{"role": "user", "content": "Привет"}])
    assert "<USER>" in s and "Привет" in s and "<ASSISTANT>" in s


def test_encode_cache_consistency(tokenizer):
    a = tokenizer.encode("hello")
    b = tokenizer.encode("hello")  # hits cache
    assert a == b
