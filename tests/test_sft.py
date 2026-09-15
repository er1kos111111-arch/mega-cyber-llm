"""Tests for the SFT stage: chat tokenization, loss masking, dataset, trainer."""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.architecture import MCLLM
from data.sft_data import SFTDataset, collate_sft, text_to_messages, build_sft_messages


def test_special_tokens_encode_as_ids(tokenizer):
    ids = tokenizer.encode("<USER>Привет<ASSISTANT>")
    assert tokenizer.special_map["<USER>"] in ids
    assert tokenizer.special_map["<ASSISTANT>"] in ids
    # special token id appears exactly as a single token
    assert ids.count(tokenizer.special_map["<USER>"]) == 1


def test_special_tokens_roundtrip(tokenizer):
    text = "<USER>привет<ASSISTANT>Привет!"
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_tokenize_chat_structure(tokenizer):
    ids = tokenizer.tokenize_chat([{"role": "user", "content": "привет"},
                                   {"role": "assistant", "content": "Привет!"}],
                                  add_generation_prompt=False)
    assert ids[0] == tokenizer.bos_token_id
    assert tokenizer.special_map["<USER>"] in ids
    assert tokenizer.special_map["<ASSISTANT>"] in ids
    assert ids[-1] == tokenizer.eos_token_id


def test_tokenize_chat_with_labels_masks_user(tokenizer):
    ids, labels = tokenizer.tokenize_chat_with_labels([
        {"role": "user", "content": "привет"},
        {"role": "assistant", "content": "Привет!"},
    ])
    assert len(ids) == len(labels)
    # assistant content tokens must be supervised
    asst_idx = ids.index(tokenizer.special_map["<ASSISTANT>"])
    assert any(l != -100 for l in labels[asst_idx:])
    # user content tokens must be masked
    user_idx = ids.index(tokenizer.special_map["<USER>"])
    assert all(l == -100 for l in labels[user_idx:asst_idx])


def test_text_to_messages():
    msgs = text_to_messages("USER: привет\n\nASSISTANT: Привет! Как дела?")
    assert msgs[0] == {"role": "user", "content": "привет"}
    assert msgs[1]["role"] == "assistant"


def test_sft_dataset_and_collate(tokenizer):
    messages = [{"role": "user", "content": "привет"},
                {"role": "assistant", "content": "Привет!"}]
    ds = SFTDataset([messages], tokenizer, max_length=64)
    assert len(ds) == 1
    ids, labels = ds[0]
    assert ids.dtype == torch.long
    x, y = collate_sft([(ids, labels), (ids, labels)])
    assert x.shape[0] == 2 and y.shape[0] == 2


def test_build_sft_messages_synthetic():
    msgs = build_sft_messages(synthetic_n=50, persona_max_rows=0, seed=0)
    assert len(msgs) >= 40
    assert all(any(m["role"] == "assistant" for m in msgs[i]) for i in range(len(msgs)))


def test_sft_train_step(tokenizer, tiny_config):
    torch.manual_seed(0)
    model = MCLLM(tiny_config)
    messages = build_sft_messages(synthetic_n=30, persona_max_rows=0, seed=1)
    from post_training.sft import sft_train
    summary = sft_train(model, tokenizer, messages, out_dir=str(
        __import__("tempfile").mkdtemp()), epochs=1, batch_size=4, max_length=64,
        device="cpu", log_every=100)
    assert summary["sft_loss"] > 0
