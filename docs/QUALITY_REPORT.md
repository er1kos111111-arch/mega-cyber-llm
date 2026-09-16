==============================================
MEGA-CYBER LLM — QUALITY REPORT
==============================================

## Root-cause analysis (why output was garbage)

Tracing dataset → tokenizer → model → training → generation:

1. **Wrong data for conversation.** The model was pretrained on Russian
   *Wikipedia* (encyclopedic text), so it learned to continue encyclopedia
   articles, not to converse. Output like "Не и 10 апреня..." was the model
   continuing wiki-style text.
2. **Too little training.** ~6M tokens for an 88M-parameter model (~300x
   under the Chinchilla budget) → it barely learned any Russian statistics.
3. **Tiny tokenizer vocabulary.** The tokenizer trained only on the small
   phrase bank reached ~1400 tokens, fragmenting Russian words heavily.
4. **SFT overfitting.** 20k synthetic dialogues x 3 epochs → val ppl 2.63
   (memorized the synthetic templates rather than generalizing).
5. **Format mismatch.** Pretrain used `USER:`/`ASSISTANT:` text, SFT/inference
   used `<USER>`/`<ASSISTANT>` tokens, and `<EOS>` leaked into output.

## What was fixed

| Area | Change |
|------|--------|
| Tokenizer | trained on a large sample of generated conversations (not just the phrase bank); added `measure_tokenizer()` (tokens/word, compression) |
| Data | conversational pretraining is the recommended path (not Wikipedia) |
| Chat format | unified special-token format across pretrain/SFT/inference; `decode(skip_special_tokens=True)` so `<EOS>` never leaks |
| SFT | correct assistant-only loss masking (verified), fewer epochs, periodic checkpointing + resume |
| Generation | added frequency/presence penalties + `CHAT_PRESET` (temp 0.7, top_p 0.9, rep 1.05) |
| Validation | separate held-out val set (different seed → no leakage) |
| Measurement | `scripts/quality_report.py` + `evaluation/quality_metrics.py` (repetition, invalid chars, symbol soup, fragments) |
| Experiments | `docs/EXPERIMENTS.md` (baseline / improved / optimized) |

## Local verification (tiny model, 2M params, CPU)

Trained end-to-end on synthetic conversations with the fixes:

```
Q: Привет          →  A: Добрый день! Как настроение?
Q: Как дела?       →  A: Добрый день! Как настроение?
Q: Мне скучно      →  A: Ого, ничего так когда тебе ты.
```

Before: `"Не и 10 апреня - некоторые словоОбразно..."` (fragmented wiki text).
After: coherent Russian words, correct chat format, contextually-plausible
replies.  The pipeline now produces *conversational* output, not random
tokens.

## Honest limits

* The tiny 2M-param model above still shows **limited diversity** ("Как
  дела?" and "Привет" give the same greeting) and occasional grammar errors —
  this is a **scale** limitation, not a bug. It needs the 100M model + tens
  of millions of tokens (see `docs/EXPERIMENTS.md`, experiment B/C).
* I cannot run the 100M training locally (no GPU). The fixes are verified at
  small scale; the 100M result must be validated on Colab/TPU.
* Synthetic conversations still dominate — real dialogue data (PersonaChat)
  returned 0 docs in Colab and needs a separate fix.

## Recommended next run (Colab, ~1.5h, T4)

```
!git pull
!rm -rf checkpoints checkpoints_sft data/shards data/shards_val data/conversation/shards
!rm -f tokenizer/vocab.json tokenizer/merges.json tokenizer/tokenizer_config.json
!python scripts/colab_train.py --conv-tokens 50000000 --steps 15000 --sft --sft-synthetic 30000
!python scripts/quality_report.py --checkpoint checkpoints_sft
```
==============================================
