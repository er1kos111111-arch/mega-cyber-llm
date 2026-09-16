# Control experiments for MC-LLM quality

Three configurations to compare.  Run each with `scripts/colab_train.py`
(conversational path), then measure with `scripts/quality_report.py` and
compare validation loss / perplexity / generation quality / repetition.

## Experiment A — baseline

The original setup (small data, short training, small vocab):

```bash
python scripts/colab_train.py --conv-tokens 10000000 --vocab-size 4096 \
    --steps 1500 --batch 8 --seq-len 512 --sft --sft-synthetic 20000
```

## Experiment B — improved

More data + more steps + higher vocab (this is the recommended current setup):

```bash
python scripts/colab_train.py --conv-tokens 50000000 --vocab-size 8192 \
    --steps 15000 --batch 8 --seq-len 512 --sft --sft-synthetic 30000
```

## Experiment C — optimized

Longer context + higher learning rate + more SFT diversity:

```bash
python scripts/colab_train.py --conv-tokens 100000000 --vocab-size 8192 \
    --steps 30000 --batch 8 --seq-len 1024 --lr 1e-3 \
    --sft --sft-synthetic 50000
```

## What to record for each

| Metric | How |
|--------|-----|
| train loss / val loss / ppl | training log |
| generation quality | `python scripts/quality_report.py --checkpoint checkpoints_sft` |
| tokenizer efficiency | printed at dataset build (`tokens/word`, compression) |

Pick the config by **measured** generation quality, not by assumption.
The key knobs are: data volume (`--conv-tokens`), steps (`--steps`), vocab
(`--vocab-size`), sequence length (`--seq-len`), and learning rate (`--lr`).
