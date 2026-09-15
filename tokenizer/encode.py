"""CLI / helper for encoding text with CyberTokenizer."""
from __future__ import annotations

import argparse

from .tokenizer import CyberTokenizer


def encode_text(tokenizer: CyberTokenizer, text: str) -> list:
    return tokenizer.encode(text)


def main() -> None:
    parser = argparse.ArgumentParser(description="Encode text with CyberTokenizer")
    parser.add_argument("--tokenizer", default="tokenizer/tokenizer_config.json")
    parser.add_argument("--text", required=True)
    parser.add_argument("--add_bos", action="store_true")
    parser.add_argument("--add_eos", action="store_true")
    args = parser.parse_args()

    tok = CyberTokenizer(args.tokenizer)
    ids = tok.encode(args.text)
    if args.add_bos:
        ids = [tok.bos_token_id] + ids
    if args.add_eos:
        ids = ids + [tok.eos_token_id]
    print(" ".join(str(i) for i in ids))


if __name__ == "__main__":
    main()
