"""CLI / helper for decoding token IDs with CyberTokenizer."""
from __future__ import annotations

import argparse

from .tokenizer import CyberTokenizer


def decode_ids(tokenizer: CyberTokenizer, ids: list) -> str:
    return tokenizer.decode(ids)


def main() -> None:
    parser = argparse.ArgumentParser(description="Decode token IDs with CyberTokenizer")
    parser.add_argument("--tokenizer", default="tokenizer/tokenizer_config.json")
    parser.add_argument("--ids", required=True, help="space-separated token ids")
    args = parser.parse_args()

    tok = CyberTokenizer(args.tokenizer)
    ids = [int(x) for x in args.ids.split()]
    print(tok.decode(ids))


if __name__ == "__main__":
    main()
