"""CLI entry point for conversational dataset generation.

    python scripts/generate_conversation_dataset.py --max-tokens 5000000

Resuming after a Colab restart is automatic: the driver scans existing
shards and continues from the last complete one.  For an explicit resume
run, the same command can be reused.

    python scripts/resume_dataset_generation.py --max-tokens 10000000
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.conversation.generate_dataset import main

if __name__ == "__main__":
    main()
