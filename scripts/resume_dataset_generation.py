"""Resume conversational dataset generation after a Colab restart.

Determines the last complete shard and continues from the next index, without
losing already-generated data.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.conversation.generate_dataset import main

if __name__ == "__main__":
    main()
