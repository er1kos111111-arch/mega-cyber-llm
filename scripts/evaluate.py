"""Evaluate MC-LLM CLI wrapper (delegates to evaluation.evaluate)."""
from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.evaluate import main

if __name__ == "__main__":
    main()
