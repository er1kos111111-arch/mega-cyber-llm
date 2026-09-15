"""Built-in benchmark suites for MC-LLM.

These are lightweight, self-contained evaluation tasks (no external
downloads).  They assess basic language modeling, instruction following,
reasoning, arithmetic, and multilingual generation.

Each benchmark is a list of ``{"prompt": str, "answers": [str], "type": str}``
entries.  Scoring is a fuzzy substring/answer check — a proxy, not a
replacement for full downstream evals like MMLU/HumanEval.
"""
from __future__ import annotations

from typing import Dict, List


ARITHMETIC = [
    {"prompt": "2 + 2 =", "answers": ["4"], "type": "math"},
    {"prompt": "10 - 3 =", "answers": ["7"], "type": "math"},
    {"prompt": "6 * 7 =", "answers": ["42"], "type": "math"},
    {"prompt": "Сколько будет 5 + 5?", "answers": ["10"], "type": "math"},
    {"prompt": "What is 12 / 3?", "answers": ["4"], "type": "math"},
]

INSTRUCTION_FOLLOWING = [
    {"prompt": "Напиши ответ одним словом: столица России?", "answers": ["Москва", "москва"], "type": "instruction"},
    {"prompt": "Ответь только словом: 2+2 равно", "answers": ["4"], "type": "instruction"},
    {"prompt": "Say the single word: color of the sky", "answers": ["blue", "Blue"], "type": "instruction"},
]

REASONING = [
    {"prompt": "У Маши было 3 яблока, ей дали ещё 2. Сколько яблок стало?", "answers": ["5"], "type": "reasoning"},
    {"prompt": "If a train travels 60 km in 1 hour, how far in 2 hours?", "answers": ["120"], "type": "reasoning"},
]

MULTILINGUAL = [
    {"prompt": "Переведи на английский: привет", "answers": ["hello", "hi"], "type": "multilingual"},
    {"prompt": "Translate to Russian: goodbye", "answers": ["пока", "до свидания", "прощай"], "type": "multilingual"},
]

CONVERSATION = [
    {"prompt": "Привет! Как дела?", "answers": [], "type": "conversation"},
    {"prompt": "Hello! How are you?", "answers": [], "type": "conversation"},
]

PROGRAMMING = [
    {"prompt": "Напиши функцию на Python, которая возвращает сумму двух чисел.", "answers": ["def", "return"], "type": "code"},
    {"prompt": "Write a Python function that returns the square of x.", "answers": ["def", "return"], "type": "code"},
]


def get_benchmark(name: str) -> List[Dict]:
    suites = {
        "arithmetic": ARITHMETIC,
        "instruction": INSTRUCTION_FOLLOWING,
        "reasoning": REASONING,
        "multilingual": MULTILINGUAL,
        "conversation": CONVERSATION,
        "programming": PROGRAMMING,
    }
    if name == "all":
        out = []
        for v in suites.values():
            out.extend(v)
        return out
    return suites.get(name, [])


def list_benchmarks() -> List[str]:
    return ["arithmetic", "instruction", "reasoning", "multilingual",
            "conversation", "programming", "all"]
