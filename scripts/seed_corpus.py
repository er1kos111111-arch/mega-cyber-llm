"""Generate a small multilingual seed corpus for demo/tests.

This is *demo* data used to prove the end-to-end pipeline works without any
external download.  For real training, point the pipeline at your own
corpus (books, Common Crawl extracts, code, dialogues, etc.) in data/raw.
"""
from __future__ import annotations

import argparse
import itertools
import os
import random

RU_TEMPLATES = [
    "Привет! Как у тебя дела сегодня?",
    "Сегодня прекрасная погода, и мы решили пойти гулять в парк.",
    "Математика — это наука о числах, формах и закономерностях.",
    "Компьютер обрабатывает данные с помощью центрального процессора.",
    "Я читаю интересную книгу о путешествиях по всему миру.",
    "Программирование требует логического мышления и внимания к деталям.",
    "Вода закипает при температуре сто градусов по Цельсию.",
    "Мы изучаем историю древних цивилизаций и их достижения.",
    "Искусственный интеллект помогает людям решать сложные задачи.",
    "Музыка объединяет людей разных культур и поколений.",
]

EN_TEMPLATES = [
    "Hello! How are you doing today?",
    "The weather is beautiful today, so we decided to walk in the park.",
    "Mathematics is the science of numbers, shapes, and patterns.",
    "A computer processes data using its central processing unit.",
    "I am reading an interesting book about traveling around the world.",
    "Programming requires logical thinking and attention to detail.",
    "Water boils at one hundred degrees Celsius.",
    "We study the history of ancient civilizations and their achievements.",
    "Artificial intelligence helps people solve complex problems.",
    "Music brings together people of different cultures and generations.",
]

CODE_SNIPPETS = [
    "def add(a, b):\n    return a + b",
    "local function sum(a, b)\n    return a + b\nend",
    "const x = 10;\nconsole.log(x * 2);",
    '{"name": "test", "value": 42}',
    "import math\nprint(math.sqrt(16))",
    "# This is a comment\nfor i in range(10):\n    print(i)",
]

DIALOGUES = [
    "Пользователь: Привет\nАссистент: Здравствуйте! Чем могу помочь?",
    "User: What is the capital of France?\nAssistant: The capital of France is Paris.",
    "Пользователь: Напиши функцию сложения\nАссистент: def add(a, b): return a + b",
]

MATH = [
    "Два плюс два равно четыре.",
    "2 + 2 = 4, а 3 * 3 = 9.",
    "Десять минус три равно семь.",
    "Площадь круга вычисляется по формуле S = pi * r^2.",
]


def generate(num_docs: int, seed: int = 0):
    rng = random.Random(seed)
    sources = [RU_TEMPLATES, EN_TEMPLATES, CODE_SNIPPETS, DIALOGUES, MATH]
    weights = [0.40, 0.35, 0.10, 0.05, 0.10]

    for _ in range(num_docs):
        src = rng.choices(sources, weights=weights)[0]
        base = rng.choice(src)
        # produce slightly varied documents
        yield base + "\n"
        if rng.random() < 0.5:
            extra = rng.choice(RU_TEMPLATES + EN_TEMPLATES)
            yield f"{base} {extra}\n"


def main():
    parser = argparse.ArgumentParser(description="Generate demo corpus")
    parser.add_argument("--num-docs", type=int, default=2000)
    parser.add_argument("--out", default="data/raw")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, "seed_corpus.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        for doc in generate(args.num_docs, args.seed):
            f.write(doc)
            f.write("\n")
    print(f"[seed_corpus] wrote {args.num_docs} docs to {out_path}")


if __name__ == "__main__":
    main()
