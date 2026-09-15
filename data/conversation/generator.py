"""Synthetic multi-turn conversational dialogue generator for MC-LLM.

Produces natural, diverse, everyday-life conversations (Russian primary,
some English/mixed) in the USER / ASSISTANT format.  No facts, no knowledge,
no technical QA — only human conversation.

Diversity comes from combining:

* 13 user personas x 10 assistant styles;
* 24 everyday topics with statement/question/response banks;
* emotions, fillers, incomplete messages, typos, topic changes, topic
  returns, and in-dialogue memory (referencing earlier messages);
* a configurable length distribution (very short → very long).

Each dialogue carries a ``signature`` (the sequence of templates used) so the
deduplication layer can remove template-level duplicates.
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from . import phrases as P

# length distribution: (bucket, min_messages, max_messages, weight)
DEFAULT_LENGTH_DIST = [
    ("very_short", 2, 4, 0.10),
    ("short", 5, 9, 0.20),
    ("medium", 10, 29, 0.30),
    ("long", 30, 60, 0.25),
    ("very_long", 61, 120, 0.15),
]

_SLOT_RE = re.compile(r"\{(\w+)\}")

# minimal English material for mixed / occasional English dialogues
EN_GREETINGS = ["hi", "hey", "hello", "hi there"]
EN_MIX_WORDS = ["by the way", "honestly", "actually", "you know", "I guess",
                "so", "really", "kinda", "sort of", "anyway"]
EN_SMALL = {
    "user": ["hi", "hey, how are you?", "I'm a bit tired today", "what are you up to?",
             "I had a long day", "feeling kinda bored", "guess I'll just rest"],
    "assistant": ["Hey! How's it going?", "I'm here, what's up?", "Sorry to hear that.",
                  "Sounds like you need a break.", "Anything fun planned?"],
}

PARTIAL_RESPONSES = [
    "Да, я слушаю.", "Продолжай, я внимательно слушаю.", "Да-да, я здесь.",
    "Не торопись, я никуда не ухожу.", "Слушаю тебя.", "Конечно, рассказывай.",
]

NEG_MARKERS = ["плох", "груст", "скуч", "устал", "устало", "тяжел", "вымотал",
               "не могу", "не хочу", "поругал", "поссорил", "заболел", "болит",
               "не зашло", "стресс", "паршив", "тоскл", "неважно", "разочарован",
               "тошн", "скучно", "неприятно", "надоел", "злюсь", "раздража"]
POS_MARKERS = ["отлич", "здоров", "прекрасн", "супер", "круто", "классн",
               "радост", "весело", "люблю", "нравит", "выспал", "рад", "доволен",
               "хорош", "зашибись", "кайф", "счаст", "понравил"]


@dataclass
class GenConfig:
    length_dist: List[Tuple[str, int, int, float]] = field(
        default_factory=lambda: list(DEFAULT_LENGTH_DIST))
    topic_change_prob: float = 0.18
    topic_return_prob: float = 0.08
    memory_ref_prob: float = 0.10
    partial_prob: float = 0.06
    filler_prob: float = 0.08
    english_prob: float = 0.03
    mixed_prob: float = 0.06
    farewell_prob: float = 0.65
    typo_level: float = 0.5  # 0..1 overall informality


class DialogueGenerator:
    def __init__(self, seed: int = 0, config: Optional[GenConfig] = None):
        self.rng = random.Random(seed)
        self.config = config or GenConfig()
        self._topic_names = list(P.TOPICS.keys())

    # ------------------------------------------------------------------
    def _pick_length(self) -> Tuple[str, int]:
        buckets = self.config.length_dist
        r = self.rng.random()
        acc = 0.0
        for name, lo, hi, w in buckets:
            acc += w
            if r < acc:
                n = self.rng.randint(lo, hi)
                return name, max(2, n if n % 2 == 0 else n + 1)
        name, lo, hi, _ = buckets[-1]
        return name, self.rng.randint(lo, hi)

    # ------------------------------------------------------------------
    def _make_memory(self, name: str) -> Dict[str, str]:
        hobby = self.rng.choice(P.HOBBIES)
        return {
            "name": name,
            "hobby": hobby,
            "hobbies": self.rng.sample(P.HOBBIES, k=min(3, len(P.HOBBIES))),
            "game": self.rng.choice(P.GAMES),
            "movie": self.rng.choice(P.MOVIES),
            "music": self.rng.choice(P.MUSIC_GENRES),
            "food": self.rng.choice(P.FOODS),
            "activity": self.rng.choice(P.ACTIVITIES),
            "weather": self.rng.choice(P.WEATHER),
            "pet": self.rng.choice(["кот", "пёс", "хомяк", "попугай", "кошка", "собака"]),
            "emo_pos": self.rng.choice(P.POSITIVE_EMOTIONS),
            "emo_neg": self.rng.choice(P.NEGATIVE_EMOTIONS),
        }

    def _fill(self, template: str, memory: Dict[str, str], negative: bool = False) -> str:
        def rep(m):
            key = m.group(1)
            if key == "emo":
                return memory["emo_neg"] if negative else memory["emo_pos"]
            return memory.get(key, memory.get("hobby", "…"))
        return _SLOT_RE.sub(rep, template)

    # ------------------------------------------------------------------
    def _colloquialize(self, text: str, formal: float) -> str:
        """Lower formality => more lowercase / typos / contractions."""
        if text.isupper() or not text:
            return text
        # randomly lowercase the first letter
        if self.rng.random() > formal + 0.3:
            text = text[0].lower() + text[1:]
        # apply a typo with probability tied to informality
        if self.rng.random() < (1.0 - formal) * self.config.typo_level:
            for good, bad in P.TYPOS.items():
                if good in text and self.rng.random() < 0.5:
                    text = text.replace(good, bad, 1)
                    break
        return text

    # ------------------------------------------------------------------
    def _pick(self, options: List[str], avoid: List[str]) -> str:
        """Pick an option not in ``avoid`` (recent templates); fallback to any."""
        fresh = [o for o in options if o not in avoid]
        pool = fresh or list(options)
        return self.rng.choice(pool)

    def _user_utterance(self, topic: str, persona: str, memory: Dict[str, str],
                        formal: float, negative: bool,
                        recent_user: List[str]) -> Tuple[str, str]:
        tpl = self._pick(P.TOPICS[topic]["user"], recent_user)
        text = self._fill(tpl, memory, negative=negative)
        text = self._colloquialize(text, formal)
        return text, tpl

    def _assistant_utterance(self, topic: str, style: str, memory: Dict[str, str],
                             user_text: str, recent_asst: List[str]) -> Tuple[str, str]:
        bank = P.TOPICS[topic]
        neg = any(w in user_text for w in NEG_MARKERS)
        pos = any(w in user_text for w in POS_MARKERS)

        # 1. question (always fits the topic)
        if (style == "with_question" and self.rng.random() < 0.7) or self.rng.random() < 0.32:
            if "questions" in bank:
                tpl = self._pick(bank["questions"], recent_asst)
                return self._fill(tpl, memory), tpl

        # 2. polarity-aware reactions
        if style == "supportive" or neg:
            tpl = self._pick(P.SUPPORT, recent_asst)
        elif style == "short":
            tpl = self._pick(P.REACTIONS, recent_asst)
        elif pos and self.rng.random() < 0.6:
            tpl = self._pick(P.POSITIVE_REACTIONS, recent_asst)
        elif style == "detailed" and self.rng.random() < 0.5:
            tpl = self._pick(bank["assistant"], recent_asst)
        else:
            tpl = self._pick(P.REACTIONS, recent_asst)

        text = self._fill(tpl, memory, negative=neg)
        return text, tpl

    # ------------------------------------------------------------------
    def generate(self) -> Dict:
        cfg = self.config
        persona = self.rng.choice(P.PERSONAS)
        style = self.rng.choice(P.ASSISTANT_STYLES)
        traits = P.PERSONA_TRAITS[persona]
        formal = traits.get("formal", 0.5)

        # language mode
        lang = "ru"
        if self.rng.random() < cfg.english_prob:
            lang = "en"
        elif self.rng.random() < cfg.mixed_prob:
            lang = "mixed"

        name = self.rng.choice(P.NAMES)
        memory = self._make_memory(name)
        bucket, num_turns = self._pick_length()

        topic = self.rng.choice(self._topic_names)
        turns: List[str] = []
        signature: List[str] = []

        if lang == "en":
            turns, signature = self._gen_english(num_turns)
        else:
            turns, signature = self._gen_ru(num_turns, topic, persona, style,
                                            traits, formal, memory, lang)

        # assemble the final text
        text = self._assemble(turns)

        return {
            "text": text,
            "turns": len(turns),
            "lang": lang,
            "persona": persona,
            "style": style,
            "bucket": bucket,
            "signature": tuple(signature),
        }

    # ------------------------------------------------------------------
    def _gen_ru(self, num_turns, topic, persona, style, traits, formal, memory, lang):
        cfg = self.config
        turns: List[str] = []
        signature: List[str] = []
        negative = persona in ("tired", "emotional", "insecure")
        recent_user: List[str] = []
        recent_asst: List[str] = []

        # open with a greeting most of the time
        start_greeting = self.rng.random() < 0.8
        if start_greeting:
            g = self._fill(self.rng.choice(P.GREETINGS_USER), memory)
            turns.append(self._colloquialize(g, formal))
            signature.append("greet_user")
            a = self._fill(self.rng.choice(P.GREETINGS_ASSISTANT), memory)
            turns.append(a)
            signature.append("greet_asst")
            idx = 2
        else:
            idx = 0

        while idx < num_turns:
            # organic topic drift (not only via explicit change-openers)
            if self.rng.random() < 0.10:
                topic = self.rng.choice(self._topic_names)

            # user turn
            u_tpl = "filler"
            u_text = None
            r = self.rng.random()
            if r < cfg.partial_prob:
                u_text = self.rng.choice(P.PARTIAL_MESSAGES)
                u_tpl = "partial"
            elif r < cfg.partial_prob + cfg.filler_prob:
                u_text = self.rng.choice(P.FILLERS)
                u_tpl = "filler"
            elif r < cfg.partial_prob + cfg.filler_prob + cfg.topic_change_prob:
                tpl = self._pick(P.TOPIC_CHANGE_OPENERS, recent_user)
                u_text = self._fill(tpl, memory)
                u_tpl = tpl
                topic = self.rng.choice(self._topic_names)
            elif r < cfg.partial_prob + cfg.filler_prob + cfg.topic_change_prob + cfg.topic_return_prob:
                tpl = self._pick(P.TOPIC_RETURN_OPENERS, recent_user)
                u_text = self._fill(tpl, memory)
                u_tpl = tpl
            elif r < cfg.partial_prob + cfg.filler_prob + cfg.topic_change_prob + cfg.topic_return_prob + cfg.memory_ref_prob:
                tpl = self._pick(P.MEMORY_REF, recent_user)
                u_text = self._fill(tpl, memory)
                u_tpl = tpl
            else:
                u_text, u_tpl = self._user_utterance(topic, persona, memory, formal,
                                                     negative, recent_user)

            # mixed-language sprinkle
            if lang == "mixed" and self.rng.random() < 0.25:
                u_text = u_text + " " + self.rng.choice(EN_MIX_WORDS)
            u_text = self._colloquialize(u_text, formal) if u_tpl not in ("partial", "filler") else u_text
            turns.append(u_text)
            signature.append(u_tpl)
            recent_user = (recent_user + [u_tpl])[-4:]
            idx += 1

            if idx >= num_turns:
                break

            # assistant turn
            if u_tpl in ("partial", "filler"):
                a_text = self._pick(PARTIAL_RESPONSES, recent_asst)
                a_tpl = "partial_resp"
            else:
                a_text, a_tpl = self._assistant_utterance(topic, style, memory, u_text,
                                                          recent_asst)
            turns.append(a_text)
            signature.append(a_tpl)
            recent_asst = (recent_asst + [a_tpl])[-4:]
            idx += 1

        # farewell tail (unless already very short)
        if num_turns > 6 and self.rng.random() < cfg.farewell_prob:
            if turns[-1] not in P.FAREWELLS_ASSISTANT and turns[-1] not in P.FAREWELLS_USER:
                f_user = self.rng.choice(P.FAREWELLS_USER)
                turns.append(self._colloquialize(f_user, formal))
                signature.append("farewell_user")
                f_asst = self.rng.choice(P.FAREWELLS_ASSISTANT)
                turns.append(f_asst)
                signature.append("farewell_asst")

        return turns, signature

    def _gen_english(self, num_turns):
        turns: List[str] = []
        signature: List[str] = []
        for i in range(num_turns):
            if i % 2 == 0:
                turns.append(self.rng.choice(EN_SMALL["user"]))
                signature.append("en_user")
            else:
                turns.append(self.rng.choice(EN_SMALL["assistant"]))
                signature.append("en_asst")
        return turns, signature

    def _assemble(self, turns: List[str]) -> str:
        parts = []
        for i, t in enumerate(turns):
            role = "USER" if i % 2 == 0 else "ASSISTANT"
            parts.append(f"{role}: {t}")
        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    def dump_phrase_text(self) -> str:
        """Concatenate all phrase-bank content (for training the tokenizer)."""
        chunks = []
        chunks.extend(P.NAMES)
        chunks.extend(P.HOBBIES + P.GAMES + P.MOVIES + P.MUSIC_GENRES + P.FOODS)
        chunks.extend(P.ACTIVITIES + P.WEATHER)
        chunks.extend(P.POSITIVE_EMOTIONS + P.NEGATIVE_EMOTIONS + P.FILLERS)
        chunks.extend(P.PARTIAL_MESSAGES + P.GREETINGS_USER + P.GREETINGS_ASSISTANT)
        chunks.extend(P.FAREWELLS_USER + P.FAREWELLS_ASSISTANT + P.REACTIONS)
        chunks.extend(P.AGREEMENTS + P.DISAGREEMENTS + P.SUPPORT)
        chunks.extend(P.TOPIC_CHANGE_OPENERS + P.TOPIC_RETURN_OPENERS + P.MEMORY_REF)
        chunks.extend(PARTIAL_RESPONSES + EN_GREETINGS + EN_MIX_WORDS)
        chunks.extend(EN_SMALL["user"] + EN_SMALL["assistant"])
        for topic in P.TOPICS.values():
            for key in ("user", "assistant", "questions"):
                chunks.extend(topic.get(key, []))
        return "\n".join(chunks)
