from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

from .llm import GenerationParams, LocalLLM
from .state import ThoughtVertex


@dataclass
class CriticConfig:
    enabled: bool = True
    weight: float = 0.35
    prompt: str = (
        "You are a brutally honest comedy coach. Score the following stand-up line on a 0-10 scale "
        "for wit, structure, and callback potential. Return ONLY a JSON object like "
        '{"score": 7.5, "notes": "..."} .'
    )


class ComedyValidator:
    """Assigns scores and diagnostics to each candidate vertex."""

    def __init__(self, topics: Sequence[str], critic_llm: LocalLLM | None = None, critic_cfg: CriticConfig | None = None):
        self.topics = [t.lower() for t in topics]
        self.critic_llm = critic_llm
        self.critic_cfg = critic_cfg or CriticConfig(enabled=False)

    def score_candidate(
        self,
        text: str,
        role: str,
        parents: List[ThoughtVertex],
        expected_role: str,
        callbacks: Sequence[str],
    ) -> tuple[float, Dict]:
        heuristics = {
            "structure": self._structure_score(text),
            "callback": self._callback_score(text, parents, callbacks),
            "topic_alignment": self._topic_alignment(text),
            "role_bonus": 0.15 if role == expected_role else 0.05,
        }
        critic_score = self._critic_score(text) if self.critic_cfg.enabled and self.critic_llm else 0.0
        heuristics["critic"] = critic_score

        aggregate = (
            0.3 * heuristics["structure"]
            + 0.25 * heuristics["callback"]
            + 0.2 * heuristics["topic_alignment"]
            + heuristics["role_bonus"]
            + self.critic_cfg.weight * critic_score
        )
        return float(max(0.0, min(1.0, aggregate))), heuristics

    def _structure_score(self, text: str) -> float:
        length = len(text)
        if length < 60:
            return 0.2
        if length > 320:
            return 0.4
        return min(1.0, math.log(length) / 5)

    def _callback_score(self, text: str, parents: List[ThoughtVertex], callbacks: Sequence[str]) -> float:
        if not parents:
            return 0.3
        keywords = set()
        for parent in parents[-3:]:
            keywords.update(self._keywords(parent.text))
        overlap = keywords.intersection(self._keywords(text))
        bonus = 0.15 * len(callbacks)
        return max(0.2, min(1.0, (len(overlap) / (len(keywords) + 1e-5)) + bonus))

    def _topic_alignment(self, text: str) -> float:
        text_lower = text.lower()
        hits = sum(1 for topic in self.topics if topic in text_lower)
        return min(1.0, 0.2 + 0.2 * hits)

    def _critic_score(self, text: str) -> float:
        params = GenerationParams(max_new_tokens=80, temperature=0.0, top_p=0.9)
        messages = [
            {"role": "system", "content": self.critic_cfg.prompt},
            {"role": "user", "content": text},
        ]
        try:
            reply = self.critic_llm.generate_chat(messages, params)
            match = re.search(r"([\d.]+)", reply)
            if match:
                score = float(match.group(1))
                return max(0.0, min(1.0, score / 10.0))
        except Exception:
            return 0.0
        return 0.0

    @staticmethod
    def _keywords(text: str) -> set[str]:
        tokens = re.findall(r"[a-zA-Z']+", text.lower())
        return {tok for tok in tokens if len(tok) > 3}
