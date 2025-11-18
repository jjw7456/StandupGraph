from __future__ import annotations

import json
import re
from typing import Any, Dict, List

from .operations import Operation


class ComedyParser:
    """Parses LLM replies for the GoT pipeline."""

    JSON_REGEX = re.compile(r"\{.*\}", re.DOTALL)

    def parse(self, operation: Operation, raw_text: str) -> List[Dict[str, Any]]:
        payload = self._extract_json(raw_text)
        if payload is None:
            payload = {"candidates": [{"line": raw_text.strip(), "role": operation.metadata.get("role", "line")}]}

        candidates = payload.get("candidates") or payload.get("thoughts") or payload
        if isinstance(candidates, dict):
            candidates = [candidates]

        parsed: List[Dict[str, Any]] = []
        for idx, entry in enumerate(candidates):
            if isinstance(entry, str):
                entry = {"line": entry}
            line = entry.get("line") or entry.get("text") or entry.get("content")
            if not line:
                continue

            parsed.append(
                {
                    "text": line.strip(),
                    "role": entry.get("role") or operation.metadata.get("role", f"line_{idx}"),
                    "confidence": float(entry.get("confidence", entry.get("score", 0.6))),
                    "callbacks": entry.get("callbacks", []),
                    "topics": entry.get("topics", []),
                    "notes": entry.get("notes") or payload.get("notes") or "",
                }
            )
        return parsed

    def _extract_json(self, raw_text: str) -> Dict | None:
        match = self.JSON_REGEX.search(raw_text.strip())
        if not match:
            return None
        snippet = match.group(0)
        try:
            return json.loads(snippet)
        except json.JSONDecodeError:
            return None
