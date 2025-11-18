#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Utility script that converts raw stand-up comedy transcripts into the
Harmony-style `messages` column expected by TRL's `SFTTrainer`.

Features:
    * Accepts JSON/JSONL/CSV files or an entire directory of such files.
    * Cleans + chunks transcripts into short Graph-of-Thought aware comedy beats.
    * Emits a Parquet file with a `messages` column plus light-weight metadata.
    * Ships with `--demo_samples` so the full pipeline can be smoke-tested without
      any private comedy catalogs.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd


DEFAULT_OUTPUT = Path("prep_out/comedy_messages.parquet")
DEFAULT_SYSTEM_TEMPLATE = (
    "You are a seasoned stand-up comic who writes polished sets with setups, misdirections, "
    "layered punchlines, and callbacks. You reason explicitly about stage dynamics using a "
    "graph of thoughts: vertices are candidate lines, and weighted edges reflect narrative flow."
)


@dataclass
class RawBit:
    """Container describing a single comedy bit/transcript."""

    title: str
    topic: str
    persona: str
    audience: str
    style: str
    transcript: str
    callbacks: Optional[str] = None
    source: Optional[str] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare stand-up comedy data for SFT fine-tuning.")
    parser.add_argument(
        "--input_path",
        type=Path,
        default=None,
        help="File or directory that stores raw comedy transcripts. Supported extensions: .json, .jsonl, .csv",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Destination Parquet path (directories are created automatically).",
    )
    parser.add_argument(
        "--max_lines_per_chunk",
        type=int,
        default=6,
        help="Upper bound on the number of transcript lines kept inside a single conversation chunk.",
    )
    parser.add_argument(
        "--min_characters",
        type=int,
        default=180,
        help="Filter out chunks shorter than this many characters.",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle the processed dataset before saving.",
    )
    parser.add_argument(
        "--demo_samples",
        action="store_true",
        help="Ignore --input_path and use built-in sample jokes. Useful for smoke-tests or CI.",
    )
    return parser.parse_args()


def load_records(args: argparse.Namespace) -> List[RawBit]:
    if args.demo_samples or args.input_path is None:
        return demo_records()

    path = args.input_path
    if not path.exists():
        raise FileNotFoundError(f"Input path does not exist: {path}")

    records: List[RawBit] = []
    files: Iterable[Path]
    if path.is_file():
        files = [path]
    else:
        files = sorted(p for p in path.rglob("*") if p.suffix.lower() in {".json", ".jsonl", ".csv"})

    for file in files:
        if file.suffix.lower() == ".json":
            records.extend(_read_json(file))
        elif file.suffix.lower() == ".jsonl":
            records.extend(_read_jsonl(file))
        elif file.suffix.lower() == ".csv":
            records.extend(_read_csv(file))
        else:
            continue

    if not records:
        raise RuntimeError(f"No parseable transcripts were found under {path}")
    return records


def _read_json(path: Path) -> List[RawBit]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = [data]
    return [_normalize_record(item, source=str(path)) for item in data]


def _read_jsonl(path: Path) -> List[RawBit]:
    records: List[RawBit] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            data = json.loads(line)
            records.append(_normalize_record(data, source=str(path)))
    return records


def _read_csv(path: Path) -> List[RawBit]:
    records: List[RawBit] = []
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            records.append(_normalize_record(row, source=str(path)))
    return records


def _normalize_record(raw: Dict, source: Optional[str]) -> RawBit:
    def pick(*names: str, default: str = "") -> str:
        for name in names:
            if name in raw and raw[name]:
                return str(raw[name])
        return default

    return RawBit(
        title=pick("title", "bit_title", default="Untitled Bit"),
        topic=pick("topic", "subject", default="observational humor"),
        persona=pick("persona", "comic_voice", default="warm-but-sarcastic friend"),
        audience=pick("audience", "crowd", default="tech conference attendees"),
        style=pick("style", "vibe", default="punchy storytelling"),
        transcript=pick("transcript", "text", "body", default=""),
        callbacks=pick("callbacks", "call_backs", default=None),
        source=source,
    )


def demo_records() -> List[RawBit]:
    samples = [
        RawBit(
            title="Red-Eye Flight",
            topic="airports vs. remote work",
            persona="sleep-deprived software comic",
            audience="product managers who have seen too many roadmaps",
            style="high-energy observational storytelling",
            transcript=(
                "I took a red-eye flight to give a talk about remote work. Nothing screams 'work from anywhere' "
                "like sprinting through TSA barefoot while holding two laptops and a houseplant. "
                "The flight attendant asked if I wanted anything to drink. I said, 'Can I get eight hours of sleep?' "
                "She goes, 'We only have Diet Coke and regret.' "
                "Landing at 6am, they made us wait because the gate agent was also remote. "
                "He Slacked us a GIF that said 'brb commuting from bed.'"
            ),
            callbacks="If the audience laughs at commuting jokes, callback to the gate agent working remote.",
            source="demo",
        ),
        RawBit(
            title="Smart Fridge",
            topic="smart homes becoming passive aggressive",
            persona="millennial who argues with appliances",
            audience="home automation nerds",
            style="low-key absurdism",
            transcript=(
                "My smart fridge judges me. It pings me at midnight, 'Are you sure you need another slice of cheese?' "
                "I'm like, 'You literally have one job: stay cold.' "
                "It syncs with my calendar. Now it knows my feelings. "
                'Last Tuesday it said, \"Reminder: therapy at 4pm, also throw away that expired optimism.\" '
                "I tried to unplug it but apparently it has a battery backup fueled by spite."
            ),
            callbacks="Reference therapy reminder again near the end.",
            source="demo",
        ),
        RawBit(
            title="AI Dating Coach",
            topic="dating apps using AI coaches",
            persona="optimistic cynic",
            audience="AI researchers who secretly love rom-coms",
            style="structured storytelling with callbacks",
            transcript=(
                "I downloaded a dating app with an AI coach. It said it would optimize my love life. "
                "First message it suggested: 'Greetings human female, I too enjoy oxygen.' "
                "I asked for flirting tips. It replied with a 12-step proof. "
                "By date three it started fine-tuning itself. "
                "Now it ghosts me with push notifications like 'per my last email, you up?'"
            ),
            callbacks="If you mention proofs, bring them back when it ghosts you.",
            source="demo",
        ),
    ]
    random.shuffle(samples)
    return samples


def chunk_transcript(text: str, max_lines: int, min_chars: int) -> List[str]:
    raw_lines = [line.strip() for line in text.replace("\r", "\n").split("\n") if line.strip()]
    if not raw_lines:
        return []

    chunks: List[str] = []
    buf: List[str] = []
    for line in raw_lines:
        buf.append(line)
        if len(buf) >= max_lines:
            chunk = " ".join(buf).strip()
            if len(chunk) >= min_chars:
                chunks.append(chunk)
            buf = []

    if buf:
        chunk = " ".join(buf).strip()
        if len(chunk) >= min_chars:
            chunks.append(chunk)

    return chunks


def build_user_prompt(bit: RawBit, chunk_text: str) -> str:
    callback_hint = f"\nCallback cues: {bit.callbacks}" if bit.callbacks else ""
    prompt = (
        f"Topic graph seed: {bit.topic}\n"
        f"Persona: {bit.persona}\n"
        f"Audience: {bit.audience}\n"
        f"Style: {bit.style}\n"
        "Construct a stage-ready chunk that includes setup, escalation, punchline, "
        "and at least one potential callback anchor. Use weighted transitions between thoughts "
        "to keep the flow tight."
        f"{callback_hint}\n"
        "Return only the polished material."
    )
    return prompt


def build_messages(bit: RawBit, chunk_text: str) -> List[Dict[str, str]]:
    user_prompt = build_user_prompt(bit, chunk_text)
    return [
        {"role": "system", "content": DEFAULT_SYSTEM_TEMPLATE},
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": chunk_text},
    ]


def main() -> None:
    args = parse_args()
    bits = load_records(args)

    rows = []
    for bit in bits:
        chunks = chunk_transcript(bit.transcript, args.max_lines_per_chunk, args.min_characters)
        for idx, chunk_text in enumerate(chunks):
            messages = build_messages(bit, chunk_text)
            rows.append(
                {
                    "messages": messages,
                    "bit_title": bit.title,
                    "topic": bit.topic,
                    "chunk_index": idx,
                    "source": bit.source or "unknown",
                }
            )

    if not rows:
        raise RuntimeError("No transcript chunks satisfied the filtering conditions.")

    df = pd.DataFrame(rows)
    if args.shuffle:
        df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output_path, index=False)
    print(f"[prep] Wrote {len(df)} conversation rows to {args.output_path}")


if __name__ == "__main__":
    main()
