#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Graph-of-Thought inference pipeline for structured stand-up comedy generation.

Usage example:
    python inference.py --model_id meta-llama/Meta-Llama-3.1-8B-Instruct \\
        --adapter_path checkpoints/llama31-8b-qlora-comedy \\
        --topics "red-eye flights, ai assistants" --persona "overcaffeinated engineer comic"
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from got import (
    ComedyController,
    ComedyParser,
    ComedyPrompter,
    ComedyValidator,
    CriticConfig,
    GraphReasoningState,
    LocalLLM,
    build_default_comedy_plan,
)
from got.prompter import ComedyContext


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Graph-of-Thought stand-up generator.")
    parser.add_argument("--model_id", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--adapter_path", type=str, default=None, help="Path to PEFT adapters (optional).")
    parser.add_argument("--topics", type=str, required=True, help="Comma separated topics to stitch together.")
    parser.add_argument("--persona", type=str, default="friendly observational comic")
    parser.add_argument("--audience", type=str, default="tech conference audience")
    parser.add_argument("--style", type=str, default="structured storytelling with callbacks")
    parser.add_argument("--minutes", type=float, default=5.0, help="Approximate runtime for context building.")
    parser.add_argument("--num_lines", type=int, default=8, help="Target number of lines/vertices to surface.")
    parser.add_argument("--device", type=str, default=None, help="torch device override (cpu / cuda).")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--enable_critic", action="store_true", help="Use the model as a self-critic for scoring.")
    parser.add_argument("--run_name", type=str, default="Standup4AI-GOT")
    parser.add_argument("--log_path", type=Path, default=None, help="Optional file to dump candidate logs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    topics = [topic.strip() for topic in args.topics.split(",") if topic.strip()]
    if not topics:
        raise ValueError("Please provide at least one topic via --topics.")

    llm = LocalLLM(
        model_id=args.model_id,
        adapter_path=args.adapter_path,
        device=args.device,
        torch_dtype=args.dtype,
    )
    critic_llm = llm if args.enable_critic else None

    context = ComedyContext(
        topics=topics,
        persona=args.persona,
        audience=args.audience,
        style=args.style,
        desired_minutes=args.minutes,
    )

    prompter = ComedyPrompter(llm, context)
    parser = ComedyParser()
    validator = ComedyValidator(
        topics=topics,
        critic_llm=critic_llm,
        critic_cfg=CriticConfig(enabled=args.enable_critic),
    )
    goo = build_default_comedy_plan(num_lines=args.num_lines)
    grs = GraphReasoningState(experiment_tag=args.run_name)
    controller = ComedyController(goo, grs, prompter, parser, validator)

    final_path = controller.run(target_lines=args.num_lines)
    print("\n=== SELECTED PERFORMANCE ORDER ===")
    print(grs.render_path(final_path))

    if args.log_path:
        args.log_path.parent.mkdir(parents=True, exist_ok=True)
        args.log_path.write_text("\n".join(grs.candidate_logs), encoding="utf-8")
        print(f"[got] Candidate logs saved to {args.log_path}")


if __name__ == "__main__":
    main()
