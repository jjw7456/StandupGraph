from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

from .llm import GenerationParams, LocalLLM
from .operations import Operation
from .state import ThoughtVertex


@dataclass
class ComedyContext:
    topics: Sequence[str]
    persona: str
    audience: str
    style: str
    desired_minutes: float


class ComedyPrompter:
    """Transforms controller intents into chat prompts for the LLM."""

    def __init__(self, llm: LocalLLM, context: ComedyContext):
        self.llm = llm
        self.context = context
        self.system_prompt = (
            "You are an award-winning stand-up comedian and comedy writer. "
            "You reason with a Graph-of-Thoughts mindset: each joke line is a vertex connected "
            "by weighted edges representing transitions (setup→misdirection→punchline→callback). "
            "Always respond with JSON: {\"candidates\": [{\"line\": ..., \"role\": ..., "
            "\"confidence\": float, \"callbacks\": [], \"topics\": []}]}."
        )

    def run_operation(self, operation: Operation, parents: List[ThoughtVertex]) -> str:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self._build_user_prompt(operation, parents)},
        ]
        params = GenerationParams(
            max_new_tokens=operation.max_tokens,
            temperature=operation.temperature,
        )
        return self.llm.generate_chat(messages, params)

    def _build_user_prompt(self, operation: Operation, parents: List[ThoughtVertex]) -> str:
        parent_summary = "\n".join(
            f"- ({p.role}, score={p.score:.2f}) {p.text}" for p in parents[-4:]
        ) or "- (none yet) starting the set."
        op_goal = self._operation_goal(operation)
        prompt = (
            f"Persona: {self.context.persona}\n"
            f"Audience: {self.context.audience}\n"
            f"Topics in play: {', '.join(self.context.topics)}\n"
            f"Stage style: {self.context.style}\n"
            f"Desired runtime: {self.context.desired_minutes:.1f} minutes\n"
            f"Current graph neighborhood:\n{parent_summary}\n\n"
            f"Task: {op_goal}\n"
            "Return JSON with 1..{branching} candidates under key 'candidates'. "
            "Each candidate must include: line, role, confidence (0-1), callbacks (strings), topics."
        ).format(branching=operation.branching)
        return prompt

    def _operation_goal(self, operation: Operation) -> str:
        match operation.kind:
            case "seed":
                return "Generate a single comedic micro-plot that stitches together the topics and hints at future callbacks."
            case "generate":
                return (
                    "Brainstorm several high-energy setups (vivid observations, contradictions) that end with "
                    "an implicit question to be resolved later."
                )
            case "punchline":
                return (
                    "Attach punchlines or reversals to the provided setups. Highlight incongruity, escalate stakes, "
                    "and suggest where callbacks could tether."
                )
            case "callback":
                return (
                    "Weave callbacks that explicitly reference earlier setups/punchlines. Edge weights should capture how "
                    "smoothly the audience can follow the reference."
                )
            case "tag":
                return (
                    "Produce short tag lines or topper jokes that keep the energy up while staying in the topic graph."
                )
            case "refine":
                return (
                    "Polish the closing segment by combining the highest scoring callbacks and delivering a clean closer."
                )
            default:
                return "Produce additional humorous beats that respect the existing graph structure."
