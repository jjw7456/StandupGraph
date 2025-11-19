from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence


@dataclass
class Operation:
    name: str
    kind: str
    branching: int
    parents: List[str] = field(default_factory=list)
    parent_sample_size: int = 3
    temperature: float = 0.7
    max_tokens: int = 256
    metadata: Dict = field(default_factory=dict)


class GraphOfOperations:
    """Static execution plan that mirrors the GoO definition in GoT."""

    def __init__(self, operations: Sequence[Operation]):
        self.operations = list(operations)
        self._index = {op.name: op for op in self.operations}

    def __iter__(self):
        return iter(self.operations)

    def __len__(self):
        return len(self.operations)

    def get(self, name: str) -> Operation:
        return self._index[name]

    def parents(self, operation: Operation) -> List[Operation]:
        return [self._index[p] for p in operation.parents if p in self._index]


def build_default_comedy_plan(num_lines: int = 8) -> GraphOfOperations:
    """
    Build a default GoO tailored to stand-up generation.

    The plan features:
        * A seed idea vertex.
        * Setup exploration with high branching.
        * Punchline and reversal discovery.
        * Callback weaving that links back to prior setups.
        * Refinement stage that polishes closers.
    """
    operations: List[Operation] = []
    operations.append(
        Operation(
            name="seed_topics",
            kind="seed",
            branching=1,
            parent_sample_size=0,
            temperature=0.3,
            metadata={"role": "concept"},
        )
    )
    operations.append(
        Operation(
            name="setup_search",
            kind="generate",
            branching=min(4, max(2, num_lines // 2)),
            parents=["seed_topics"],
            parent_sample_size=1,
            temperature=0.9,
            metadata={"role": "setup"},
        )
    )
    operations.append(
        Operation(
            name="punchline_forge",
            kind="punchline",
            branching=min(4, num_lines),
            parents=["setup_search"],
            parent_sample_size=min(3, max(1, num_lines // 3)),
            temperature=0.85,
            metadata={"role": "punchline"},
        )
    )
    operations.append(
        Operation(
            name="callback_weaver",
            kind="callback",
            branching=min(3, num_lines // 2 or 1),
            parents=["setup_search", "punchline_forge"],
            parent_sample_size=min(4, max(2, num_lines // 2)),
            temperature=0.75,
            metadata={"role": "callback"},
        )
    )
    operations.append(
        Operation(
            name="tag_runner",
            kind="tag",
            branching=min(3, num_lines // 2 or 1),
            parents=["punchline_forge", "callback_weaver"],
            parent_sample_size=min(3, max(2, num_lines // 3 or 1)),
            temperature=0.7,
            metadata={"role": "tag"},
        )
    )
    operations.append(
        Operation(
            name="closer_refine",
            kind="refine",
            branching=1,
            parents=["tag_runner"],
            parent_sample_size=min(4, max(2, num_lines // 2 or 1)),
            temperature=0.55,
            metadata={"role": "closer"},
        )
    )
    return GraphOfOperations(operations)
