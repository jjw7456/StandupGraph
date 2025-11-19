from __future__ import annotations

from typing import List

from .operations import GraphOfOperations, Operation
from .parser import ComedyParser
from .prompter import ComedyPrompter
from .state import GraphReasoningState, ThoughtVertex
from .validator import ComedyValidator


class ComedyController:
    """Coordinates Prompter, Parser, Validator, and maintains the GRS."""

    def __init__(
        self,
        goo: GraphOfOperations,
        grs: GraphReasoningState,
        prompter: ComedyPrompter,
        parser: ComedyParser,
        validator: ComedyValidator,
    ):
        self.goo = goo
        self.grs = grs
        self.prompter = prompter
        self.parser = parser
        self.validator = validator

    def run(self, target_lines: int) -> List[ThoughtVertex]:
        for operation in self.goo:
            parents = self._select_parents(operation)
            raw = self.prompter.run_operation(operation, parents)
            parsed = self.parser.parse(operation, raw)
            for candidate in parsed[: operation.branching]:
                score, diagnostics = self.validator.score_candidate(
                    text=candidate["text"],
                    role=candidate["role"],
                    parents=parents,
                    expected_role=operation.metadata.get("role", candidate["role"]),
                    callbacks=candidate.get("callbacks", []),
                )
                vertex = self.grs.add_vertex(
                    text=candidate["text"],
                    operation=operation.name,
                    role=candidate["role"],
                    score=score,
                    parents=[p.vertex_id for p in parents] or [],
                    metadata={"diagnostics": diagnostics, "notes": candidate.get("notes", "")},
                )
                if parents:
                    for parent in parents:
                        weight = 0.5 * score + 0.5 * parent.score
                        self.grs.add_edge(parent.vertex_id, vertex.vertex_id, weight=weight, label=operation.name)

        final_path = self.grs.beam_search_path(num_steps=target_lines)
        if not final_path:
            final_path = self.grs.descend_path(num_steps=target_lines)
        return final_path

    def _select_parents(self, operation: Operation) -> List[ThoughtVertex]:
        if not operation.parents:
            return []
        candidates: List[ThoughtVertex] = self.grs.vertices_from_ops(operation.parents)
        if not candidates:
            candidates = self.grs.best_vertices(limit=3)
        limit = operation.parent_sample_size or len(candidates)
        return sorted(candidates, key=lambda v: v.score, reverse=True)[:limit]
