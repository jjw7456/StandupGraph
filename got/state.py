from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple


@dataclass
class ThoughtVertex:
    vertex_id: str
    text: str
    operation: str
    role: str
    score: float
    parents: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)


@dataclass
class Edge:
    source: str
    target: str
    weight: float
    label: str


class GraphReasoningState:
    """Maintains the evolving Graph-of-Thoughts (GoT) state."""

    def __init__(self, experiment_tag: str):
        self.experiment_tag = experiment_tag
        self._id_counter = itertools.count(1)
        self.vertices: Dict[str, ThoughtVertex] = {}
        self.edges: List[Edge] = []
        self._adjacency: Dict[str, List[Edge]] = {}
        self._candidate_logs: List[str] = []

    def new_vertex_id(self) -> str:
        return f"v{next(self._id_counter)}"

    def add_vertex(self, text: str, operation: str, role: str, score: float, parents: Optional[List[str]] = None,
                   metadata: Optional[Dict] = None) -> ThoughtVertex:
        vid = self.new_vertex_id()
        vertex = ThoughtVertex(
            vertex_id=vid,
            text=text.strip(),
            operation=operation,
            role=role,
            score=score,
            parents=parents or [],
            metadata=metadata or {},
        )
        self.vertices[vid] = vertex
        self._log_vertex(vertex)
        return vertex

    def add_edge(self, source: str, target: str, weight: float, label: str) -> Edge:
        weight = max(0.0, min(1.0, weight))
        edge = Edge(source=source, target=target, weight=weight, label=label)
        self.edges.append(edge)
        self._adjacency.setdefault(source, []).append(edge)
        return edge

    def vertices_from_ops(self, op_names: Iterable[str]) -> List[ThoughtVertex]:
        op_set = set(op_names)
        return [v for v in self.vertices.values() if v.operation in op_set]

    def best_vertices(self, limit: int = 5) -> List[ThoughtVertex]:
        return sorted(self.vertices.values(), key=lambda v: v.score, reverse=True)[:limit]

    def descend_path(self, num_steps: int) -> List[ThoughtVertex]:
        """Follow weighted edges to obtain a candidate performance order."""
        if not self.vertices:
            return []

        current = max(self.vertices.values(), key=lambda v: v.score)
        path = [current]
        for _ in range(num_steps - 1):
            outgoing = self._adjacency.get(current.vertex_id, [])
            if not outgoing:
                break
            outgoing = sorted(outgoing, key=lambda e: (e.weight, self.vertices[e.target].score), reverse=True)
            next_edge = outgoing[0]
            current = self.vertices[next_edge.target]
            path.append(current)
        return path

    def render_path(self, path: List[ThoughtVertex]) -> str:
        lines = []
        for idx, vertex in enumerate(path, start=1):
            lines.append(f"{idx:02d}. [{vertex.role} | score={vertex.score:.2f}] {vertex.text}")
        return "\n".join(lines)

    def _log_vertex(self, vertex: ThoughtVertex) -> None:
        parents = ", ".join(vertex.parents) if vertex.parents else "root"
        log_line = (
            f"[{self.experiment_tag}] vertex={vertex.vertex_id} op={vertex.operation} role={vertex.role} "
            f"score={vertex.score:.3f} parents={parents}\n    text={vertex.text}"
        )
        self._candidate_logs.append(log_line)
        print(log_line)

    @property
    def candidate_logs(self) -> List[str]:
        return self._candidate_logs
