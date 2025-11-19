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
        self._reverse_adjacency: Dict[str, List[Edge]] = {}
        self._candidate_logs: List[str] = []
        self._volume_cache: Dict[str, int] = {}

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
        self._reverse_adjacency.setdefault(vid, [])
        self._volume_cache.clear()
        self._log_vertex(vertex)
        return vertex

    def add_edge(self, source: str, target: str, weight: float, label: str) -> Edge:
        weight = max(0.0, min(1.0, weight))
        edge = Edge(source=source, target=target, weight=weight, label=label)
        self.edges.append(edge)
        self._adjacency.setdefault(source, []).append(edge)
        self._reverse_adjacency.setdefault(target, []).append(edge)
        self._volume_cache.clear()
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

    def beam_search_path(self, num_steps: int, beam_width: int = 3) -> List[ThoughtVertex]:
        """Select a high-quality performance order via beam search scored by GoT-inspired heuristics."""
        if not self.vertices:
            return []

        beam_width = max(1, beam_width)
        seeds = self.best_vertices(limit=beam_width)
        beams: List[Tuple[List[ThoughtVertex], float]] = [([seed], seed.score) for seed in seeds]
        if not beams:
            return []

        best_path, best_score = beams[0]
        for _ in range(max(0, num_steps - 1)):
            next_beams: List[Tuple[List[ThoughtVertex], float]] = []
            for path, path_score in beams:
                last_vertex = path[-1]
                outgoing = self._adjacency.get(last_vertex.vertex_id, [])
                if not outgoing:
                    next_beams.append((path, path_score))
                    continue

                scored_edges = [
                    (
                        self._transition_reward(edge, self.vertices[edge.target]),
                        edge,
                    )
                    for edge in outgoing
                ]
                scored_edges.sort(key=lambda item: item[0], reverse=True)
                for reward, edge in scored_edges[:beam_width]:
                    target_vertex = self.vertices[edge.target]
                    new_path = path + [target_vertex]
                    next_beams.append((new_path, path_score + reward))

            if not next_beams:
                break
            next_beams.sort(key=lambda item: item[1], reverse=True)
            beams = next_beams[:beam_width]
            if beams and beams[0][1] > best_score:
                best_path, best_score = beams[0]

        return best_path

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

    def thought_volume(self, vertex_id: str) -> int:
        """Return the GoT 'volume' metric: number of ancestors that can reach this vertex."""
        if vertex_id in self._volume_cache:
            return self._volume_cache[vertex_id]
        visited = set()
        stack = [vertex_id]
        while stack:
            current = stack.pop()
            for edge in self._reverse_adjacency.get(current, []):
                parent_id = edge.source
                if parent_id not in visited:
                    visited.add(parent_id)
                    stack.append(parent_id)
        self._volume_cache[vertex_id] = len(visited)
        return self._volume_cache[vertex_id]

    def _transition_reward(self, edge: Edge, target: ThoughtVertex) -> float:
        """Score how promising it is to travel across `edge` into `target`."""
        volume = self.thought_volume(target.vertex_id)
        neighborhood = len(self.vertices)
        if neighborhood <= 1:
            volume_score = 0.0
        else:
            denom = math.log1p(neighborhood)
            volume_score = min(1.0, math.log1p(volume + 1) / denom) if denom > 0 else 0.0
        return 0.45 * edge.weight + 0.4 * target.score + 0.15 * volume_score
