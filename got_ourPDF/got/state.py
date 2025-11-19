import networkx as nx
from typing import List, Dict, Any, Optional
import uuid

class GraphReasoningState:
    """
    Graph Reasoning State (GRS).
    Maintains the state of the graph, including vertices (thoughts) and edges (dependencies).
    """

    def __init__(self):
        self.graph = nx.DiGraph()
        self.root_id = "root"
        self.graph.add_node(self.root_id, content="Start of Standup Routine", type="root", score=0)

    def add_vertex(self, content: str, type: str, parent_id: str = None, score: float = 0.0, metadata: Dict = None) -> str:
        """Adds a new vertex to the graph."""
        vertex_id = str(uuid.uuid4())
        self.graph.add_node(vertex_id, content=content, type=type, score=score, metadata=metadata or {})
        
        if parent_id:
            if parent_id not in self.graph:
                raise ValueError(f"Parent node {parent_id} does not exist.")
            self.graph.add_edge(parent_id, vertex_id)
        
        return vertex_id

    def get_vertex(self, vertex_id: str) -> Dict:
        """Retrieves a vertex by ID."""
        return self.graph.nodes[vertex_id]

    def get_successors(self, vertex_id: str) -> List[str]:
        """Returns the IDs of successor nodes."""
        return list(self.graph.successors(vertex_id))

    def get_leaves(self) -> List[str]:
        """Returns the IDs of leaf nodes."""
        return [n for n in self.graph.nodes() if self.graph.out_degree(n) == 0]

    def get_path_to_root(self, vertex_id: str) -> List[str]:
        """Returns the path from the given vertex back to the root."""
        path = []
        current = vertex_id
        while current:
            path.append(current)
            preds = list(self.graph.predecessors(current))
            if preds:
                current = preds[0] # Assuming tree structure for now
            else:
                break
        return list(reversed(path))
