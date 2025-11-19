from typing import List, Dict, Any, Callable
from .state import GraphReasoningState
from .llm_interface import LLMInterface
from .prompter import Prompter
from .parser import Parser
from .validation import Validation

class GraphOfOperations:
    """
    Graph of Operations (GoO).
    Defines the operations that can be performed on the graph state.
    """

    def __init__(self, llm: LLMInterface, prompter: Prompter, parser: Parser, validation: Validation):
        self.llm = llm
        self.prompter = prompter
        self.parser = parser
        self.validation = validation

    def generate_setup(self, state: GraphReasoningState, parent_id: str, topic: str, num_samples: int = 3) -> List[str]:
        """Generates candidate setups."""
        candidates = []
        prompt = self.prompter.generate_setup_prompt(topic)
        
        for _ in range(num_samples):
            response = self.llm.generate(prompt)
            setup = self.parser.parse_setup(response)
            if self.validation.validate_setup(setup):
                # Add to graph
                vertex_id = state.add_vertex(content=setup, type="setup", parent_id=parent_id)
                candidates.append(vertex_id)
        
        return candidates

    def generate_punchline(self, state: GraphReasoningState, setup_id: str, num_samples: int = 3) -> List[str]:
        """Generates candidate punchlines for a setup."""
        setup_node = state.get_vertex(setup_id)
        setup_text = setup_node["content"]
        
        candidates = []
        prompt = self.prompter.generate_punchline_prompt(setup_text)
        
        for _ in range(num_samples):
            response = self.llm.generate(prompt)
            punchline = self.parser.parse_punchline(response)
            if self.validation.validate_punchline(punchline):
                vertex_id = state.add_vertex(content=punchline, type="punchline", parent_id=setup_id)
                candidates.append(vertex_id)
                
        return candidates

    def score_candidates(self, state: GraphReasoningState, candidate_ids: List[str]):
        """Scores a list of candidate nodes."""
        for cid in candidate_ids:
            node = state.get_vertex(cid)
            content = node["content"]
            
            # If it's a punchline, we need the setup for context
            context = ""
            if node["type"] == "punchline":
                preds = list(state.graph.predecessors(cid))
                if preds:
                    setup_node = state.get_vertex(preds[0])
                    context = setup_node["content"]
            
            # Construct prompt based on type
            if node["type"] == "punchline":
                prompt = self.prompter.score_joke_prompt(context, content)
            else:
                # Simple scoring for setup? Or just skip?
                # Let's score setups by their potential
                prompt = f"Rate this comedy setup: '{content}' on 1-10."

            response = self.llm.generate(prompt)
            score_data = self.parser.parse_score(response)
            
            if self.validation.validate_score(score_data):
                state.graph.nodes[cid]["score"] = score_data["score"]
                state.graph.nodes[cid]["reasoning"] = score_data["reasoning"]

    def keep_best(self, state: GraphReasoningState, candidate_ids: List[str], k: int = 1) -> List[str]:
        """Selects the top k candidates based on score."""
        scored_candidates = []
        for cid in candidate_ids:
            node = state.get_vertex(cid)
            scored_candidates.append((cid, node.get("score", 0)))
        
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        return [x[0] for x in scored_candidates[:k]]
