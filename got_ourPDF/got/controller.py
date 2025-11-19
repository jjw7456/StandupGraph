from typing import List, Dict
from .state import GraphReasoningState
from .operations import GraphOfOperations
from .llm_interface import LLMInterface, MockLLM, LlamaLLM
from .prompter import Prompter
from .parser import Parser
from .validation import Validation
import json

class Controller:
    """
    Controller for the Graph of Thoughts system.
    Orchestrates the generation process.
    """

    def __init__(self, llm: LLMInterface = None):
        self.llm = llm if llm else MockLLM()
        self.prompter = Prompter()
        self.parser = Parser()
        self.validation = Validation()
        self.goo = GraphOfOperations(self.llm, self.prompter, self.parser, self.validation)
        self.state = GraphReasoningState()

    def run(self, topic: str, num_branches: int = 3, k: int = 1):
        """
        Runs the GoT process.
        1. Generate Setups
        2. Score Setups
        3. Keep Best Setup(s)
        4. Generate Punchlines for Best Setup(s)
        5. Score Punchlines
        6. Keep Best Punchline(s)
        """
        print(f"Starting GoT for topic: {topic}")

        # Step 1: Generate Setups
        print("Generating setups...")
        setup_ids = self.goo.generate_setup(self.state, self.state.root_id, topic, num_samples=num_branches)
        self.log_candidates(setup_ids, "Setup Candidates")

        # Step 2: Score Setups
        print("Scoring setups...")
        self.goo.score_candidates(self.state, setup_ids)

        # Step 3: Keep Best Setup
        print("Selecting best setup...")
        best_setup_ids = self.goo.keep_best(self.state, setup_ids, k=k)
        print(f"Selected {len(best_setup_ids)} best setup(s).")

        final_jokes = []

        # Step 4: Generate Punchlines for Best Setup
        for setup_id in best_setup_ids:
            print(f"Generating punchlines for setup {setup_id}...")
            punchline_ids = self.goo.generate_punchline(self.state, setup_id, num_samples=num_branches)
            self.log_candidates(punchline_ids, f"Punchline Candidates for Setup {setup_id}")

            # Step 5: Score Punchlines
            print("Scoring punchlines...")
            self.goo.score_candidates(self.state, punchline_ids)

            # Step 6: Keep Best Punchline
            print("Selecting best punchline...")
            best_punchline_ids = self.goo.keep_best(self.state, punchline_ids, k=1)
            
            if best_punchline_ids:
                best_pid = best_punchline_ids[0]
                punchline_node = self.state.get_vertex(best_pid)
                setup_node = self.state.get_vertex(setup_id)
                
                joke = {
                    "setup": setup_node["content"],
                    "punchline": punchline_node["content"],
                    "score": punchline_node.get("score", 0)
                }
                final_jokes.append(joke)

        return final_jokes

    def log_candidates(self, candidate_ids: List[str], title: str):
        """Logs candidate vertices for research and debugging."""
        print(f"\n--- {title} ---")
        for cid in candidate_ids:
            node = self.state.get_vertex(cid)
            print(f"ID: {cid}")
            print(f"Content: {node['content']}")
            print(f"Score: {node.get('score', 'N/A')}")
            print(f"Reasoning: {node.get('reasoning', 'N/A')}")
            print("-" * 20)
        print("-------------------\n")
