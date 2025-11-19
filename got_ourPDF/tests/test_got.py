import unittest
from got.prompter import Prompter
from got.parser import Parser
from got.validation import Validation
from got.state import GraphReasoningState
from got.operations import GraphOfOperations
from got.llm_interface import MockLLM

class TestGoT(unittest.TestCase):
    def setUp(self):
        self.prompter = Prompter()
        self.parser = Parser()
        self.validation = Validation()
        self.llm = MockLLM()
        self.goo = GraphOfOperations(self.llm, self.prompter, self.parser, self.validation)
        self.state = GraphReasoningState()

    def test_prompter(self):
        prompt = self.prompter.generate_setup_prompt("Test Topic")
        self.assertIn("Test Topic", prompt)
        self.assertIn("Setup:", prompt)

    def test_parser(self):
        text = "Setup: This is a setup."
        parsed = self.parser.parse_setup(text)
        self.assertEqual(parsed, "This is a setup.")

        score_text = "Score: 8\nReasoning: Good."
        score_data = self.parser.parse_score(score_text)
        self.assertEqual(score_data["score"], 8)

    def test_validation(self):
        self.assertTrue(self.validation.validate_setup("This is a long enough setup."))
        self.assertFalse(self.validation.validate_setup("Short"))

    def test_graph_operations(self):
        # Test generating setups
        setup_ids = self.goo.generate_setup(self.state, self.state.root_id, "Test", num_samples=2)
        self.assertEqual(len(setup_ids), 2)
        
        # Test scoring
        self.goo.score_candidates(self.state, setup_ids)
        for sid in setup_ids:
            node = self.state.get_vertex(sid)
            self.assertIn("score", node)

if __name__ == '__main__':
    unittest.main()
