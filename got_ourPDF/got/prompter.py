from typing import List, Dict, Any

class Prompter:
    """Generates prompts for the LLM."""

    def __init__(self):
        pass

    def generate_setup_prompt(self, topic: str, style: str = "observational") -> str:
        """Generates a prompt for creating a comedy setup."""
        return f"""You are a professional stand-up comedian.
Style: {style}
Topic: {topic}

Generate a setup for a joke. Do not include the punchline yet.
Format your output as:
Setup: [Your setup here]
"""

    def generate_punchline_prompt(self, setup: str) -> str:
        """Generates a prompt for creating a punchline given a setup."""
        return f"""You are a professional stand-up comedian.
Setup: {setup}

Generate a funny punchline for this setup.
Format your output as:
Punchline: [Your punchline here]
"""

    def score_joke_prompt(self, setup: str, punchline: str) -> str:
        """Generates a prompt for scoring a joke."""
        return f"""Rate the following joke on a scale of 1 to 10 based on humor, structure, and originality.
Setup: {setup}
Punchline: {punchline}

Provide a score and a brief reasoning.
Format your output as:
Score: [1-10]
Reasoning: [Your reasoning]
"""
