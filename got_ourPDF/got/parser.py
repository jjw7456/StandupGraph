import re
from typing import Dict, Any, Optional

class Parser:
    """Parses LLM outputs."""

    def parse_setup(self, text: str) -> str:
        """Extracts the setup from the text."""
        match = re.search(r"Setup:\s*(.*)", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return text.strip() # Fallback

    def parse_punchline(self, text: str) -> str:
        """Extracts the punchline from the text."""
        match = re.search(r"Punchline:\s*(.*)", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return text.strip() # Fallback

    def parse_score(self, text: str) -> Dict[str, Any]:
        """Extracts score and reasoning."""
        score_match = re.search(r"Score:\s*(\d+)", text)
        reasoning_match = re.search(r"Reasoning:\s*(.*)", text, re.DOTALL)
        
        score = int(score_match.group(1)) if score_match else 0
        reasoning = reasoning_match.group(1).strip() if reasoning_match else ""
        
        return {"score": score, "reasoning": reasoning}
