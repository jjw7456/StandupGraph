class Validation:
    """Validates generated content."""

    def validate_setup(self, setup: str) -> bool:
        """Checks if the setup is valid."""
        if not setup or len(setup) < 10:
            return False
        return True

    def validate_punchline(self, punchline: str) -> bool:
        """Checks if the punchline is valid."""
        if not punchline or len(punchline) < 2:
            return False
        return True

    def validate_score(self, score_data: dict) -> bool:
        """Checks if the score is valid."""
        if not isinstance(score_data.get("score"), int):
            return False
        if score_data["score"] < 1 or score_data["score"] > 10:
            return False
        return True
