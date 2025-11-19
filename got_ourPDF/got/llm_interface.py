from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import random

class LLMInterface(ABC):
    """Abstract interface for the LLM."""

    @abstractmethod
    def generate(self, prompt: str, max_new_tokens: int = 512, temperature: float = 0.7) -> str:
        """Generates text based on the prompt."""
        pass

class MockLLM(LLMInterface):
    """Mock LLM for testing purposes."""

    def generate(self, prompt: str, max_new_tokens: int = 512, temperature: float = 0.7) -> str:
        """Returns a mock response."""
        if "Score:" in prompt or "Rate the following" in prompt:
            return f"Score: {random.randint(6, 9)}\nReasoning: This is a funny joke with good structure."
        elif "Generate a setup" in prompt:
            return f"Setup: Why did the programmer quit his job? {random.randint(1, 100)}"
        elif "Generate a punchline" in prompt:
            return f"Punchline: Because he didn't get arrays! {random.randint(1, 100)}"
        else:
            return "Mock response"

class LlamaLLM(LLMInterface):
    """LLM implementation using transformers for Llama 3.1."""

    def __init__(self, model_path: str = "meta-llama/Meta-Llama-3.1-8B-Instruct", device: str = "cuda"):
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
        except ImportError:
            raise ImportError("Please install transformers and torch to use LlamaLLM.")

        self.device = device
        print(f"Loading model from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        print("Model loaded.")

    def generate(self, prompt: str, max_new_tokens: int = 512, temperature: float = 0.7) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return response
