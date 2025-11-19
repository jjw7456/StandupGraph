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

    def __init__(self, model_path: str = "meta-llama/Meta-Llama-3.1-8B-Instruct", adapter_path: str = None, quantization: str = None, device: str = "cuda"):
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
            import torch
        except ImportError:
            raise ImportError("Please install transformers, torch, and bitsandbytes to use LlamaLLM.")

        self.device = device
        print(f"Loading model from {model_path}...")
        
        quantization_config = None
        if quantization == "4bit":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        elif quantization == "8bit":
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Load base model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=quantization_config,
            torch_dtype=torch.bfloat16 if quantization is None else None,
            device_map="auto"
        )

        # Load adapter if provided
        if adapter_path:
            try:
                from peft import PeftModel
                print(f"Loading adapter from {adapter_path}...")
                self.model = PeftModel.from_pretrained(self.model, adapter_path)
            except ImportError:
                raise ImportError("Please install peft to use LoRA adapters.")
        
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
