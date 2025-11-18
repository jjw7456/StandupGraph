from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class GenerationParams:
    max_new_tokens: int = 220
    temperature: float = 0.7
    top_p: float = 0.9
    repetition_penalty: float = 1.05


class LocalLLM:
    """Lightweight helper around a Hugging Face causal LM."""

    def __init__(
        self,
        model_id: str,
        adapter_path: str | None = None,
        device: str | None = None,
        torch_dtype: str = "bfloat16",
    ):
        requested_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = requested_device
        self._auto_device_map = requested_device.startswith("cuda")
        dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[torch_dtype]
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto" if self._auto_device_map else None,
            torch_dtype=dtype,
        )
        if not self._auto_device_map:
            model.to(self.device)
        if adapter_path:
            model = PeftModel.from_pretrained(model, adapter_path)
        self.model = model.eval()
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate_chat(self, messages: Sequence[dict], params: GenerationParams) -> str:
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        return self.generate_text(prompt, params)

    def generate_text(self, prompt: str, params: GenerationParams) -> str:
        target_device = self.device
        inputs = self.tokenizer(prompt, return_tensors="pt").to(target_device)
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=params.max_new_tokens,
                temperature=params.temperature,
                top_p=params.top_p,
                do_sample=params.temperature > 0,
                repetition_penalty=params.repetition_penalty,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        # Remove the prompt to get only the completion
        completion = text[len(prompt):].strip()
        return completion
