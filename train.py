#!/usr/bin/env python
# -*- coding: utf-8 -*-
# train.py — QLoRA fine-tune for Meta-Llama-3.1-8B-Instruct
"""
QLoRA training script for meta-llama/Meta-Llama-3.1-8B-Instruct using TRL's SFTTrainer.

Highlights
----------
* Loads the base model in 4-bit NF4 via bitsandbytes (avoids accidental fp16/fp32 dequantization).
* Injects LoRA adapters on standard Llama projection modules (q/k/v/o/up/gate/down).
* Consumes Harmony-style `messages` stored inside `prep_out/comedy_messages.parquet`.
"""
from __future__ import annotations

import argparse
import json
import os

import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="QLoRA fine-tune for Meta-Llama-3.1-8B-Instruct (NF4 quantization)."
    )
    parser.add_argument("--model_id", type=str, default=MODEL_ID)
    parser.add_argument(
        "--data_path",
        type=str,
        default="prep_out/comedy_messages.parquet",
        help="Parquet file with a `messages` column formatted like OpenAI Harmony chats.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints/llama31-8b-qlora-comedy",
        help="Directory where adapters/tokenizer/manifests are written.",
    )

    # Sequence / batch
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=1, help="Per-device micro batch size.")
    parser.add_argument("--grad_accum", type=int, default=8, help="Gradient accumulation steps.")
    parser.add_argument("--epochs", type=float, default=1.0)

    # Optim / Scheduler
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine_with_min_lr")
    parser.add_argument("--min_lr_rate", type=float, default=0.1)

    # LoRA
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    # Logging / saving
    parser.add_argument("--logging_steps", type=int, default=5)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--save_total_limit", type=int, default=2)

    # Misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--bf16",
        action="store_true",
        default=True,
        help="Use BF16 for compute inside the 4-bit layers (recommended on recent GPUs).",
    )
    parser.add_argument("--no_bf16", dest="bf16", action="store_false")
    return parser.parse_args()


def load_dataset_from_parquet(path: str) -> Dataset:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")
    df = pd.read_parquet(path)

    if "messages" not in df.columns:
        cols = ", ".join(df.columns)
        raise KeyError(
            f"Parquet must contain a `messages` column in harmony format; found columns: {cols}"
        )

    df = df[["messages"]]  # drop auxiliary metadata columns

    sample = df.iloc[0]["messages"]
    if not (isinstance(sample, list) and isinstance(sample[0], dict) and "role" in sample[0]):
        raise ValueError("'messages' must be a list of {role, content, ...} dicts.")

    ds = Dataset.from_pandas(df, preserve_index=False)
    return ds


def build_model_and_tokenizer(model_id: str, use_bf16: bool):
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float32
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quant_config,
        device_map="auto",
    )
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def attach_lora_adapters(model, r: int, alpha: int, dropout: float):
    """Attach LoRA adapters to the usual Llama projection modules."""
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj"]
    lora = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        target_modules=target_modules,
        task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(model, lora)
    peft_model.print_trainable_parameters()
    return peft_model


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    # Data
    train_ds = load_dataset_from_parquet(args.data_path)

    # Model + tokenizer
    model, tokenizer = build_model_and_tokenizer(args.model_id, use_bf16=args.bf16)

    # LoRA on 4-bit weights = QLoRA
    model = attach_lora_adapters(model, args.lora_r, args.lora_alpha, args.lora_dropout)

    train_cfg = SFTConfig(
        output_dir=args.output_dir,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        lr_scheduler_kwargs={"min_lr_rate": args.min_lr_rate},
        max_seq_length=args.max_length,
        gradient_checkpointing=True,
        bf16=args.bf16,
        fp16=False,
        optim=(
            "adamw_torch_fused"
            if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
            else "adamw_torch"
        ),
        report_to=None,
    )

    trainer = SFTTrainer(
        model=model,
        args=train_cfg,
        train_dataset=train_ds,
        tokenizer=tokenizer,
        dataset_text_field="messages",
        packing=False,
    )

    trainer.train()

    # Save adapters + tokenizer
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    manifest = {
        "model_id": args.model_id,
        "data_path": args.data_path,
        "output_dir": args.output_dir,
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "epochs": args.epochs,
        "lr": args.lr,
        "bf16": args.bf16,
        "quantization": "bnb.nf4",
        "lora": {"r": args.lora_r, "alpha": args.lora_alpha, "dropout": args.lora_dropout},
    }
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "train_manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
