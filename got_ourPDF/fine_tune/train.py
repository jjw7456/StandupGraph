import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer
import os

def train():
    # Configuration
    model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    dataset_path = "fine_tune/train.jsonl"
    output_dir = "fine_tune/adapters"
    
    # Load Dataset
    if not os.path.exists(dataset_path):
        print(f"Dataset {dataset_path} not found. Please run preprocess.py first.")
        return

    dataset = load_dataset("json", data_files=dataset_path, split="train")

    # Quantization Config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" # Fix for fp16

    # Load Model
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto"
    )
    
    model = prepare_model_for_kbit_training(model)

    # LoRA Config
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Training Arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=2, # Low batch size for consumer GPU
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        fp16=False,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        optim="paged_adamw_32bit",
        report_to="none" # Disable wandb/tensorboard for simplicity
    )

    # Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        tokenizer=tokenizer,
        args=training_args,
        packing=False, # Can set to True if we want to pack sequences
    )

    # Train
    print("Starting training...")
    trainer.train()
    
    # Save
    print(f"Saving model to {output_dir}")
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    train()
