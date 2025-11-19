import pandas as pd
import json
import os
from transformers import AutoTokenizer

def preprocess_data(input_file, output_file, model_id="meta-llama/Meta-Llama-3.1-8B-Instruct", max_length=2048, overlap=256):
    """
    Reads transcript.csv, splits long transcripts into chunks, and saves as JSONL.
    """
    print(f"Loading data from {input_file}...")
    # Read CSV, skipping bad lines if any
    try:
        df = pd.read_csv(input_file, header=None, names=["file_id", "text"], sep="\t", engine="python")
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for _, row in df.iterrows():
            text = row['text']
            if not isinstance(text, str):
                continue
                
            # Tokenize
            tokens = tokenizer.encode(text, add_special_tokens=False)
            
            # Sliding window chunking
            start = 0
            while start < len(tokens):
                end = min(start + max_length, len(tokens))
                chunk_tokens = tokens[start:end]
                chunk_text = tokenizer.decode(chunk_tokens)
                
                # Create training example
                # Format as a simple completion or instruction
                # For standup, we might just want the model to continue text, 
                # but Llama 3 is instruct-tuned. Let's format it as a "Generate standup" request 
                # or just raw text completion. 
                # Given the goal is style mimicry, raw text (pretraining style) or 
                # "Write a standup routine about..." -> [text] is good.
                # Since we don't have topics labeled, we'll use a generic prompt or raw text.
                # Let's use a generic instruction to align with Instruct model.
                
                example = {
                    "messages": [
                        {"role": "system", "content": "You are a professional stand-up comedian."},
                        {"role": "user", "content": "Perform a stand-up comedy routine."},
                        {"role": "assistant", "content": chunk_text}
                    ]
                }
                
                f.write(json.dumps(example) + '\n')
                
                if end == len(tokens):
                    break
                
                start += (max_length - overlap)
                
    print(f"Preprocessed data saved to {output_file}")

if __name__ == "__main__":
    input_csv = "transcript.csv"
    output_jsonl = "fine_tune/train.jsonl"
    
    if not os.path.exists(input_csv):
        print(f"File {input_csv} not found.")
    else:
        preprocess_data(input_csv, output_jsonl)
