import pandas as pd
import json
import os

def preprocess_data(input_file, output_file):
    """
    Reads transcript.csv and splits each transcript into its own JSONL record in output.txt.
    Each transcript becomes one training example for fine-tuning.
    """
    print(f"Loading data from {input_file}...")
    # Read CSV, skipping bad lines if any
    try:
        df = pd.read_csv(input_file, header=None, names=["file_id", "text"], sep="\t", engine="python")
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    with open(output_file, 'w', encoding='utf-8') as f:
        for _, row in df.iterrows():
            text = row['text']
            if not isinstance(text, str) or not text.strip():
                continue
                
            # Create training example - one JSONL record per transcript
            # Format for fine-tuning with instruction-following models
            example = {
                "messages": [
                    {"role": "system", "content": "You are a professional stand-up comedian."},
                    {"role": "user", "content": "Perform a stand-up comedy routine."},
                    {"role": "assistant", "content": text.strip()}
                ]
            }
            
            f.write(json.dumps(example) + '\n')
                
    print(f"Preprocessed data saved to {output_file}")
    print(f"Total records: {len(df)}")

if __name__ == "__main__":
    # Paths relative to the script location (got_ourPDF/fine_tune/)
    input_csv = "../transcript.csv"
    output_jsonl = "../output.txt"
    
    if not os.path.exists(input_csv):
        print(f"File {input_csv} not found.")
    else:
        preprocess_data(input_csv, output_jsonl)
