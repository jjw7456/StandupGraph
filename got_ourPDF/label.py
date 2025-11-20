import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import json
import os
from tqdm import tqdm
import re

# Configuration
MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
INPUT_FILE = "transcript.txt"
OUTPUT_FILE = "fine_tune/labeled_dataset.json"
# Set to None to process all lines, or a number for testing
MAX_SAMPLES = 1 

def load_model():
    print(f"Loading model {MODEL_ID}...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto"
    )
    
    return tokenizer, model

def create_prompt(transcript_text):
    system_prompt = (
        "You are a professional stand-up comedian and comedy analyst. "
        "Your task is to label the components of a stand-up comedy transcript. "
        "Each component must have a type: Setup, Incongruity, Punchline, or Callback.\n"
        "Choose the most appropriate type for each component based on context.\n"
        "Follow a format of { (TYPE): \"SENTENCE\"}\n"
        "Example Output:\n"
        "(Setup): \"I tried to cook healthy last week.\"\n"
        "(Incongruity): \"Turns out my idea of healthy is microwaving kale chips.\"\n"
        "(Punchline): \"Now my smoke alarm has abs.\"\n"
        "(Callback): \"Even my smoke alarm started giving nutrition advice\""
    )
    
    user_prompt = f"Generate a stand-up comedy transcript based on the following text, labeling each part.\n\nTranscript:\n{transcript_text}"
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    return messages

def parse_labeled_output(output_text):
    """
    Parses lines like '(Setup): "Sentence content"' into structured objects.
    """
    parsed_items = []
    # Regex to capture (Type): "Content" or (Type): Content
    # We handle potential variations in spacing or quotes
    pattern = r'\((Setup|Incongruity|Punchline|Callback)\):\s*"?([^"\n]+)"?'
    
    lines = output_text.split('\n')
    for line in lines:
        line = line.strip()
        match = re.search(pattern, line, re.IGNORECASE)
        if match:
            item_type = match.group(1).capitalize() # Ensure proper casing
            content = match.group(2).strip()
            parsed_items.append({
                "type": item_type,
                "text": content
            })
    
    return parsed_items

def process_transcripts():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return

    tokenizer, model = load_model()
    
    output_data = []
    
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    if MAX_SAMPLES:
        lines = lines[:MAX_SAMPLES]
        print(f"Processing first {MAX_SAMPLES} samples only.")

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    for line in tqdm(lines, desc="Processing transcripts"):
        # Remove file ID (everything before the first tab or space if tab not present)
        # Looking at the file, it seems to be "ID\t Text"
        parts = line.split('\t', 1)
        if len(parts) < 2:
            # Fallback if no tab
            parts = line.split(' ', 1)
        
        if len(parts) < 2:
            continue # Skip empty or malformed lines
            
        transcript_text = parts[1].strip()
        if not transcript_text:
            continue

        # Create prompt
        messages = create_prompt(transcript_text)
        input_ids = tokenizer.apply_chat_template(
            messages, 
            add_generation_prompt=True, 
            return_tensors="pt"
        ).to(model.device)

        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=2048, # Adjust as needed
                eos_token_id=terminators,
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
            )
        
        # Decode response
        response = outputs[0][input_ids.shape[-1]:]
        decoded_output = tokenizer.decode(response, skip_special_tokens=True)
        
        # Parse the output
        labeled_components = parse_labeled_output(decoded_output)
        
        # Construct the final training example
        # The user wants the assistant content to be the LIST of objects
        # This is a bit unusual for standard chat templates (usually string), 
        # but we will store it as requested in the JSON structure.
        # When fine-tuning, the data loader will need to handle this list -> string conversion 
        # or we should store it as a string representation of the list if that's what they mean.
        # However, the user explicitly showed: "content": [ { "type": ... } ]
        # We will save it exactly as requested.
        
        training_example = {
            "messages": [
                {"role": "system", "content": "You are a professional stand-up comedian."},
                {"role": "user", "content": "Generate a stand-up comedy transcript.\nEach component must have a type: Setup, Incongruity, Punchline, or Callback.\nChoose the most appropriate type for each component based on context.\nFollow a format of {Component Number (TYPE): \"SENTENCE\"."},
                {"role": "assistant", "content": labeled_components}
            ]
        }
        
        output_data.append(training_example)
        
        # Save incrementally (optional, but good for long processes) or just at end
        # For now, let's just append to list and save at end to keep valid JSON
    
    print(f"Saving {len(output_data)} labeled examples to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        # Saving as a JSON object with a "messages" key? 
        # Or a list of these objects? The user example showed a single object.
        # Usually fine-tuning data is JSONL (one json per line).
        # The user said "our data above should be translated to json(like below) when training starts."
        # and showed a single JSON object. 
        # I will save as JSONL for standard fine-tuning compatibility, 
        # where each line is one of those objects.
        for entry in output_data:
            f.write(json.dumps(entry) + '\n')

if __name__ == "__main__":
    process_transcripts()
