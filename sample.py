import torch
import argparse
import json
from transformers import AutoModelForCausalLM, AutoTokenizer

# Model mapping
MODEL_MAPPING = {
    "gpt2-large": "gpt2-large",
    "llama-3.2-1b": "meta-llama/Llama-3.2-1B",
    "qwen3-0.6b": "Qwen/Qwen3-0.6B-Base"
}

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

def load_model_and_tokenizer(model_key):
    """Loads the model and tokenizer based on the key."""
    if model_key not in MODEL_MAPPING:
        raise ValueError(f"Unknown model: {model_key}. Available models: {list(MODEL_MAPPING.keys())}")
    
    model_id = MODEL_MAPPING[model_key]
    print(f"Loading model: {model_id}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
    
    return model, tokenizer

def generate_text(model_key, num_completions):
    model, tokenizer = load_model_and_tokenizer(model_key)

    # Determine BOS token
    if tokenizer.bos_token_id is not None:
        bos_token_id = tokenizer.bos_token_id
    elif tokenizer.eos_token_id is not None:
        bos_token_id = tokenizer.eos_token_id
    else:
        # Fallback or error if neither exists (unlikely for causal LM)
        raise ValueError("Tokenizer does not have a BOS or EOS token defined.")
    
    print(f"Generating {num_completions} completions starting with BOS token (ID: {bos_token_id})...")

    # Create input with just the BOS token
    input_ids = torch.tensor([[bos_token_id]]).to(device)

    # Generate completions
    output = model.generate(
        input_ids,
        max_new_tokens=100,
        num_return_sequences=num_completions,
        do_sample=True,
        temperature=1.0,
        top_p=1.0,
        top_k=0,
        pad_token_id=tokenizer.eos_token_id
    )

    completions = []
    for sample_output in output:
        text = tokenizer.decode(sample_output, skip_special_tokens=True)
        completions.append(text)

    output_file = f"samples/{model_key}.json"
    with open(output_file, "w") as f:
        json.dump(completions, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(completions)} completions to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate text completions using a specified model.")
    parser.add_argument(
        "--model", 
        type=str, 
        default="gpt2", 
        choices=list(MODEL_MAPPING.keys()),
        help="The model to use for generation."
    )
    parser.add_argument(
        "--num_completions", 
        type=int, 
        default=10, 
        help="The number of completions to generate."
    )
    
    args = parser.parse_args()
    generate_text(args.model, args.num_completions)
