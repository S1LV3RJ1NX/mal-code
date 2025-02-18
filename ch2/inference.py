import torch
import tiktoken
from pathlib import Path

from components.gpt2 import GPT2
from constants import GPT_CONFIG_124M
from utils import generate, encode_text, decode_text

def load_checkpoint(model, checkpoint_path):
    """Load a saved model checkpoint.
    
    Args:
        model (GPT2): The model instance to load weights into
        checkpoint_path (str): Path to the checkpoint file
        
    Returns:
        dict: Checkpoint information including epoch and validation loss
    """
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")
        
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']} "
          f"with validation loss {checkpoint['val_loss']:.4f}")
    
    return checkpoint

def main():
    # Initialize model configuration
    config = GPT_CONFIG_124M.copy()
    config['drop_rate'] = 0.0  # Disable dropout for inference
    
    # Initialize model and load checkpoint
    model = GPT2(config)
    checkpoint_path = "checkpoints/best_model.pt"
    load_checkpoint(model, checkpoint_path)
    
    # Set model to evaluation mode
    model.eval()
    
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() 
                         else "mps" if torch.backends.mps.is_available() 
                         else "cpu")
    model.to(device)
    
    # Initialize tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")
    
    # Set generation parameters
    generation_config = {
        'max_new_tokens': 100,    # Maximum number of tokens to generate
        'temperature': 0.7,       # Higher values increase randomness
        'top_k': 40,             # Number of top tokens to consider for sampling
        'context_size': config['context_len']
    }
    
    # Interactive generation loop
    print("\nGPT-2 Text Generation")
    print("Initial text: (or 'quit' to exit):")
    
    while True:
        # Get user input
        prompt = input("\nInitial text:> ").strip()
        
        if prompt.lower() == 'quit':
            break
            
        if not prompt:
            print("Please enter a prompt!")
            continue
            
        try:
            # Encode and generate
            input_ids = encode_text(prompt, tokenizer).to(device)
            
            with torch.no_grad():
                output_ids = generate(
                    model=model,
                    idx=input_ids,
                    max_new_tokens=generation_config['max_new_tokens'],
                    temperature=generation_config['temperature'],
                    top_k=generation_config['top_k'],
                    context_size=generation_config['context_size']
                )
            
            # Decode and print the generated text
            generated_text = decode_text(output_ids, tokenizer)
            print("\nGenerated Text:")
            print("-" * 50)
            print(generated_text)
            print("-" * 50)
            
        except Exception as e:
            print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main() 