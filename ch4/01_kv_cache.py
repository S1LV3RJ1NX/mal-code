"""
This script demonstrates the impact of KV (Key-Value) caching on transformer model inference speed.
It uses GPT-2 to generate text both with and without KV caching enabled, measuring the time difference.

KV caching stores the computed key and value tensors from previous tokens, avoiding redundant
recomputation during autoregressive generation. This can significantly speed up inference.
"""

import threading
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from time import time

# Load GPT-2 model and tokenizer from HuggingFace
tokenizer = AutoTokenizer.from_pretrained('gpt2')
model = AutoModelForCausalLM.from_pretrained('gpt2')

# Define input prompt and convert to token IDs
prompt = "Machine Learning is"
tokens = tokenizer.encode(prompt, return_tensors="pt")  # Shape: [1, sequence_length]

def stream_output(use_cache: bool):
    """
    Generates text from the model while streaming tokens and measuring generation time.
    
    Args:
        use_cache (bool): Whether to use KV caching during generation
            - True: Cache key/value tensors to avoid recomputation
            - False: Recompute key/value tensors for each new token
    
    The function uses threading to run model generation in the background while
    streaming tokens to stdout in real-time. It measures and reports the total
    generation time.
    """
    # Initialize streamer for token-by-token output
    streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True)
    
    # Create thread for model generation to avoid blocking
    thread = threading.Thread(
        target=model.generate,
        kwargs={
            "input_ids": tokens,
            "max_new_tokens": 100,  # Generate 100 new tokens
            "use_cache": use_cache,  # Toggle KV caching
            "streamer": streamer,    # Stream tokens as they're generated
        }
    )
    thread.start()
    
    # Stream and time the token generation
    start_time = time()
    for token in streamer:
        print(token, end="", flush=True)  # Print tokens immediately
    end_time = time()
    
    thread.join()  # Wait for generation to complete
    elapsed = end_time - start_time
    print(f"\n\nUse cache = {use_cache}, Time taken: {elapsed:.3f} seconds\n")

if __name__ == "__main__":
    # Run generation with KV caching enabled
    print("=== With KV caching ===")
    stream_output(use_cache=True)

    # Run generation with KV caching disabled
    print("=== Without KV caching ===")
    stream_output(use_cache=False)
