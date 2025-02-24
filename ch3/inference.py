# https://github.com/rasbt/LLMs-from-scratch/blob/main/ch05/07_gpt_to_llama32.ipynb
import torch
from components.llama3 import Llama3
from load_pretrained import load_pretrained_weights
from transformers import AutoTokenizer
from utils import check_model, decode_text, encode_text, generate, rescale_theta

TOKENIZER = AutoTokenizer.from_pretrained("./pretrained-weights")

# Encoding and Decoding test
print("=" * 50)
print("Encoding text...")
text = "Hello, how are you?"
tokens = encode_text(text, TOKENIZER)
decoded = decode_text(tokens, TOKENIZER)
print(f"Encoded: {tokens}")
print(f"Decoded: {decoded}")
print("=" * 50)

# Model configuration
LLAMA3_3B_CONFIG = {
    # Vocabulary size
    "vocab_size": 128_256,
    # Context length
    "context_length": 131_072,
    # Embedding dimension
    "emb_dim": 3072,
    # Number of attention heads
    "n_heads": 24,
    # Number of layers
    "n_layers": 28,
    # Number of layers
    "hidden_dim": 8192,
    # Key-Value groups for grouped-query attention
    "n_kv_groups": 8,
    # The base in RoPE's "theta"
    "rope_base": 500_000.0,
    # Lower-precision dtype to reduce memory usage
    "dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    # RoPE frequency scaling
    "rope_freq": {
        "factor": 32.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_context_length": 8192,
    },
}

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print("=" * 50)
print("Loading model...")

##### COMMENT THIS SECTION IF YOU DON'T WANT TO REDUCE CONTEXT LENGTH #####
LLAMA3_3B_CONFIG["context_length"] = 2048

print("Rescaling RoPE theta")
LLAMA3_3B_CONFIG["rope_base"] = rescale_theta(
    LLAMA3_3B_CONFIG["rope_base"], 8192, LLAMA3_3B_CONFIG["context_length"]
)
print("New RoPE theta:", LLAMA3_3B_CONFIG["rope_base"])
############################################################################

model = Llama3(LLAMA3_3B_CONFIG)
check_model(model)
load_pretrained_weights(model, LLAMA3_3B_CONFIG)
model.to(device)
print("Model loaded with pretrained weights...\n")


torch.manual_seed(123)
while True:
    try:
        # Get user input
        prompt = input("\nInitial text:> ").strip()

        if prompt.lower() == "quit":
            break

        # Generate text
        token_ids = generate(
            model=model,
            idx=encode_text(prompt, TOKENIZER).to(device),
            max_new_tokens=25,
            context_size=LLAMA3_3B_CONFIG["context_length"],
            top_k=1,
            temperature=0.0,
        )
        print("Output text:\n", decode_text(token_ids, TOKENIZER))

    except KeyboardInterrupt:
        print("Exiting...")
        break
    except Exception as e:
        print(f"Error: {e}")
        break
