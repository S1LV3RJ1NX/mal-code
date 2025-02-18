# Ref: https://github.com/rasbt/LLMs-from-scratch/blob/main/ch05/02_alternative_weight_loading/weight-loading-hf-transformers.ipynb
import torch
import tiktoken
import numpy as np
from transformers import GPT2Model

from components.gpt2 import GPT2
from utils import generate, encode_text, decode_text
from constants import GPT_CONFIG_124M

def assign_check(left, right):
    """Safely assign tensor values with shape validation.
    
    This helper function ensures safe assignment of pretrained weights by:
    1. Validating that the shapes of source and target tensors match exactly
    2. Creating a detached copy of the source tensor as a Parameter
    
    Args:
        left (torch.Tensor): Target tensor whose shape will be validated
        right (torch.Tensor): Source tensor containing values to assign
        
    Returns:
        torch.nn.Parameter: New parameter containing copied values from source
        
    Raises:
        ValueError: If shapes of left and right tensors don't match exactly
    """
    # Validate tensor shapes match before assignment
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    
    # Create a new Parameter with copied and detached values
    # clone() creates a new tensor
    # detach() removes connection to computation graph
    return torch.nn.Parameter(right.clone().detach())


def load_weights(gpt, gpt_hf):
    """Load pretrained weights from Hugging Face GPT-2 model into our custom implementation.
    
    This function carefully maps weights from the Hugging Face GPT-2 model architecture to our 
    custom implementation. The mapping handles differences in parameter naming and organization
    between the two implementations.

    Key weight transfers:
    1. Token and positional embeddings
    2. For each transformer block:
        - Multi-head attention weights (query, key, value projections)
        - Output projection weights
        - Feed-forward network weights
        - Layer normalization weights
    3. Final layer norm and output projection

    Args:
        gpt: Our custom GPT-2 model implementation
        gpt_hf: Hugging Face's pretrained GPT-2 model

    Raises:
        ValueError: If there is a shape mismatch between corresponding weights
    """
    # Get state dict from HF model containing all weights
    d = gpt_hf.state_dict()

    # Load token and positional embeddings
    gpt.embedding.weight = assign_check(gpt.embedding.weight, d["wte.weight"])
    gpt.positional_encoding.weight = assign_check(gpt.positional_encoding.weight, d["wpe.weight"])
    
    # Load weights for each transformer block
    for b in range(GPT_CONFIG_124M["n_layers"]):
        # Split concatenated Q,K,V weights from HF model
        q_w, k_w, v_w = np.split(d[f"h.{b}.attn.c_attn.weight"], 3, axis=-1)
        # Assign attention projection weights with transpose to match our shape
        gpt.transformer_blocks[b].mha.W_query.weight = assign_check(gpt.transformer_blocks[b].mha.W_query.weight, q_w.T)
        gpt.transformer_blocks[b].mha.W_key.weight = assign_check(gpt.transformer_blocks[b].mha.W_key.weight, k_w.T)
        gpt.transformer_blocks[b].mha.W_value.weight = assign_check(gpt.transformer_blocks[b].mha.W_value.weight, v_w.T)
    
        # Split and assign Q,K,V bias terms
        q_b, k_b, v_b = np.split(d[f"h.{b}.attn.c_attn.bias"], 3, axis=-1)
        gpt.transformer_blocks[b].mha.W_query.bias = assign_check(gpt.transformer_blocks[b].mha.W_query.bias, q_b)
        gpt.transformer_blocks[b].mha.W_key.bias = assign_check(gpt.transformer_blocks[b].mha.W_key.bias, k_b)
        gpt.transformer_blocks[b].mha.W_value.bias = assign_check(gpt.transformer_blocks[b].mha.W_value.bias, v_b)
    
        # Load attention output projection
        gpt.transformer_blocks[b].mha.out_proj.weight = assign_check(gpt.transformer_blocks[b].mha.out_proj.weight, d[f"h.{b}.attn.c_proj.weight"].T)
        gpt.transformer_blocks[b].mha.out_proj.bias = assign_check(gpt.transformer_blocks[b].mha.out_proj.bias, d[f"h.{b}.attn.c_proj.bias"])
    
        # Load feed-forward network weights
        # First linear layer (expansion)
        gpt.transformer_blocks[b].ffn.layers[0].weight = assign_check(gpt.transformer_blocks[b].ffn.layers[0].weight, d[f"h.{b}.mlp.c_fc.weight"].T)
        gpt.transformer_blocks[b].ffn.layers[0].bias = assign_check(gpt.transformer_blocks[b].ffn.layers[0].bias, d[f"h.{b}.mlp.c_fc.bias"])
        # Second linear layer (projection)
        gpt.transformer_blocks[b].ffn.layers[2].weight = assign_check(gpt.transformer_blocks[b].ffn.layers[2].weight, d[f"h.{b}.mlp.c_proj.weight"].T)
        gpt.transformer_blocks[b].ffn.layers[2].bias = assign_check(gpt.transformer_blocks[b].ffn.layers[2].bias, d[f"h.{b}.mlp.c_proj.bias"])
    
        # Load layer normalization weights
        gpt.transformer_blocks[b].layer_norm1.weight = assign_check(gpt.transformer_blocks[b].layer_norm1.weight, d[f"h.{b}.ln_1.weight"])
        gpt.transformer_blocks[b].layer_norm1.bias = assign_check(gpt.transformer_blocks[b].layer_norm1.bias, d[f"h.{b}.ln_1.bias"])
        gpt.transformer_blocks[b].layer_norm2.weight = assign_check(gpt.transformer_blocks[b].layer_norm2.weight, d[f"h.{b}.ln_2.weight"])
        gpt.transformer_blocks[b].layer_norm2.bias = assign_check(gpt.transformer_blocks[b].layer_norm2.bias, d[f"h.{b}.ln_2.bias"])
    
    # Load final layer norm and output projection
    gpt.layer_norm.weight = assign_check(gpt.layer_norm.weight, d[f"ln_f.weight"])
    gpt.layer_norm.bias = assign_check(gpt.layer_norm.bias, d[f"ln_f.bias"])
    # Note: Output projection shares weights with token embeddings
    gpt.linear_projection.weight = assign_check(gpt.linear_projection.weight, d["wte.weight"])

# Our implementation of the GPT2 model adjusted for the pretrained model
GPT_CONFIG_124M['qkv_bias'] = True
GPT_CONFIG_124M['drop_rate'] = 0.0
gpt = GPT2(GPT_CONFIG_124M)
gpt.eval()

# Load pretrained model from Hugging Face
gpt_hf = GPT2Model.from_pretrained("openai-community/gpt2", cache_dir="pretrained-checkpoints")
gpt_hf.eval()

print("Loading weights...")
load_weights(gpt, gpt_hf)
print("Weights loaded successfully!")

torch.manual_seed(123)

# Instantiate the tokenizer
tokenizer = tiktoken.get_encoding("gpt2")
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

token_ids = generate(
    model=gpt.to(device),
    idx=encode_text("Once upon a time, there was a", tokenizer).to(device),
    max_new_tokens=30,
    context_size=GPT_CONFIG_124M["context_len"],
    top_k=20,
    temperature=0.7
)

print("Output text:\n", decode_text(token_ids, tokenizer))





