"""
Utility functions for the DeepSeek Language Model.

This module provides helper functions for text encoding/decoding and generation.
"""

import torch
import torch.nn.functional as F


def encode_text(text: str, tokenizer) -> torch.Tensor:
    """
    Encode text string to token IDs.
    
    Args:
        text (str): Input text to encode
        tokenizer: Tokenizer instance (e.g., tiktoken)
    
    Returns:
        torch.Tensor: Token IDs of shape (1, seq_len) with batch dimension
    
    Example:
        >>> import tiktoken
        >>> tokenizer = tiktoken.get_encoding("gpt2")
        >>> encoded = encode_text("Hello world", tokenizer)
        >>> print(encoded.shape)  # torch.Size([1, 2])
    """
    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    # Add a batch dimension to the encoded tensor
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor


def decode_text(token_ids: torch.Tensor, tokenizer) -> str:
    """
    Decode token IDs to text string.
    
    Args:
        token_ids (torch.Tensor): Token IDs of shape (batch_size, seq_len) or (seq_len,)
        tokenizer: Tokenizer instance (e.g., tiktoken)
    
    Returns:
        str: Decoded text string
    
    Example:
        >>> import tiktoken
        >>> tokenizer = tiktoken.get_encoding("gpt2")
        >>> token_ids = torch.tensor([[15496, 995]])  # "Hello world"
        >>> text = decode_text(token_ids, tokenizer)
        >>> print(text)  # "Hello world"
    """
    # Remove the batch dimension if present
    if token_ids.dim() > 1:
        flat = token_ids.squeeze(0)
    else:
        flat = token_ids
    
    return tokenizer.decode(flat.tolist())


def generate(
    model,
    idx: torch.Tensor,
    max_new_tokens: int,
    context_size: int,
    temperature: float = 1.0,
    top_k: int = None,
    eos_id: int = None,
) -> torch.Tensor:
    """
    Generate text using a language model with temperature and top-k sampling.
    
    This function generates new tokens autoregressively by repeatedly:
    1. Getting model predictions for the next token
    2. Sampling from the probability distribution
    3. Appending the sampled token to the sequence
    
    Args:
        model: The language model to use for generation
        idx (torch.Tensor): Input token indices of shape (batch_size, seq_len)
        max_new_tokens (int): Maximum number of new tokens to generate
        context_size (int): Number of previous tokens to use as context
                           (should match model's context_length)
        temperature (float): Temperature for sampling. Higher values increase diversity.
                           If 1.0, sample from unmodified distribution.
                           If 0.0, use greedy sampling (default: 1.0)
        top_k (int, optional): If specified, only sample from the top k most likely tokens.
                              Helps reduce low-probability token selection (default: None)
        eos_id (int, optional): Token ID that signals end of sequence. Generation stops 
                               if encountered (default: None)
    
    Returns:
        torch.Tensor: Generated token indices of shape (batch_size, seq_len + new_tokens)
    
    Example:
        >>> import tiktoken
        >>> from components.deepseek_model import DeepSeekModel
        >>> from constants import DEEPSEEK_CONFIG_SMALL
        >>> 
        >>> model = DeepSeekModel(DEEPSEEK_CONFIG_SMALL)
        >>> tokenizer = tiktoken.get_encoding("gpt2")
        >>> 
        >>> prompt = "Once upon a time"
        >>> input_ids = encode_text(prompt, tokenizer)
        >>> 
        >>> output_ids = generate(
        ...     model=model,
        ...     idx=input_ids,
        ...     max_new_tokens=50,
        ...     context_size=DEEPSEEK_CONFIG_SMALL["context_length"],
        ...     temperature=0.8,
        ...     top_k=40
        ... )
        >>> 
        >>> generated_text = decode_text(output_ids, tokenizer)
        >>> print(generated_text)
    """
    for _ in range(max_new_tokens):
        # Only use the last context_size tokens as input
        # This prevents exceeding the model's maximum context length
        idx_cond = idx if idx.size(1) <= context_size else idx[:, -context_size:]

        # Get logits from model without computing gradients
        with torch.no_grad():
            logits = model(idx_cond)
            
            # Handle both single return value and tuple return
            if isinstance(logits, tuple):
                logits = logits[0]  # Extract logits from (logits, loss) tuple

        # Only consider logits for next token prediction (last position)
        logits = logits[:, -1, :]  # (batch_size, vocab_size)

        # Apply top-k sampling if specified
        if top_k is not None:
            # Get the top k logits and their indices
            top_logits, _ = torch.topk(logits, top_k)
            # Find minimum value among top k
            min_val = top_logits[:, -1]
            # Set all logits below min_val to -inf to exclude them from sampling
            logits = torch.where(
                logits < min_val,
                torch.tensor(float("-inf"), device=logits.device, dtype=logits.dtype),
                logits
            )

        # Apply temperature sampling
        if temperature > 0.0:
            # Scale logits by temperature - higher temp = more uniform distribution
            logits = logits / temperature
            # Convert to probabilities
            probs = F.softmax(logits, dim=-1)
            # Sample from probability distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)
        else:
            # Greedy sampling - just take most likely token
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch_size, 1)

        # Stop if we generate an end-of-sequence token
        if eos_id is not None and (idx_next == eos_id).any():
            break

        # Append new token to sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch_size, seq_len + 1)

    return idx


def count_parameters(model) -> dict:
    """
    Count the number of parameters in the model.
    
    Args:
        model: PyTorch model
    
    Returns:
        dict: Dictionary with parameter counts
    
    Example:
        >>> from components.deepseek_model import DeepSeekModel
        >>> from constants import DEEPSEEK_CONFIG_SMALL
        >>> model = DeepSeekModel(DEEPSEEK_CONFIG_SMALL)
        >>> params = count_parameters(model)
        >>> print(f"Total: {params['total']:,}")
    """
    if hasattr(model, 'count_parameters'):
        return model.count_parameters()
    else:
        return {
            "total": sum(p.numel() for p in model.parameters()),
            "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
        }


def print_model_info(model, config: dict):
    """
    Print detailed information about the model.
    
    Args:
        model: The model to analyze
        config (dict): Model configuration
    """
    print("=" * 60)
    print("MODEL CONFIGURATION")
    print("=" * 60)
    
    print("\nArchitecture:")
    print(f"  Vocabulary size: {config['vocab_size']:,}")
    print(f"  Context length: {config['context_length']}")
    print(f"  Embedding dimension: {config['emb_dim']}")
    print(f"  Number of layers: {config['n_layers']}")
    print(f"  Number of heads: {config['n_heads']}")
    
    print("\nMLA (Multi-Head Latent Attention):")
    print(f"  KV latent dimension: {config['kv_latent_dim']}")
    compression_ratio = (2 * config['n_heads'] * config['emb_dim'] // config['n_heads']) / config['kv_latent_dim']
    print(f"  Memory compression: {compression_ratio:.1f}×")
    
    print("\nMoE (Mixture of Experts):")
    print(f"  Routed experts: {config['num_experts']}")
    print(f"  Shared experts: {config['num_shared_experts']}")
    print(f"  Top-k: {config['top_k']}")
    print(f"  Expert hidden dim: {config['expert_hidden_dim']}")
    
    params = count_parameters(model)
    print("\nParameters:")
    print(f"  Total: {params['total']:,}")
    if 'routed_experts' in params:
        print(f"  Routed experts: {params['routed_experts']:,}")
        print(f"  Shared experts: {params['shared_experts']:,}")
        print(f"  Non-expert: {params['non_expert']:,}")
        
        # Calculate active parameters per forward pass
        active_routed = (config['top_k'] * params['routed_experts']) / config['num_experts']
        active_total = active_routed + params['shared_experts'] + params['non_expert']
        efficiency = params['total'] / active_total
        
        print(f"\nEfficiency:")
        print(f"  Active per forward: {active_total:,.0f}")
        print(f"  Parameter efficiency: {efficiency:.2f}×")
    
    print("=" * 60)