import torch

def encode_text(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    # Add a batch dimension to the encoded tensor
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor

def decode_text(token_ids, tokenizer):
    # Remove the batch dimension from the token IDs
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())


def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):
    """Generate text using a language model with temperature and top-k sampling.
    
    Args:
        model: The language model to use for generation
        idx (torch.Tensor): Input token indices of shape (batch_size, seq_len)
        max_new_tokens (int): Maximum number of new tokens to generate
        context_size (int): Number of previous tokens to use as context
        temperature (float): Temperature for sampling. Higher values increase diversity.
                           If 0.0, use greedy sampling (default: 0.0)
        top_k (int): If specified, only sample from the top k most likely tokens (default: None)
        eos_id (int): Token ID that signals end of sequence. Generation stops if encountered.
                      (default: None)
    
    Returns:
        torch.Tensor: Generated token indices of shape (batch_size, seq_len + new_tokens)
    """
    for _ in range(max_new_tokens):
        # Only use the last context_size tokens as input
        idx_cond = idx[:, -context_size:]
        
        # Get logits from model without computing gradients
        with torch.no_grad():
            logits = model(idx_cond)
        
        # Only consider logits for next token prediction
        logits = logits[:, -1, :]
        
        # Apply top-k sampling if specified
        if top_k is not None:
            # Get the top k logits and their indices
            top_logits, _ = torch.topk(logits, top_k)
            # Find minimum value among top k
            min_val = top_logits[:, -1]
            # Set all logits below min_val to -inf to exclude them from sampling
            logits = torch.where(
                logits < min_val,
                torch.tensor(float('-inf')).to(logits.device),
                logits
            )
        
        # Apply temperature sampling if temperature > 0
        if temperature > 0.0:
            # Scale logits by temperature - higher temp = more uniform distribution
            logits = logits / temperature
            # Convert to probabilities
            probs = torch.softmax(logits, dim=-1)
            # Sample from probability distribution
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            # Greedy sampling - just take most likely token
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
            
        # Stop if we generate an end-of-sequence token
        if idx_next == eos_id:
            break
            
        # Append new token to sequence
        idx = torch.cat((idx, idx_next), dim=1)
        
    return idx