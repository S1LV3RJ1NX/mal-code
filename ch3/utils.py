# Ref: https://github.com/rasbt/LLMs-from-scratch/blob/main/ch05/07_gpt_to_llama/standalone-llama32.ipynb
from components.common import nn, torch
from transformers import AutoTokenizer


def rescale_theta(
    theta_old: float, context_length_old: int, context_length_new: int
) -> float:
    """
    Rescales the theta parameter based on the old and new context lengths.

    Args:
        theta_old (float): The original theta value to be rescaled.
        context_length_old (int): The original context length associated with theta_old.
        context_length_new (int): The new context length to which theta_old will be adjusted.

    Returns:
        float: The rescaled theta value.
    """
    scaling_factor = context_length_new / context_length_old
    theta_new = theta_old * scaling_factor
    return theta_new


def check_buffers(model: nn.Module) -> None:
    """
    Checks the consistency of buffer tensors across transformer blocks in the model.

    This function asserts that the attention masks and rotary positional encoding tensors
    (cosine and sine) are the same across the first and last transformer blocks.

    Args:
        model (nn.Module): The model containing transformer blocks to check.

    Raises:
        AssertionError: If any of the checks fail.
    """
    assert model.transformer_blocks[0].att.mask is model.transformer_blocks[-1].att.mask
    assert model.transformer_blocks[0].att.cos is model.transformer_blocks[-1].att.cos
    assert model.transformer_blocks[0].att.sin is model.transformer_blocks[-1].att.sin
    print("Buffers check passed...")


def count_parameters(model: nn.Module) -> int:
    """
    Counts the total number of parameters in the model.

    Args:
        model (nn.Module): The model for which to count parameters.

    Returns:
        int: The total number of parameters in the model.
    """
    return sum(p.numel() for p in model.parameters())


def count_unique_parameters(model: nn.Module) -> int:
    """
    Counts the number of unique parameters in the model, excluding the embedding weights.

    Args:
        model (nn.Module): The model for which to count unique parameters.

    Returns:
        int: The total number of unique parameters in the model.
    """
    return sum(p.numel() for p in model.parameters()) - model.embedding.weight.numel()


def model_memory_size(model, input_dtype=torch.float32) -> float:
    """
    Calculates the memory size required by the model in gigabytes.

    This includes the size of parameters, gradients, and buffers.

    Args:
        model (nn.Module): The model for which to calculate memory size.
        input_dtype (torch.dtype): The data type of the model parameters (default: torch.float32).

    Returns:
        float: The total memory size in gigabytes.
    """
    total_params = 0
    total_grads = 0
    for param in model.parameters():
        # Calculate total number of elements per parameter
        param_size = param.numel()
        total_params += param_size
        # Check if gradients are stored for this parameter
        if param.requires_grad:
            total_grads += param_size

    # Calculate buffer size (non-parameters that require memory)
    total_buffers = sum(buf.numel() for buf in model.buffers())

    # Size in bytes = (Number of elements) * (Size of each element in bytes)
    # We assume parameters and gradients are stored in the same type as input dtype
    element_size = torch.tensor(0, dtype=input_dtype).element_size()
    total_memory_bytes = (total_params + total_grads + total_buffers) * element_size

    # Convert bytes to gigabytes
    total_memory_gb = total_memory_bytes / (1024**3)

    return total_memory_gb


def check_model(model: nn.Module) -> None:
    """
    Performs a series of checks on the model, including buffer consistency and parameter counts.

    Args:
        model (nn.Module): The model to check.
    """
    print("=" * 50)
    print("Checking buffers")
    check_buffers(model)
    print("=" * 50)

    print("Counting parameters")
    print(f"Total number of parameters: {count_parameters(model):,}")
    print(f"\nTotal number of unique parameters: {count_unique_parameters(model):,}")
    print("=" * 50)

    print("=" * 50)
    print(
        f"float32 (PyTorch default): {model_memory_size(model, input_dtype=torch.float32):.2f} GB"
    )
    if torch.cuda.is_available():
        print(
            f"bfloat16: {model_memory_size(model, input_dtype=torch.bfloat16):.2f} GB"
        )
    print("=" * 50)


# Same from chapter 2
def encode_text(text: str, tokenizer: AutoTokenizer) -> torch.Tensor:
    """
    Encodes a given text string into a tensor of token indices.

    Args:
        text (str): The input text to encode.
        tokenizer (AutoTokenizer): The tokenizer to use for encoding.

    Returns:
        torch.Tensor: A tensor containing the encoded token indices with an added batch dimension.
    """
    encoded = tokenizer.encode(text)
    # Add a batch dimension to the encoded tensor
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor


# Same from chapter 2
def decode_text(token_ids: torch.Tensor, tokenizer: AutoTokenizer) -> str:
    """
    Decodes a tensor of token indices back into a string.

    Args:
        token_ids (torch.Tensor): The tensor containing token indices to decode.
        tokenizer (AutoTokenizer): The tokenizer to use for decoding.

    Returns:
        str: The decoded text string.
    """
    # Remove the batch dimension from the token IDs
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist(), skip_special_tokens=True)


# Same from chapter 2
def generate(
    model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None
):
    """
    Generate text using a language model with temperature and top-k sampling.

    Args:
        model: The language model to use for generation.
        idx (torch.Tensor): Input token indices of shape (batch_size, seq_len).
        max_new_tokens (int): Maximum number of new tokens to generate.
        context_size (int): Number of previous tokens to use as context.
        temperature (float): Temperature for sampling. Higher values increase diversity.
                             If 0.0, use greedy sampling (default: 0.0).
        top_k (int): If specified, only sample from the top k most likely tokens (default: None).
        eos_id (int): Token ID that signals end of sequence. Generation stops if encountered (default: None).

    Returns:
        torch.Tensor: Generated token indices of shape (batch_size, seq_len + new_tokens).
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
                logits < min_val, torch.tensor(float("-inf")).to(logits.device), logits
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
