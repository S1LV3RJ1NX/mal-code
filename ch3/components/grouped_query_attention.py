# Ref: https://github.com/rasbt/LLMs-from-scratch/blob/main/ch05/07_gpt_to_llama/standalone-llama32.ipynb
from components.common import nn, torch
from components.rope import RoPE


class SharedBuffers:
    """
    A class to manage shared buffers for attention mechanisms, specifically for
    the Grouped Query Attention mechanism. This class ensures that buffers are
    created only once for a given configuration, optimizing memory usage.

    Attributes:
        _buffers (dict): A dictionary to store precomputed buffers based on
                         unique keys derived from input parameters.
    """

    _buffers = {}

    @staticmethod
    def get_buffers(
        context_length: int,
        head_dim: int,
        rope_base: int,
        freq_config: dict = None,
        dtype: torch.dtype = torch.float32,
    ):
        """
        Retrieves or computes the necessary buffers for attention calculations.

        Args:
            context_length (int): The length of the context (number of tokens).
            head_dim (int): The dimensionality of each attention head.
            rope_base (int): The base for the rotary positional encoding.
            freq_config (dict): Configuration for frequency settings.
            dtype (torch.dtype): The data type for the buffers (default: float32).

        Returns:
            tuple: A tuple containing the mask, cosine, and sine buffers.
        """
        key = (
            context_length,
            head_dim,
            rope_base,
            tuple(freq_config.values()) if freq_config else freq_config,
            dtype,
        )

        if key not in SharedBuffers._buffers:
            # Create or fetch the buffers if they do not exist
            mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
            cos, sin = RoPE.precompute_rope(
                head_dim, rope_base, context_length, freq_config
            )
            if dtype is not None:
                cos = cos.to(dtype)
                sin = sin.to(dtype)
            SharedBuffers._buffers[key] = (mask, cos, sin)

        return SharedBuffers._buffers[key]


class GroupedQueryAttention(nn.Module):
    """
    Implements the Grouped Query Attention mechanism, which allows for
    efficient attention computation by grouping keys and values.

    Args:
        d_in (int): Input dimension.
        d_out (int): Output dimension.
        context_length (int): The length of the context (number of tokens).
        num_heads (int): The number of attention heads.
        num_kv_groups (int): The number of key-value groups.
        rope_base (int): The base for the rotary positional encoding.
        rope_config (dict): Configuration for rotary positional encoding (default: None).
        dtype (torch.dtype): The data type for the model parameters (default: None).
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        context_length: int,
        num_heads: int,
        num_kv_groups: int,
        rope_base: int,
        rope_config: dict = None,
        dtype: torch.dtype = None,
    ):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        assert (
            num_heads % num_kv_groups == 0
        ), "num_heads must be divisible by num_kv_groups"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        # Linear layers for keys, values, and queries
        # Shape: (d_in, num_kv_groups * head_dim)
        self.W_key = nn.Linear(
            d_in, num_kv_groups * self.head_dim, bias=False, dtype=dtype
        )
        self.W_value = nn.Linear(
            d_in, num_kv_groups * self.head_dim, bias=False, dtype=dtype
        )
        self.num_kv_groups = num_kv_groups
        self.group_size = num_heads // num_kv_groups

        self.W_query = nn.Linear(d_in, d_out, bias=False, dtype=dtype)
        self.out_proj = nn.Linear(d_out, d_out, bias=False, dtype=dtype)

        # Fetch buffers using SharedBuffers
        mask, cos, sin = SharedBuffers.get_buffers(
            context_length, self.head_dim, rope_base, rope_config, dtype
        )
        self.register_buffer("mask", mask)
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)

    def forward(self, x):
        """
        Forward pass for the Grouped Query Attention mechanism.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_tokens, d_in).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_tokens, d_out).
        """
        b, num_tokens, d_in = x.shape

        # Compute queries, keys, and values
        # Shape: (b, num_tokens, d_out)
        queries = self.W_query(x)
        # Shape: (b, num_tokens, num_kv_groups * head_dim)
        keys = self.W_key(x)
        # Shape: (b, num_tokens, num_kv_groups * head_dim)
        values = self.W_value(x)

        # Reshape queries, keys, and values for multi-head attention
        # Shape: (b, num_tokens, num_heads, head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        # Shape: (b, num_tokens, num_kv_groups, head_dim)
        keys = keys.view(b, num_tokens, self.num_kv_groups, self.head_dim)
        # Shape: (b, num_tokens, num_kv_groups, head_dim)
        values = values.view(b, num_tokens, self.num_kv_groups, self.head_dim)

        # Transpose keys, values, and queries for attention computation
        # Shape: (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        # Shape: (b, num_heads, num_tokens, head_dim)
        values = values.transpose(1, 2)
        # Shape: (b, num_query_groups, num_tokens, head_dim)
        queries = queries.transpose(1, 2)

        # Apply Rotary Positional Encoding (RoPE)
        # Shape: (b, num_heads, num_tokens, head_dim)
        keys = RoPE.apply_rope(keys, self.cos, self.sin)
        # Shape: (b, num_heads, num_tokens, head_dim)
        queries = RoPE.apply_rope(queries, self.cos, self.sin)

        # Expand keys and values to match the number of heads
        # Shape: (b, num_heads, num_tokens, head_dim)
        keys = keys.repeat_interleave(self.group_size, dim=1)
        # Shape: (b, num_heads, num_tokens, head_dim)
        values = values.repeat_interleave(self.group_size, dim=1)

        # Compute scaled dot-product attention with a causal mask
        # Shape: (b, num_heads, num_tokens, num_tokens)
        attn_scores = queries @ keys.transpose(2, 3)

        # Create a boolean mask to prevent attending to future tokens
        # Shape: (num_tokens, num_tokens)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        # Apply the mask to the attention scores
        # Shape: (b, num_heads, num_tokens, num_tokens)
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        # Compute attention weights
        # Shape: (b, num_heads, num_tokens, num_tokens)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        assert keys.shape[-1] == self.head_dim

        # Compute the context vector
        # Shape: (b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)

        # Combine heads and project to output dimension
        # Shape: (b, num_tokens, d_out)
        context_vec = context_vec.reshape(b, num_tokens, self.d_out)
        # Shape: (b, num_tokens, d_out)
        context_vec = self.out_proj(context_vec)

        return context_vec


if __name__ == "__main__":
    import os
    import sys

    # Add the root directory of your project to the Python path
    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    sys.path.append(project_root)
    from ch2.components.multi_head_attention import MultiHeadAttention

    # To illustrate the parameter savings, consider the following multi-head attention example from the GPT and Llama 2 code:
    # Settings
    batch_size = 1
    context_len = 3000
    max_context_len = 8192
    embed_dim = 4096
    num_heads = 32

    example_batch = torch.randn((batch_size, context_len, embed_dim))
    mha = MultiHeadAttention(
        d_in=embed_dim, d_out=embed_dim, num_heads=num_heads, dropout=0.0
    )

    mha(example_batch, example_batch, example_batch)

    print("=" * 50)
    print("Multi-Head Attention")
    print("W_key:", mha.W_key.weight.shape)
    print("W_value:", mha.W_value.weight.shape)
    print("W_query:", mha.W_query.weight.shape)
    print("=" * 50)

    # Now, if we use grouped-query attention instead, with 8 kv-groups (that's how many Llama 3 8B uses), we can see that the number of rows of the key and value matrices are reduced by a factor of 4 (because 32 attention heads divided by 8 kv-groups is 4)
    gqa = GroupedQueryAttention(
        d_in=embed_dim,
        d_out=embed_dim,
        context_length=max_context_len,
        num_heads=num_heads,
        num_kv_groups=8,
        rope_base=500_000,
    )

    gqa(example_batch)

    print("=" * 50)
    print("Grouped Query Attention")
    print("W_key:", gqa.W_key.weight.shape)
    print("W_value:", gqa.W_value.weight.shape)
    print("W_query:", gqa.W_query.weight.shape)
    print("=" * 50)

    # Lastly, let's compare the number of parameters below:
    print("Total number of parameters:")

    mha_total_params = sum(p.numel() for p in mha.parameters())
    print(f"MHA: {mha_total_params:,}")

    gqa_total_params = sum(p.numel() for p in gqa.parameters())
    print(f"GQA: {gqa_total_params:,}")
