from components.common import nn, np, torch


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism that allows parallel attention computations.

    This implements the multi-head attention mechanism from "Attention is All You Need" with
    modifications for causal/autoregressive attention as used in GPT models. The attention
    mechanism projects inputs into query, key and value representations, splits them into
    multiple heads, computes scaled dot-product attention independently for each head, and
    combines the results.

    Key features:
    - Splits attention into multiple heads that can attend to different parts of sequences
    - Uses learned projections for queries, keys and values
    - Implements causal masking to prevent attending to future tokens
    - Applies dropout for regularization
    - Combines multiple heads via learned projection

    Args:
        d_in (int): Input dimension size
        d_out (int): Output dimension size (must be divisible by num_heads)
        dropout (float): Dropout probability
        num_heads (int): Number of attention heads
        qkv_bias (bool, optional): Whether to use bias in Q,K,V projections. Defaults to False.
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        dropout: float,
        num_heads: int,
        qkv_bias: bool = False,
    ):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        self.d_in = d_in
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads  # Dimension of each attention head

        # Learned projection matrices for query, key, value
        # Each projects from d_in to d_out with optional bias
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)  # Query projection (Wq)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)  # Key projection (Wk)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)  # Value projection (Wv)

        # Final projection to combine heads back to d_out dimension
        self.out_proj = nn.Linear(d_out, d_out, bias=qkv_bias)  # Output projection (Wo)

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

    def causal_mask(self, size: int, device: torch.device) -> torch.Tensor:
        """Generates a causal/autoregressive attention mask.

        Creates a triangular mask that prevents positions from attending to future positions,
        ensuring autoregressive property needed for language modeling.

        Args:
            size (int): Sequence length to generate mask for

        Returns:
            torch.Tensor: Boolean tensor of shape (1, size, size) where True values allow
                         attention and False values prevent it. Upper triangle is False.
        """
        # Create upper triangular matrix of ones (1s above diagonal, 0s below)
        mask = torch.triu(torch.ones(1, size, size, device=device), diagonal=1).int()
        # Convert to boolean mask where 1->False (block attention) and 0->True (allow attention)
        return mask == 0

    @staticmethod
    def calculate_attention_scores(
        queries: torch.Tensor,
        keys: torch.Tensor,
        mask: torch.Tensor,
        dropout: nn.Dropout = None,
    ):
        """Computes scaled dot-product attention scores and weights.

        Implements the attention mechanism: Attention(Q,K,V) = softmax(QK^T/sqrt(d_k))
        with optional masking and dropout.

        Args:
            queries (torch.Tensor): Query tensor (batch, num_heads, num_tokens, head_dim)
            keys (torch.Tensor): Key tensor (batch, num_heads, num_tokens, head_dim)
            mask (torch.Tensor): Boolean attention mask
            dropout (nn.Dropout, optional): Dropout module for attention weights

        Returns:
            torch.Tensor: Attention weights tensor (batch, num_heads, num_tokens, num_tokens)
        """
        # Compute scaled dot-product attention scores
        # QK^T/sqrt(d_k) for stable gradients
        attention_scores = (queries @ keys.transpose(2, 3)) / np.sqrt(queries.shape[-1])

        # Apply causal mask by setting masked positions to -inf before softmax
        if mask is not None:
            mask = mask.unsqueeze(0)  # Add batch dimension
            attention_scores.masked_fill_(mask == 0, -torch.inf)

        # Normalize scores to probabilities with softmax
        attention_weights = torch.softmax(attention_scores, dim=-1)

        # Apply dropout if provided
        if dropout is not None:
            attention_weights = dropout(attention_weights)
        return attention_weights

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass computing multi-head attention.

        The forward pass:
        1. Projects inputs to queries, keys and values
        2. Splits projections into multiple heads
        3. Computes scaled dot-product attention per head
        4. Combines heads and projects to output dimension

        Args:
            query (torch.Tensor): Query input (batch_size, num_tokens, d_in)
            key (torch.Tensor): Key input (batch_size, num_tokens, d_in)
            value (torch.Tensor): Value input (batch_size, num_tokens, d_in)

        Returns:
            torch.Tensor: Attention output (batch_size, num_tokens, d_out)
        """
        num_tokens = query.shape[1]
        mask = self.causal_mask(num_tokens, device=query.device)

        # Project inputs to Q,K,V representations
        keys = self.W_key(key)  # (batch, num_tokens, d_out)
        queries = self.W_query(query)  # (batch, num_tokens, d_out)
        values = self.W_value(value)  # (batch, num_tokens, d_out)

        # Reshape to split into multiple heads
        # (batch, num_tokens, d_out) -> (batch, num_tokens, num_heads, head_dim)
        queries = queries.view(
            query.shape[0], query.shape[1], self.num_heads, self.head_dim
        )
        keys = keys.view(key.shape[0], key.shape[1], self.num_heads, self.head_dim)
        values = values.view(
            value.shape[0], value.shape[1], self.num_heads, self.head_dim
        )

        # Transpose for attention computation
        # (batch, num_tokens, num_heads, head_dim) -> (batch, num_heads, num_tokens, head_dim)
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        # Compute attention weights and apply to values
        attention_weights = MultiHeadAttention.calculate_attention_scores(
            queries, keys, mask, self.dropout
        )

        # Apply attention weights to values
        # (batch, num_heads, num_tokens, num_tokens) @ (batch, num_heads, num_tokens, head_dim)
        # -> (batch, num_heads, num_tokens, head_dim)
        context_vecs = attention_weights @ values

        # Restore original shape ordering
        # (batch, num_heads, num_tokens, head_dim) -> (batch, num_heads, num_tokens, head_dim)
        context_vecs = context_vecs.transpose(1, 2)

        # Combine all heads
        # (batch, num_heads, num_tokens, head_dim) -> (batch, num_heads, num_tokens, d_out)
        context_vecs = context_vecs.contiguous().view(
            context_vecs.shape[0], -1, self.d_out
        )

        # Final output projection
        return self.out_proj(context_vecs)
