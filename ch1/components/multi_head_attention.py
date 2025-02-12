from components.common import nn, np, torch


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism that allows the model to attend to different parts of the input sequence.

    This mechanism uses multiple attention heads to capture different aspects of the input data.
    Each head has its own query, key, and value matrices, which are learned during training.

    Args:
        d_in (int): Dimension of the input data
        d_out (int): Dimension of the output data
        dropout (float): Dropout rate for the attention mechanism
        num_heads (int): Number of attention heads
        qkv_bias (bool, optional): Whether to use bias in the query, key, and value matrices
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
        self.head_dim = d_out // num_heads  # This is the dimension of each head

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)  # Wq
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)  # Wk
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)  # Wv

        # Uses a Linear layer to combine head outputs
        self.out_proj = nn.Linear(d_out, d_out, bias=qkv_bias)  # Wo

        # Dropout layer to prevent overfitting
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def calculate_attention_scores(queries, keys, mask, dropout: nn.Dropout = None):
        # We now perform the attention calculation for each head
        # Shape: (b, num_heads, seq_len, head_dim) @ (b, num_heads, head_dim, seq_len) -> (b, num_heads, seq_len, seq_len)
        attention_scores = (queries @ keys.transpose(2, 3)) / np.sqrt(queries.shape[-1])

        # Apply the causal mask
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -torch.inf)

        # Compute the attention weights
        attention_weights = torch.softmax(attention_scores, dim=-1)
        if dropout is not None:
            attention_weights = dropout(attention_weights)
        return attention_weights

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        # 0 -> batch_size, 1 -> seq_len, 2 -> d_in

        # Generate keys, queries and values matrix for the input query, key, value via matrix multiplication
        # Shape: (batch_size, seq_len, d_in) -> (batch_size, seq_len, d_out)
        keys = self.W_key(key)
        queries = self.W_query(query)
        values = self.W_value(value)

        # We implicitly split the matrix by adding a num_heads dimension.
        # Then we unroll the last dim: (b, seq_len, d_out) -> (b, seq_len, num_heads, head_dim)
        # This was possible because d_out is divisible by num_heads.
        queries = queries.view(
            query.shape[0], query.shape[1], self.num_heads, self.head_dim
        )
        keys = keys.view(key.shape[0], key.shape[1], self.num_heads, self.head_dim)
        values = values.view(
            value.shape[0], value.shape[1], self.num_heads, self.head_dim
        )

        # We now move the num_heads dimension to the first position
        # This allows us to perform the attention calculation for each head independently
        # Shape: (b, seq_len, num_heads, head_dim) -> (b, num_heads, seq_len, head_dim)
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        # Compute the attention weights
        attention_weights = MultiHeadAttention.calculate_attention_scores(
            queries, keys, mask, self.dropout
        )

        # Compute the context vector
        # Shape: (b, num_heads, seq_len, seq_len) @ (b, num_heads, seq_len, head_dim) -> (b, num_heads, seq_len, head_dim)
        context_vecs = attention_weights @ values

        # We now move the num_heads dimension as it was before
        # Shape: (b, num_heads, seq_len, head_dim) -> (b, seq_len, num_heads, head_dim)
        context_vecs = context_vecs.transpose(1, 2)

        # We now combine the heads, where num_heads * head_dim = d_out
        # Shape: (b, seq_len, num_heads, head_dim) -> (b, seq_len, d_out)
        context_vecs = context_vecs.contiguous().view(
            context_vecs.shape[0], -1, self.d_out
        )
        # Adds an optional linear projection
        return self.out_proj(context_vecs)
