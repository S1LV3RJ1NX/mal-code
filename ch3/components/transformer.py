from components.common import nn, torch
from components.feedforward import FeedForward
from components.grouped_query_attention import GroupedQueryAttention


class TransformerBlock(nn.Module):
    """
    TransformerBlock implements a single block of the Transformer architecture.
    It consists of a grouped query attention mechanism followed by a feed-forward network,
    with layer normalization applied before each sub-layer and residual connections.

    Args:
        cfg (dict): Configuration dictionary containing the following keys:
            - emb_dim (int): The dimensionality of the embedding space.
            - context_length (int): The maximum length of the input sequences.
            - n_heads (int): The number of attention heads.
            - n_kv_groups (int): The number of key-value groups for grouped query attention.
            - rope_base (float): The base for rotary positional encoding.
            - rope_freq (float): The frequency configuration for rotary positional encoding.
            - dtype (torch.dtype): The data type for the model parameters (e.g., torch.float32, torch.bfloat16).
    """

    def __init__(self, cfg: dict):
        super().__init__()
        # Initialize the grouped query attention layer
        self.att = GroupedQueryAttention(
            d_in=cfg["emb_dim"],  # Input dimension
            d_out=cfg["emb_dim"],  # Output dimension (same as input for this block)
            context_length=cfg["context_length"],  # Maximum context length
            num_heads=cfg["n_heads"],  # Number of attention heads
            num_kv_groups=cfg["n_kv_groups"],  # Number of key-value groups
            rope_base=cfg["rope_base"],  # Base for rotary positional encoding
            rope_config=cfg[
                "rope_freq"
            ],  # Configuration for rotary positional encoding
            dtype=cfg["dtype"],  # Data type for model parameters
        )
        # Initialize the feed-forward network
        self.ff = FeedForward(cfg)
        # Initialize layer normalization for the attention output
        self.norm1 = nn.RMSNorm(cfg["emb_dim"], eps=1e-5)
        # Initialize layer normalization for the feed-forward output
        self.norm2 = nn.RMSNorm(cfg["emb_dim"], eps=1e-5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the TransformerBlock.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, num_tokens, emb_size].

        Returns:
            torch.Tensor: Output tensor of the same shape as input after applying
                          attention, feed-forward, and residual connections.
        """
        # Shortcut connection for the attention block
        shortcut = x
        x = self.norm1(x)  # Apply layer normalization before attention
        x = self.att(x.to(torch.bfloat16))  # Apply grouped query attention
        x = x + shortcut  # Add the original input back (residual connection)

        # Shortcut connection for the feed-forward block
        shortcut = x
        x = self.norm2(x)  # Apply layer normalization before feed-forward
        x = self.ff(x.to(torch.bfloat16))  # Apply feed-forward network
        x = x + shortcut  # Add the original input back (residual connection)

        return x  # Return the output tensor
