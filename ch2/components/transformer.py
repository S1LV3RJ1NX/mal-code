from components.common import nn, torch
from components.feed_forward import FeedForward
from components.multi_head_attention import MultiHeadAttention


class TransformerBlock(nn.Module):
    """A single transformer block/layer that combines multi-head self-attention and feed-forward networks.

    The transformer block implements the core transformer architecture from "Attention is All You Need"
    with two main sub-layers:
    1. Multi-head self-attention with layer normalization and residual connection
    2. Position-wise feed-forward network with layer normalization and residual connection

    Each sub-layer employs a pre-norm architecture where layer normalization is applied before
    the main computation.

    Args:
        config (dict): Configuration dictionary containing:
            - emb_dim (int): Embedding dimension size
            - drop_rate (float): Dropout probability
            - n_heads (int): Number of attention heads
            - qkv_bias (bool): Whether to use bias in attention projections
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config = config

        # First layer norm before multi-head attention
        self.layer_norm1 = nn.LayerNorm(config["emb_dim"])

        # Multi-head self-attention layer
        self.mha = MultiHeadAttention(
            d_in=config["emb_dim"],
            d_out=config["emb_dim"],
            dropout=config["drop_rate"],
            num_heads=config["n_heads"],
            qkv_bias=config["qkv_bias"],
        )

        # Second layer norm before feed-forward network
        self.layer_norm2 = nn.LayerNorm(config["emb_dim"])

        # Position-wise feed-forward network
        self.ffn = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the transformer block.

        Applies the following sequence of operations:
        1. Layer norm -> Multi-head attention -> Residual connection
        2. Layer norm -> Feed-forward network -> Residual connection

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_tokens, emb_dim)

        Returns:
            torch.Tensor: Transformed tensor of shape (batch_size, num_tokens, emb_dim)
        """
        # First sub-layer: Multi-head attention with residual connection
        # Pre-norm architecture: layer norm applied before attention
        norm_x = self.layer_norm1(x)
        x = x + self.mha(
            norm_x, norm_x, norm_x
        )  # Self-attention: query, key, value are the same

        # Second sub-layer: Feed-forward network with residual connection
        # Pre-norm architecture: layer norm applied before FFN
        x = x + self.ffn(self.layer_norm2(x))
        return x
