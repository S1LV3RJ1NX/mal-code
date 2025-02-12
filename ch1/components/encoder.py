from components.common import nn, torch
from components.feed_forward import FeedForward
from components.multi_head_attention import MultiHeadAttention


class EncoderBlock(nn.Module):
    """Encoder block that consists of a multi-head attention mechanism and a feed-forward network.

    This block is used in the encoder of the transformer model. It processes input sequences through
    self-attention and feed-forward layers with residual connections and layer normalization.

    The block uses a Pre-LN (Layer Normalization) architecture where normalization is applied before
    the attention and feed-forward components. This approach provides more stable training compared
    to the Post-LN architecture described in the original transformer paper.

    Args:
        d_model (int): Dimension of the input and output tensors (model dimension)
        d_ff (int): Dimension of the intermediate feed-forward layer
        num_heads (int): Number of attention heads for parallel attention computation
        dropout (float, optional): Dropout rate for regularization. Defaults to 0.1
    """

    def __init__(self, d_model: int, d_ff: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        # Multi-head self-attention layer
        self.attention = MultiHeadAttention(d_model, d_model, dropout, num_heads)
        # Position-wise feed-forward network
        self.feed_forward = FeedForward(d_model, d_ff, dropout)

        # Layer normalization layers - one before attention and one before feed-forward
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """Process input through the encoder block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model)
            mask (torch.Tensor, optional): Attention mask tensor. Defaults to None

        Returns:
            torch.Tensor: Processed tensor of shape (batch_size, seq_len, d_model)
        """
        # 1. Self-attention with residual connection
        # First normalize, then apply attention, then dropout, then add residual
        attended = x + self.dropout(
            self.attention(self.norm1(x), self.norm1(x), self.norm1(x), mask)
        )

        # 2. Feed-forward with residual connection
        # First normalize, then apply feed-forward, then dropout, then add residual
        return x + self.dropout(self.feed_forward(self.norm2(attended)))


class Encoder(nn.Module):
    """Encoder that consists of a stack of encoder blocks.

    This encoder implements the core encoding component of the transformer architecture.
    It processes input sequences through multiple identical encoder blocks in sequence,
    where each block refines the representations through self-attention and feed-forward
    processing.

    Args:
        d_model (int, optional): Dimension of the model's internal representations. Defaults to 512
        d_ff (int, optional): Dimension of feed-forward layer. Defaults to 2048
        num_heads (int, optional): Number of attention heads in each block. Defaults to 8
        num_layers (int, optional): Number of encoder blocks to stack. Defaults to 6
        dropout (float, optional): Dropout rate for regularization. Defaults to 0.1
    """

    def __init__(
        self,
        d_model: int = 512,
        d_ff: int = 2048,
        num_heads: int = 8,
        num_layers: int = 6,
        dropout: float = 0.1,
    ):
        super().__init__()
        # Create a ModuleList of identical encoder blocks
        self.layers = nn.ModuleList(
            [
                EncoderBlock(
                    d_model=d_model, d_ff=d_ff, num_heads=num_heads, dropout=dropout
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """Process input through the entire encoder stack.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model)
            mask (torch.Tensor, optional): Attention mask tensor. Defaults to None

        Returns:
            torch.Tensor: Encoded output tensor of shape (batch_size, seq_len, d_model)
        """
        # Pass input through each encoder block in sequence
        for layer in self.layers:
            x = layer(x, mask)
        return x
