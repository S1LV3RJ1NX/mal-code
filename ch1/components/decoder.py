from components.common import nn, torch
from components.feed_forward import FeedForward
from components.multi_head_attention import MultiHeadAttention


class DecoderBlock(nn.Module):
    """Decoder block that consists of self-attention, cross-attention and feed-forward layers.

    This block is used in the decoder of the transformer model. It processes target sequences through
    three sub-layers:
    1. Masked self-attention that prevents positions from attending to subsequent positions
    2. Cross-attention that allows the decoder to attend to all positions in the encoder output
    3. Feed-forward network for further processing

    The block uses a Pre-LN (Layer Normalization) architecture where normalization is applied before
    each sub-layer. This approach provides more stable training compared to Post-LN.

    Args:
        d_model (int): Dimension of the input and output tensors (model dimension)
        d_ff (int): Dimension of the intermediate feed-forward layer
        num_heads (int): Number of attention heads for parallel attention computation
        dropout (float, optional): Dropout rate for regularization. Defaults to 0.1
    """

    def __init__(self, d_model: int, d_ff: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        # Multi-head self-attention layer for target sequence
        self.self_attention = MultiHeadAttention(d_model, d_model, dropout, num_heads)
        # Multi-head cross-attention layer to attend to encoder outputs
        self.cross_attention = MultiHeadAttention(d_model, d_model, dropout, num_heads)
        # Position-wise feed-forward network
        self.feed_forward = FeedForward(d_model, d_ff, dropout)

        # Layer normalization layers - one before each sub-layer
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        """Process input through the decoder block.

        Args:
            x (torch.Tensor): Target sequence tensor of shape (batch_size, tgt_seq_len, d_model)
            encoder_output (torch.Tensor): Encoder output tensor of shape (batch_size, src_seq_len, d_model)
            src_mask (torch.Tensor, optional): Mask for encoder outputs. Defaults to None
            tgt_mask (torch.Tensor, optional): Causal mask for target sequence. Defaults to None

        Returns:
            torch.Tensor: Processed tensor of shape (batch_size, tgt_seq_len, d_model)
        """
        # 1. Self attention with residual connection
        # First normalize, then apply masked self-attention, then dropout, then add residual
        attended = x + self.dropout(
            self.self_attention(self.norm1(x), self.norm1(x), self.norm1(x), tgt_mask)
        )

        # 2. Cross attention with residual connection
        # First normalize, then apply cross-attention to encoder output, then dropout, then add residual
        cross_attended = attended + self.dropout(
            self.cross_attention(
                self.norm2(attended), encoder_output, encoder_output, src_mask
            )
        )

        # 3. Feed-forward with residual connection
        # First normalize, then apply feed-forward, then dropout, then add residual
        return cross_attended + self.dropout(
            self.feed_forward(self.norm3(cross_attended))
        )


class Decoder(nn.Module):
    """Decoder that consists of a stack of decoder blocks.

    This decoder implements the core decoding component of the transformer architecture.
    It processes target sequences through multiple identical decoder blocks in sequence,
    where each block refines the representations through self-attention, cross-attention
    with encoder outputs, and feed-forward processing.

    Args:
        d_model (int, optional): Dimension of the model's internal representations. Defaults to 512
        d_ff (int, optional): Dimension of feed-forward layer. Defaults to 2048
        num_heads (int, optional): Number of attention heads in each block. Defaults to 8
        num_layers (int, optional): Number of decoder blocks to stack. Defaults to 6
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
        # Create a ModuleList of identical decoder blocks
        self.layers = nn.ModuleList(
            [DecoderBlock(d_model, d_ff, num_heads, dropout) for _ in range(num_layers)]
        )

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        src_mask: torch.Tensor = None,
        tgt_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """Process input through the entire decoder stack.

        Args:
            x (torch.Tensor): Target sequence tensor of shape (batch_size, tgt_seq_len, d_model)
            encoder_output (torch.Tensor): Encoder output tensor of shape (batch_size, src_seq_len, d_model)
            src_mask (torch.Tensor, optional): Mask for encoder outputs. Defaults to None
            tgt_mask (torch.Tensor, optional): Causal mask for target sequence. Defaults to None

        Returns:
            torch.Tensor: Decoded output tensor of shape (batch_size, tgt_seq_len, d_model)
        """
        # Pass input through each decoder block in sequence
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return x
