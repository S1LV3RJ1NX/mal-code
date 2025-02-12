from components.common import nn, torch
from components.decoder import Decoder
from components.encoder import Encoder
from components.input_embeddings import InputEmbeddings
from components.positional_encoding import PositionalEncoding
from components.projection import ProjectionLayer


class Transformer(nn.Module):
    """
    The complete Transformer model that combines encoder, decoder and projection layers.

    This implements the full Transformer architecture from "Attention is All You Need" (Vaswani et al., 2017).
    It processes input sequences through an encoder-decoder architecture with self-attention mechanisms.

    The model flow is:
    1. Source text → Embeddings → Positional Encoding → Encoder
    2. Target text → Embeddings → Positional Encoding → Decoder (using encoder output)
    3. Decoder output → Projection Layer → Final output probabilities
    """

    def __init__(
        self,
        src_embeddings: InputEmbeddings,
        src_positional_encoding: PositionalEncoding,
        tgt_embeddings: InputEmbeddings,
        tgt_positional_encoding: PositionalEncoding,
        encoder: Encoder,
        decoder: Decoder,
        projection_layer: ProjectionLayer,
    ):
        """
        Initialize the Transformer with all its components.

        Args:
            src_embeddings: Embedding layer for source language
            src_positional_encoding: Positional encoding for source sequences
            tgt_embeddings: Embedding layer for target language
            tgt_positional_encoding: Positional encoding for target sequences
            encoder: The encoder component
            decoder: The decoder component
            projection_layer: Final projection layer to vocabulary size
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embeddings
        self.tgt_embed = tgt_embeddings
        self.src_positional_encoding = src_positional_encoding
        self.tgt_positional_encoding = tgt_positional_encoding
        self.projection_layer = projection_layer

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Encode the source sequence.

        Process flow:
        1. Convert source tokens to embeddings
        2. Add positional encoding
        3. Pass through encoder

        Args:
            src: Source sequence tensor of token ids
            src_mask: Mask to prevent attention to padded tokens

        Returns:
            Encoded representation of the source sequence
        """
        # Convert to embeddings and ensure int64 dtype
        src = self.src_embed(src.to(dtype=torch.int64))
        # Add positional encoding
        src = self.src_positional_encoding(src)
        # Pass through encoder
        return self.encoder(src, src_mask)

    def decode(
        self,
        tgt: torch.Tensor,
        encoder_output: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Decode the target sequence using encoder output.

        Process flow:
        1. Convert target tokens to embeddings
        2. Add positional encoding
        3. Pass through decoder along with encoder output

        Args:
            tgt: Target sequence tensor of token ids
            encoder_output: Output from the encoder
            src_mask: Mask for source sequence
            tgt_mask: Causal mask for target sequence

        Returns:
            Decoded representation before final projection
        """
        # Convert to embeddings and ensure int64 dtype
        tgt = self.tgt_embed(tgt.to(dtype=torch.int64))
        # Add positional encoding
        tgt = self.tgt_positional_encoding(tgt)
        # Pass through decoder
        return self.decoder(
            x=tgt, encoder_output=encoder_output, src_mask=src_mask, tgt_mask=tgt_mask
        )

    def project(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project decoder output to vocabulary size.

        Args:
            x: Decoder output tensor

        Returns:
            Logits over target vocabulary
        """
        return self.projection_layer(x)


def build_transformer(
    src_vocab_size: int,
    tgt_vocab_size: int,
    src_seq_len: int,
    tgt_seq_len: int,
    d_model: int = 512,
    d_ff: int = 2048,
    num_heads: int = 8,
    num_layers: int = 6,
    dropout: float = 0.1,
) -> Transformer:
    """
    Builds and initializes a complete Transformer model for sequence-to-sequence tasks.

    This function creates all the components of a Transformer architecture including:
    - Source and target embeddings
    - Positional encodings for both sequences
    - Multi-layer encoder and decoder
    - Final projection layer

    Args:
        src_vocab_size (int): Size of source language vocabulary
        tgt_vocab_size (int): Size of target language vocabulary
        src_seq_len (int): Maximum length of source sequences
        tgt_seq_len (int): Maximum length of target sequences
        d_model (int, optional): Dimension of model embeddings. Defaults to 512.
        d_ff (int, optional): Dimension of feed-forward networks. Defaults to 2048.
        num_heads (int, optional): Number of attention heads. Defaults to 8.
        num_layers (int, optional): Number of encoder/decoder layers. Defaults to 6.
        dropout (float, optional): Dropout probability. Defaults to 0.1.

    Returns:
        Transformer: Initialized transformer model ready for training
    """
    # Create embedding layers that convert tokens to vectors
    # These capture semantic relationships between words
    src_embeddings = InputEmbeddings(vocab_size=src_vocab_size, d_model=d_model)
    tgt_embeddings = InputEmbeddings(vocab_size=tgt_vocab_size, d_model=d_model)

    # Create positional encoding layers that add position information
    # This helps the model understand word order in sequences
    src_positional_encoding = PositionalEncoding(
        d_model=d_model, seq_len=src_seq_len, dropout=dropout
    )
    tgt_positional_encoding = PositionalEncoding(
        d_model=d_model, seq_len=tgt_seq_len, dropout=dropout
    )

    # Create projection layer that converts decoder output to vocabulary probabilities
    projection_layer = ProjectionLayer(d_model=d_model, vocab_size=tgt_vocab_size)

    # Create the encoder stack that processes the input sequence
    encoder = Encoder(
        d_model=d_model,
        d_ff=d_ff,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
    )

    # Create the decoder stack that generates the output sequence
    decoder = Decoder(
        d_model=d_model,
        d_ff=d_ff,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
    )

    # Assemble all components into the full transformer
    transformer = Transformer(
        src_embeddings=src_embeddings,
        src_positional_encoding=src_positional_encoding,
        tgt_embeddings=tgt_embeddings,
        tgt_positional_encoding=tgt_positional_encoding,
        encoder=encoder,
        decoder=decoder,
        projection_layer=projection_layer,
    )

    # Initialize all parameters using Xavier uniform initialization
    # This helps achieve stable training by keeping variance constant
    # across layers
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer
