from components.common import nn, torch
from components.transformer import TransformerBlock


class Llama3(nn.Module):
    """
    Llama3Model implements the Llama 3.2-3B transformer model architecture.

    This model consists of an embedding layer, multiple transformer blocks,
    a final normalization layer, and an output head for generating logits.

    Args:
        cfg (dict): Configuration dictionary containing the following keys:
            - vocab_size (int): The size of the vocabulary.
            - emb_dim (int): The dimensionality of the embedding space.
            - n_layers (int): The number of transformer blocks in the model.
            - dtype (torch.dtype): The data type for the model parameters (e.g., torch.float32, torch.bfloat16).
    """

    def __init__(self, cfg: dict):
        super().__init__()
        # Token embedding layer that maps input indices to dense vectors
        self.embedding = nn.Embedding(
            cfg["vocab_size"], cfg["emb_dim"], dtype=cfg["dtype"]
        )

        # Stacking multiple transformer blocks to form the core of the model
        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        # Final layer normalization to stabilize the output of the transformer blocks
        self.final_norm = nn.RMSNorm(cfg["emb_dim"], eps=1e-5)

        # Output head that projects the final hidden states to the vocabulary size for logits
        self.output_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False, dtype=cfg["dtype"]
        )

    def forward(self, in_idx: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Llama3Model.

        Args:
            in_idx (torch.Tensor): Input tensor of shape [batch_size, sequence_length] containing token indices.

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, sequence_length, vocab_size] containing logits for each token.
        """
        # Obtain token embeddings for the input indices
        embeddings = self.embedding(in_idx)

        # Pass the embeddings through the transformer blocks
        x = embeddings
        x = self.transformer_blocks(x)

        # Apply final layer normalization
        x = self.final_norm(x)

        # Generate logits by passing through the output head
        logits = self.output_head(
            x.to(torch.bfloat16 if torch.cuda.is_available() else torch.float32)
        )

        # Return the logits for the next token predictions
        return logits
