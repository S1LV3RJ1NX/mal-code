from components.common import nn, np, torch


class InputEmbeddings(nn.Module):
    """Input embeddings layer that converts token IDs to continuous vector representations.

    This layer performs token embedding lookup and scales the embeddings by sqrt(d_model)
    as described in the "Attention is All You Need" paper. The scaling helps maintain
    variance of the embeddings after initialization.

    Args:
        vocab_size (int): Size of the vocabulary
        d_model (int): Dimension of the embedding vectors
    """

    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        # Dimension of the embedding vectors
        self.d_model = d_model
        # Size of vocabulary
        self.vocab_size = vocab_size
        # Create embedding lookup table
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Convert input token IDs to embeddings and apply scaling.

        Args:
            x (torch.Tensor): Input tensor of token IDs [batch_size, seq_len]

        Returns:
            torch.Tensor: Scaled embedding vectors [batch_size, seq_len, d_model]
        """
        # Look up embeddings and scale by sqrt(d_model)
        return self.embedding(x) * np.sqrt(self.d_model)
