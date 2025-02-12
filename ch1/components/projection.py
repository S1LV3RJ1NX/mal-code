from components.common import nn, torch


class ProjectionLayer(nn.Module):
    """Projection layer that maps model dimension to vocabulary size.

    This layer is typically used as the final layer in a transformer model to convert
    the model's hidden representations into logits over the vocabulary. These logits
    can then be used to predict the next token in the sequence.

    The projection is performed using a simple linear transformation without any
    non-linear activation function, as is standard in transformer architectures.

    Args:
        d_model (int): Dimension of the input hidden representations
        vocab_size (int): Size of the target vocabulary
    """

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        # Linear transformation to project from model dimension to vocabulary size
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project hidden representations to vocabulary logits.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model)
                containing the model's hidden representations

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, vocab_size)
                containing unnormalized probabilities (logits) over the vocabulary
        """
        # (batch_size, seq_len, d_model) --> (batch_size, seq_len, vocab_size)
        projection = self.linear(x)
        return projection
