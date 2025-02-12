from components.common import nn, torch


class FeedForward(nn.Module):
    """Feed-forward neural network that applies a linear transformation followed by a non-linear activation function.

    This implementation allows for more flexibility because it learns separate scaling and shifting parameters for each feature.
    This can lead to better performance in scenarios where different features require different levels of normalization.

    Args:
        d_model (int): Dimension of the input tensor
        d_ff (int): Dimension of the intermediate tensor
        dropout (float, optional): Dropout rate for the feed-forward network
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (batch_size, seq_len, d_model) --> (batch_size, seq_len, d_ff) --> (batch_size, seq_len, d_model)
        return self.model(x)
