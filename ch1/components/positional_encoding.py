from components.common import nn, np, torch


class PositionalEncoding(nn.Module):
    """Positional encoding layer that adds positional information to word embeddings.

    This layer generates a matrix of positional encodings that are added to the word
    embeddings to provide information about the position of each word in the sequence.
    The encodings are a combination of sine and cosine waves of different frequencies.

    Args:
        d_model (int): Dimension of the word embeddings
        seq_len (int): Maximum length of a sentence
        dropout (float, optional): Dropout rate for the positional encodings
    """

    def __init__(self, d_model: int, seq_len: int, dropout: float = 0.1):
        super().__init__()

        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # Create a matrix of shape (seq_len, d_model) to hold the positional encodings
        # We need vectors of d_model dimension and we need a total of seq_len such vectors
        pe = torch.zeros(seq_len, d_model)
        # Create a sequence of positions from 0 to seq_len
        # We need to unsqueeze it to make it a column vector
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(
            1
        )  # shape: (seq_len, 1)

        # Create a sequence of frequencies for the sine and cosine waves
        # We use a geometric progression from 1 to 10000
        # Value calculated in log space for numerical stability
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )

        # Apply sine to even indices and cosine to odd indices
        # We use a slice to apply the sine function to every second element
        # The first slice `:` means we apply it to all rows
        # The second slice `0::2` means we apply it to every second column (even indices)
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cosine to odd indices
        # The second slice `1::2` means we apply it to every second column (odd indices)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add a batch dimension
        pe = pe.unsqueeze(0)  # shape: (1, seq_len, d_model)
        # Register the positional encodings as a buffer
        # Buffers are not parameters and are not updated during training
        # They are used for storing constants or precomputed values
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor):
        # Add the positional encodings to the word embeddings
        # We use a slice to get the encodings for the current sequence length
        # pe is expanded to match batch size
        # x shape: (batch_size, seq_len, d_model)
        # pe shape: (1, seq_len, d_model)
        x = x + (self.pe[:, : x.shape[1], :]).requires_grad_(
            False
        )  # shape: (batch_size, seq_len, d_model)
        # Apply dropout to the positional encodings
        return self.dropout(x)
