from components.common import nn

class FeedForward(nn.Module):
    """
    FeedForward is a neural network module that implements a feedforward layer 
    with two linear transformations followed by a non-linear activation function. 
    This module is commonly used in transformer architectures to process 
    embeddings and enhance their representational power.

    Attributes:
        fc1 (nn.Linear): The first linear transformation layer that maps 
                         input embeddings to a hidden dimension.
        fc2 (nn.Linear): The second linear transformation layer that also maps 
                         input embeddings to a hidden dimension.
        fc3 (nn.Linear): The third linear transformation layer that maps 
                         the output from the hidden dimension back to the 
                         original embedding dimension.
    """

    def __init__(self, cfg):
        """
        Initializes the FeedForward layer.

        Args:
            cfg (dict): A configuration dictionary containing the following keys:
                - emb_dim (int): The dimensionality of the input embeddings.
                - hidden_dim (int): The dimensionality of the hidden layer.
                - dtype (torch.dtype): The data type for the parameters (e.g., torch.float32).
        
        The constructor initializes three linear layers:
        - fc1 and fc2 transform the input embeddings to the hidden dimension.
        - fc3 transforms the hidden representation back to the embedding dimension.
        """
        super().__init__()
        self.fc1 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False)
        self.fc2 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False)
        self.fc3 = nn.Linear(cfg["hidden_dim"], cfg["emb_dim"], dtype=cfg["dtype"], bias=False)

    def forward(self, x):
        """
        Forward pass for the FeedForward layer.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, ..., emb_dim).
                              This tensor represents the input embeddings to be processed.

        Returns:
            torch.Tensor: The output tensor after applying the feedforward operations.
                          The output will have the same shape as the input tensor.

        The forward method performs the following steps:
        1. Applies the first linear transformation (fc1) to the input tensor.
        2. Applies the second linear transformation (fc2) to the input tensor.
        3. Applies the SiLU activation function to the output of the first transformation.
        4. Multiplies the activated output with the output of the second transformation.
        5. Applies the final linear transformation (fc3) to produce the output.
        """
        x_fc1 = self.fc1(x)  # First linear transformation
        x_fc2 = self.fc2(x)  # Second linear transformation
        x = nn.functional.silu(x_fc1) * x_fc2  # Element-wise multiplication with activation
        return self.fc3(x)  # Final linear transformation to output