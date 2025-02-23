from components.common import torch, nn

class RMSNorm(nn.Module):
    """
    RMSNorm is a layer normalization technique that normalizes the input tensor
    using the Root Mean Square (RMS) of the input values. This is particularly useful
    in deep learning models to stabilize training and improve convergence.

    Attributes:
        emb_dim (int): The dimensionality of the input embeddings.
        eps (float): A small constant added to the denominator for numerical stability.
        weight (nn.Parameter): A learnable parameter that scales the normalized output.
    """

    def __init__(self, emb_dim, eps=1e-5):
        """
        Initializes the RMSNorm layer.

        Args:
            emb_dim (int): The dimensionality of the input embeddings.
            eps (float, optional): A small constant for numerical stability. Default is 1e-5.
        """
        super().__init__()
        self.eps = eps
        self.emb_dim = emb_dim
        self.weight = nn.Parameter(torch.ones(emb_dim)).float()

    def forward(self, x):
        """
        Forward pass for the RMSNorm layer.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, ..., emb_dim).

        Returns:
            torch.Tensor: The normalized output tensor, scaled by the learnable weight.
        """
        # Calculate the mean of the squares of the input tensor along the last dimension
        means = x.pow(2).mean(dim=-1, keepdim=True)
        
        # Normalize the input tensor using the RMS value
        x_normed = x * torch.rsqrt(means + self.eps)
        
        # Scale the normalized tensor by the learnable weight and return
        return (x_normed * self.weight).to(dtype=x.dtype)


if __name__ == "__main__":
    # Set the random seed for reproducibility
    torch.manual_seed(123)

    # Create an example input tensor with random values
    example_batch = torch.randn(2, 3, 4)

    # Instantiate the RMSNorm layer with the appropriate embedding dimension
    rms_norm = RMSNorm(emb_dim=example_batch.shape[-1])
    
    # Instantiate the built-in PyTorch RMSNorm for comparison
    rmsnorm_pytorch = torch.nn.RMSNorm(example_batch.shape[-1], eps=1e-5)

    # Assert that the outputs of the custom and built-in RMSNorm are close
    print(f"Both are implemented correctly within tolerance: {torch.allclose(rms_norm(example_batch), rmsnorm_pytorch(example_batch), atol=1e-5)}")    