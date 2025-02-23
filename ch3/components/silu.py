from components.common import torch, nn

class SiLU(nn.Module):
    """
    SiLU (Sigmoid Linear Unit) is an activation function defined as 
    f(x) = x * sigmoid(x). It is a smooth, non-monotonic function 
    that has been shown to perform well in various deep learning 
    applications, particularly in neural networks.

    This implementation inherits from PyTorch's nn.Module, allowing 
    it to be used as a layer in a neural network model.

    Attributes:
        None
    """

    def __init__(self):
        """
        Initializes the SiLU activation layer. This constructor does not 
        require any parameters, as the SiLU function is defined purely 
        in terms of its input.

        Args:
            None
        """
        super(SiLU, self).__init__()

    def forward(self, x):
        """
        Forward pass for the SiLU activation function.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, ..., input_dim).
                              This tensor can have any number of dimensions.

        Returns:
            torch.Tensor: The output tensor after applying the SiLU activation function.
                          The output will have the same shape as the input tensor.
        """
        return x * torch.sigmoid(x)

if __name__ == "__main__":
    # Set the random seed for reproducibility
    torch.manual_seed(123)

    # Create an example input tensor with random values
    example_batch = torch.randn(2, 3, 4)

    # Instantiate the custom SiLU activation layer
    silu = SiLU()
    
    # Instantiate the built-in PyTorch SiLU for comparison
    silu_pytorch = torch.nn.SiLU()

    # Check if the outputs of the custom and built-in SiLU are equivalent
    print(f"Are the two implementations equivalent? {torch.allclose(silu(example_batch), silu_pytorch(example_batch))}")