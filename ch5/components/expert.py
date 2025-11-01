"""
Expert Module for Mixture of Experts.

This module implements individual expert networks, which are simple feed-forward
neural networks that specialize in different aspects of the input data.
"""

from components.common import nn, torch


class Expert(nn.Module):
    """
    An individual expert in the Mixture of Experts architecture.
    
    Each expert is a simple feed-forward neural network with:
    - A linear layer that expands the input dimension
    - A ReLU activation function
    - A linear layer that projects back to the original dimension
    - Optional dropout for regularization
    
    The simplicity is intentional: experts specialize through the data they see
    during training and the routing decisions, not through architectural complexity.
    
    Args:
        n_embd (int): The input/output embedding dimension
        expert_hidden_dim (int): The hidden dimension of the expert network
        dropout (float): Dropout rate for regularization (default: 0.1)
    
    Example:
        >>> expert = Expert(n_embd=512, expert_hidden_dim=2048, dropout=0.1)
        >>> x = torch.randn(2, 10, 512)  # (batch_size, seq_len, n_embd)
        >>> output = expert(x)
        >>> print(output.shape)  # torch.Size([2, 10, 512])
    """
    
    def __init__(self, n_embd: int, expert_hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, expert_hidden_dim),  # Expand to hidden dimension
            nn.ReLU(),                              # Non-linearity
            nn.Linear(expert_hidden_dim, n_embd),  # Project back to original dimension
            nn.Dropout(dropout),                    # Regularization
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the expert network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, n_embd)
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, n_embd)
        """
        return self.net(x)