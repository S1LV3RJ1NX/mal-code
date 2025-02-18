from components.common import torch, nn

class FeedForward(nn.Module):
    """Feed-forward network with two linear layers and a GeLU activation function.
    
    This implements a position-wise feed-forward network as described in the GPT-2 paper.
    The network consists of:
    1. A linear expansion layer that projects to 4x the embedding dimension 
    2. A GeLU activation function using tanh approximation
    3. A linear projection back to the original embedding dimension

    The expansion factor of 4 and GeLU activation are specific architectural choices made in GPT-2,
    differing from the ReLU activation used in the original Transformer.

    Args:
        config (dict): Model configuration containing:
            - emb_dim (int): Embedding dimension that determines input/output size
    """
    def __init__(self, config: dict):
        super().__init__()
        # Create sequential layers:
        # 1. Project from emb_dim to 4*emb_dim
        # 2. Apply GeLU activation with tanh approximation for better performance
        # 3. Project back to emb_dim
        self.layers = nn.Sequential(
            nn.Linear(config['emb_dim'], config['emb_dim'] * 4),  # Expansion layer
            nn.GELU(approximate="tanh"),                          # Non-linear activation
            nn.Linear(config['emb_dim'] * 4, config['emb_dim']), # Projection layer
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the feed-forward network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_tokens, emb_dim)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_tokens, emb_dim)
        """
        return self.layers(x)