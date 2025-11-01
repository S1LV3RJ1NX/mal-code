"""
Top-K Router for Mixture of Experts.

This module implements the routing mechanism that determines which experts
should process each token. It uses a noisy top-k selection strategy for
load balancing during training.
"""

from components.common import nn, torch, F


class NoisyTopKRouter(nn.Module):
    """
    Noisy Top-K Router that assigns tokens to the most suitable experts.
    
    The router computes logits for each expert using a linear projection,
    adds learnable Gaussian noise during training for load balancing,
    then selects the top-k experts with the highest scores.
    
    Key features:
    - Sparse routing: only top-k experts receive non-zero weights
    - Noisy training: Gaussian noise prevents expert collapse
    - Efficient inference: deterministic routing without noise
    
    Args:
        n_embd (int): Input embedding dimension
        num_experts (int): Total number of experts to route among
        top_k (int): Number of experts to activate per token
    
    How it works:
        1. Project input to expert logits: x @ W -> logits
        2. Add noise during training: logits + N(0, σ²)
        3. Select top-k experts: topk(noisy_logits) -> top_k_logits, indices
        4. Create sparse logits: scatter top_k values into -inf tensor
        5. Apply softmax to get routing weights
    
    Example:
        >>> router = NoisyTopKRouter(n_embd=512, num_experts=8, top_k=2)
        >>> x = torch.randn(2, 10, 512)  # (batch_size, seq_len, n_embd)
        >>> routing_weights, expert_indices = router(x)
        >>> print(routing_weights.shape)  # torch.Size([2, 10, 8])
        >>> print(expert_indices.shape)   # torch.Size([2, 10, 2])
    """
    
    def __init__(self, n_embd: int, num_experts: int, top_k: int):
        super().__init__()
        self.top_k = top_k
        self.num_experts = num_experts
        
        # Linear layer to compute expert affinity logits
        self.topk_linear = nn.Linear(n_embd, num_experts, bias=False)
        
        # Linear layer to compute noise scale for load balancing
        self.noise_linear = nn.Linear(n_embd, num_experts, bias=False)
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Route tokens to top-k experts with optional noise.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, n_embd)
        
        Returns:
            tuple containing:
                - routing_weights (torch.Tensor): Softmax weights for all experts,
                  shape (batch_size, seq_len, num_experts)
                - expert_indices (torch.Tensor): Indices of selected top-k experts,
                  shape (batch_size, seq_len, top_k)
        """
        # Compute base routing logits
        logits = self.topk_linear(x)  # (batch_size, seq_len, num_experts)
        
        # Add Gaussian noise during training for load balancing
        if self.training:
            noise_logits = self.noise_linear(x)
            # Generate noise scaled by softplus of noise_logits
            # softplus ensures positive scale: softplus(x) = log(1 + exp(x))
            noise = torch.randn_like(logits) * F.softplus(noise_logits)
            noisy_logits = logits + noise
        else:
            noisy_logits = logits
        
        # Select top-k experts from noisy logits
        top_k_logits, expert_indices = torch.topk(
            noisy_logits, self.top_k, dim=-1
        )  # Both: (batch_size, seq_len, top_k)
        
        # Create sparse logits tensor (only top-k have non-zero values)
        # This is crucial: we set all non-selected experts to -inf
        sparse_logits = torch.full_like(noisy_logits, float('-inf'))
        sparse_logits.scatter_(
            dim=-1,
            index=expert_indices,
            src=top_k_logits
        )  # (batch_size, seq_len, num_experts)
        
        # Apply softmax to get routing weights
        # Only top-k experts will have non-zero weights due to -inf values
        routing_weights = F.softmax(sparse_logits, dim=-1)
        
        return routing_weights, expert_indices