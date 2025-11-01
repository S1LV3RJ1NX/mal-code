"""
Sparse Mixture of Experts Layer.

This module implements the complete MoE layer with both shared and routed experts,
following DeepSeek's architecture innovations.
"""

from components.common import nn, torch
from components.expert import Expert
from components.router import NoisyTopKRouter


class SparseMoE(nn.Module):
    """
    Sparse Mixture of Experts layer with shared and routed experts.
    
    This implementation follows DeepSeek's innovations:
    1. Shared Experts: Always activated, capture common knowledge
    2. Routed Experts: Conditionally activated based on top-k routing
    3. Sparse Execution: Only top-k experts are actually executed
    
    Architecture:
        Input -> Router (select top-k experts)
               -> Shared Experts (always active)
               -> Routed Experts (top-k active)
               -> Weighted combination -> Output
    
    Args:
        n_embd (int): Input/output embedding dimension
        num_experts (int): Number of routed experts
        num_shared_experts (int): Number of shared experts (default: 1)
        top_k (int): Number of routed experts to activate per token
        expert_hidden_dim (int): Hidden dimension of expert networks
        dropout (float): Dropout rate for regularization
    
    Example:
        >>> moe = SparseMoE(
        ...     n_embd=512,
        ...     num_experts=8,
        ...     num_shared_experts=1,
        ...     top_k=2,
        ...     expert_hidden_dim=2048
        ... )
        >>> x = torch.randn(2, 10, 512)
        >>> output = moe(x)
        >>> print(output.shape)  # torch.Size([2, 10, 512])
    """
    
    def __init__(
        self,
        n_embd: int,
        num_experts: int,
        num_shared_experts: int,
        top_k: int,
        expert_hidden_dim: int,
        dropout: float = 0.1
    ):
        super().__init__()
        self.num_experts = num_experts
        self.num_shared_experts = num_shared_experts
        self.top_k = top_k
        
        # Router for selecting routed experts
        self.router = NoisyTopKRouter(n_embd, num_experts, top_k)
        
        # Routed experts (conditionally activated)
        self.experts = nn.ModuleList([
            Expert(n_embd, expert_hidden_dim, dropout)
            for _ in range(num_experts)
        ])
        
        # Shared experts (always activated)
        if num_shared_experts > 0:
            self.shared_experts = nn.ModuleList([
                Expert(n_embd, expert_hidden_dim, dropout)
                for _ in range(num_shared_experts)
            ])
        else:
            self.shared_experts = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Mixture of Experts layer.
        
        Process:
        1. Route tokens to top-k experts
        2. Apply shared experts (if any) to all tokens
        3. Apply selected routed experts
        4. Combine outputs with routing weights
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, n_embd)
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, n_embd)
        """
        batch_size, seq_len, n_embd = x.shape
        
        # Get routing weights and expert indices
        routing_weights, expert_indices = self.router(x)
        # routing_weights: (batch_size, seq_len, num_experts)
        # expert_indices: (batch_size, seq_len, top_k)
        
        # Initialize final output
        final_output = torch.zeros_like(x)
        
        # Reshape inputs for efficient batch processing
        # Flatten batch and sequence dimensions
        flat_x = x.view(-1, n_embd)  # (batch_size * seq_len, n_embd)
        flat_routing_weights = routing_weights.view(-1, self.num_experts)
        
        # Process each routed expert
        # We iterate through experts and process all tokens assigned to each expert
        for i, expert in enumerate(self.experts):
            # Create mask for tokens assigned to this expert
            # expert_mask checks if expert i is in the top-k for each token
            expert_mask = (expert_indices == i).any(dim=-1)  # (batch_size, seq_len)
            flat_mask = expert_mask.view(-1)  # (batch_size * seq_len)
            
            # Only process if this expert is selected for at least one token
            if flat_mask.any():
                # Extract inputs for this expert
                expert_input = flat_x[flat_mask]  # (num_selected_tokens, n_embd)
                
                # Apply expert
                expert_output = expert(expert_input)  # (num_selected_tokens, n_embd)
                
                # Get routing weights for this expert
                # Extract weights for tokens that use this expert
                expert_weights = flat_routing_weights[flat_mask, i].unsqueeze(1)
                # (num_selected_tokens, 1)
                
                # Weight the expert output
                weighted_output = expert_output * expert_weights
                
                # Add to final output at the correct positions
                final_output.view(-1, n_embd)[flat_mask] += weighted_output
        
        # Add shared expert outputs (if any)
        if self.shared_experts is not None:
            for shared_expert in self.shared_experts:
                shared_output = shared_expert(x)
                # Shared experts contribute equally to all tokens
                # Normalize by number of shared experts
                final_output += shared_output / self.num_shared_experts
        
        return final_output