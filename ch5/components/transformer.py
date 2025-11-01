"""
Transformer Block with Multi-Head Latent Attention and Mixture of Experts.

This module implements a transformer block that uses MLA for attention
and MoE for the feed-forward network, combining both DeepSeek innovations.
"""

from components.common import nn, torch
from components.mla import MultiHeadLatentAttention
from components.moe import SparseMoE


class TransformerBlock(nn.Module):
    """
    TransformerBlock implements a single block combining MLA and MoE.
    
    It consists of multi-head latent attention followed by a mixture of experts,
    with layer normalization applied before each sub-layer and residual connections.
    
    Architecture:
        Input
          ↓
        LayerNorm -> Multi-Head Latent Attention (MLA) -> Residual
          ↓
        LayerNorm -> Mixture of Experts (MoE) -> Residual
          ↓
        Output
    
    This combines DeepSeek's two major innovations:
    - MLA: Reduces KV cache memory by 4-8×
    - MoE: Increases model capacity without proportional compute increase

    Args:
        cfg (dict): Configuration dictionary containing the following keys:
            - emb_dim (int): The dimensionality of the embedding space
            - context_length (int): The maximum length of the input sequences
            - n_heads (int): The number of attention heads
            - kv_latent_dim (int): The latent dimension for MLA compression
            - num_experts (int): Number of routed experts in MoE
            - num_shared_experts (int): Number of shared experts
            - top_k (int): Number of experts to activate per token
            - expert_hidden_dim (int): Hidden dimension for expert networks
            - drop_rate (float): Dropout rate for regularization
    
    Example:
        >>> cfg = {
        ...     "emb_dim": 512, "context_length": 256, "n_heads": 8,
        ...     "kv_latent_dim": 128, "num_experts": 8, "num_shared_experts": 1,
        ...     "top_k": 2, "expert_hidden_dim": 2048, "drop_rate": 0.1
        ... }
        >>> block = TransformerBlock(cfg)
        >>> x = torch.randn(2, 10, 512)
        >>> output = block(x)
        >>> print(output.shape)  # torch.Size([2, 10, 512])
    """

    def __init__(self, cfg: dict):
        super().__init__()
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(cfg["emb_dim"])
        self.norm2 = nn.LayerNorm(cfg["emb_dim"])
        
        # Multi-head latent attention
        self.att = MultiHeadLatentAttention(
            n_embd=cfg["emb_dim"],
            n_heads=cfg["n_heads"],
            kv_latent_dim=cfg["kv_latent_dim"],
            context_len=cfg["context_length"],
            dropout=cfg["drop_rate"]
        )
        
        # Mixture of Experts (replaces standard FFN)
        self.moe = SparseMoE(
            n_embd=cfg["emb_dim"],
            num_experts=cfg["num_experts"],
            num_shared_experts=cfg["num_shared_experts"],
            top_k=cfg["top_k"],
            expert_hidden_dim=cfg["expert_hidden_dim"],
            dropout=cfg["drop_rate"]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the TransformerBlock.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, num_tokens, emb_dim].

        Returns:
            torch.Tensor: Output tensor of the same shape as input after applying
                          attention, MoE, and residual connections.
        """
        # MLA block with residual connection
        # Pre-normalization: apply LayerNorm before attention
        x = x + self.att(self.norm1(x))
        
        # MoE block with residual connection
        # Pre-normalization: apply LayerNorm before MoE
        x = x + self.moe(self.norm2(x))

        return x