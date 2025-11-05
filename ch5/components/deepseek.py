"""
DeepSeek Language Model.

This module implements a complete autoregressive language model combining
Multi-Head Latent Attention (MLA) and Mixture of Experts (MoE).

Optional Multi-Token Prediction (MTP) can be enabled during training for
improved data efficiency and model quality.
"""

from components.common import nn, torch, F
from components.transformer import TransformerBlock
from components.mtp import MultiTokenPrediction, compute_mtp_loss


class DeepSeekModel(nn.Module):
    """
    Complete DeepSeek language model with MLA and MoE.
    
    Architecture:
        Token Embeddings + Positional Embeddings
          ↓
        Dropout
          ↓
        Stack of Transformer Blocks (with MLA + MoE)
          ↓
        Layer Norm
          ↓
        Language Model Head (linear projection to vocabulary)
    
    Optional Multi-Token Prediction (MTP):
        During training, MTP can be enabled to predict multiple future tokens
        at each position, providing denser training signals and improving
        data efficiency. MTP is disabled during inference.
    
    Args:
        cfg (dict): Configuration dictionary containing:
            - vocab_size (int): Size of the vocabulary
            - context_length (int): Maximum sequence length
            - emb_dim (int): Embedding dimension
            - n_heads (int): Number of attention heads
            - kv_latent_dim (int): Latent dimension for MLA
            - n_layers (int): Number of transformer blocks
            - num_experts (int): Number of routed experts
            - num_shared_experts (int): Number of shared experts
            - top_k (int): Number of experts to activate per token
            - expert_hidden_dim (int): Hidden dimension for experts
            - drop_rate (float): Dropout rate
            - use_mtp (bool, optional): Enable Multi-Token Prediction (default: False)
            - mtp_depth (int, optional): Number of future tokens to predict (default: 3)
            - mtp_weight (float, optional): Weight for MTP loss (default: 0.3)
    
    Example:
        >>> cfg = {
        ...     "vocab_size": 50257, "context_length": 256, "emb_dim": 512,
        ...     "n_heads": 8, "kv_latent_dim": 128, "n_layers": 6,
        ...     "num_experts": 8, "num_shared_experts": 1, "top_k": 2,
        ...     "expert_hidden_dim": 2048, "drop_rate": 0.1,
        ...     "use_mtp": True, "mtp_depth": 3, "mtp_weight": 0.3
        ... }
        >>> model = DeepSeekModel(cfg)
        >>> input_ids = torch.randint(0, 50257, (2, 10))
        >>> # Training with MTP
        >>> model.train()
        >>> logits, loss = model(input_ids, input_ids)
        >>> # Inference (MTP automatically disabled)
        >>> model.eval()
        >>> logits = model(input_ids)
    """
    
    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        
        # Token and position embeddings
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        
        # Stack of transformer blocks with MLA and MoE
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        
        # Final layer normalization
        self.final_norm = nn.LayerNorm(cfg["emb_dim"])
        
        # Language model head
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)
        
        # Weight tying: share weights between token embedding and output projection
        self.out_head.weight = self.tok_emb.weight
        
        # Multi-Token Prediction (optional, for training only)
        self.use_mtp = cfg.get("use_mtp", False)
        if self.use_mtp:
            self.mtp = MultiTokenPrediction(
                d_model=cfg["emb_dim"],
                vocab_size=cfg["vocab_size"],
                num_heads=cfg.get("mtp_depth", 3),
                nhead=cfg["n_heads"],
                cfg=cfg
            )
            # Weight tying: share embeddings between base model and MTP
            self.mtp.embed.weight = self.tok_emb.weight
            self.mtp.unembed.weight = self.out_head.weight
            self.mtp_weight = cfg.get("mtp_weight", 0.3)
            print(f"MTP enabled: depth={cfg.get('mtp_depth', 3)}, weight={self.mtp_weight}")
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """
        Initialize weights using normal distribution.
        
        Args:
            module: PyTorch module to initialize
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def forward(self, in_idx: torch.Tensor, target_idx: torch.Tensor = None):
        """
        Forward pass through the language model.
        
        During training with MTP enabled, computes both standard next-token
        prediction loss and multi-token prediction loss, combining them with
        configurable weighting.
        
        During inference or when MTP is disabled, performs standard autoregressive
        forward pass.
        
        Args:
            in_idx (torch.Tensor): Input token indices of shape (batch_size, seq_len)
            target_idx (torch.Tensor, optional): Target token indices for loss computation
        
        Returns:
            If target_idx is None:
                torch.Tensor: Output logits of shape (batch_size, seq_len, vocab_size)
            If target_idx is provided:
                tuple: (logits, loss)
                    - logits (torch.Tensor): Output logits
                    - loss (torch.Tensor): Combined loss (standard + MTP if enabled)
        """
        batch_size, seq_len = in_idx.shape
        
        # Validate sequence length
        assert seq_len <= self.cfg["context_length"], \
            f"Sequence length {seq_len} exceeds context length {self.cfg['context_length']}"
        
        # Get token embeddings
        tok_embeds = self.tok_emb(in_idx)  # (batch_size, seq_len, emb_dim)
        
        # Get position embeddings
        pos_indices = torch.arange(seq_len, device=in_idx.device)
        pos_embeds = self.pos_emb(pos_indices)  # (seq_len, emb_dim)
        
        # Combine embeddings with dropout
        x = self.drop_emb(tok_embeds + pos_embeds)
        
        # Apply transformer blocks
        x = self.trf_blocks(x)
        
        # Final layer normalization
        x = self.final_norm(x)
        
        # Project to vocabulary
        logits = self.out_head(x)  # (batch_size, seq_len, vocab_size)
        
        # Compute loss if targets provided
        if target_idx is None:
            loss = None
        else:
            # Standard next-token prediction loss
            standard_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                target_idx.view(-1)
            )
            
            # Add MTP loss if enabled and in training mode
            if self.use_mtp and self.training:
                # Get hidden states for MTP (before final projection)
                # We reuse x which is the output after final_norm
                mtp_logits = self.mtp(in_idx, init_hidden=x)
                mtp_loss = compute_mtp_loss(mtp_logits, target_idx)
                
                # Combine losses: (1-α) * standard + α * MTP
                loss = (1 - self.mtp_weight) * standard_loss + self.mtp_weight * mtp_loss
            else:
                loss = standard_loss
        
        return logits if loss is None else (logits, loss)
    
    def count_parameters(self) -> dict:
        """
        Count the number of parameters in the model.
        
        Returns:
            dict: Dictionary containing parameter counts:
                - total: Total parameters
                - trainable: Trainable parameters
                - routed_experts: Parameters in routed experts
                - shared_experts: Parameters in shared experts
                - non_expert: Parameters outside experts (embeddings, attention, etc.)
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        # Count MoE-specific parameters
        expert_params = 0
        shared_expert_params = 0
        
        for block in self.trf_blocks:
            # Routed experts
            for expert in block.moe.experts:
                expert_params += sum(p.numel() for p in expert.parameters())
            
            # Shared experts
            if block.moe.shared_experts is not None:
                for expert in block.moe.shared_experts:
                    shared_expert_params += sum(p.numel() for p in expert.parameters())
        
        non_expert_params = total_params - expert_params - shared_expert_params
        
        return {
            "total": total_params,
            "trainable": trainable_params,
            "routed_experts": expert_params,
            "shared_experts": shared_expert_params,
            "non_expert": non_expert_params,
        }