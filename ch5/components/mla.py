"""
Multi-Head Latent Attention Module.

This module implements DeepSeek's Multi-Head Latent Attention mechanism,
which dramatically reduces KV cache memory requirements while maintaining
or improving model performance.
"""

from components.common import nn, torch, F, math


class MultiHeadLatentAttention(nn.Module):
    """
    Multi-Head Latent Attention (MLA) mechanism from DeepSeek.
    
    Key innovations:
    1. Projects KV to low-dimensional latent space (compression)
    2. Caches only latent representation (4-8× memory savings)
    3. Uses absorption trick for efficient computation
    4. Maintains multi-head diversity without multi-head KV cache
    
    Memory savings compared to standard MHA:
    - Standard MHA: 2 × n_heads × head_dim per token
    - MLA: kv_latent_dim per token
    - Typical reduction: 4-8×
    
    Args:
        n_embd (int): Embedding dimension (d_model)
        n_heads (int): Number of attention heads
        kv_latent_dim (int): Latent dimension for KV compression
        context_len (int): Maximum sequence length
        dropout (float): Dropout rate
    
    Example:
        >>> mla = MultiHeadLatentAttention(
        ...     n_embd=512, n_heads=8, kv_latent_dim=128, context_len=256
        ... )
        >>> x = torch.randn(2, 10, 512)
        >>> output = mla(x)
        >>> print(output.shape)  # torch.Size([2, 10, 512])
    """
    
    def __init__(
        self,
        n_embd: int,
        n_heads: int,
        kv_latent_dim: int,
        context_len: int,
        dropout: float = 0.1
    ):
        super().__init__()
        assert n_embd % n_heads == 0, "n_embd must be divisible by n_heads"
        
        self.n_embd = n_embd
        self.n_heads = n_heads
        self.head_dim = n_embd // n_heads
        self.kv_latent_dim = kv_latent_dim
        self.context_len = context_len
        
        # Query projection (standard)
        self.W_q = nn.Linear(n_embd, n_embd, bias=False)
        
        # KV compression: project to low-dimensional latent space
        self.W_dkv = nn.Linear(n_embd, kv_latent_dim, bias=False)
        
        # KV decompression: expand back from latent space
        self.W_uk = nn.Linear(kv_latent_dim, n_embd, bias=False)  # Keys
        self.W_uv = nn.Linear(kv_latent_dim, n_embd, bias=False)  # Values
        
        # Output projection
        self.W_o = nn.Linear(n_embd, n_embd, bias=False)
        
        # Layer normalization for latent representation
        self.ln = nn.LayerNorm(kv_latent_dim)
        
        # Dropout layers
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        
        # Causal mask
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(context_len, context_len)).view(
                1, 1, context_len, context_len
            )
        )
        # Initialize absorbed projection cache
        self.absorbed_qk_proj = None

    def _compute_absorbed_projection(self):
        """
        Compute and cache the absorbed query-key projection.
        
        This is the "absorption trick": instead of computing Q @ K^T where
        Q = X @ W_q and K = C_kv @ W_uk, we precompute W_q @ W_uk^T and
        compute (X @ [W_q @ W_uk^T]) @ C_kv^T, saving computation.
        
        Shape: (n_heads, head_dim, kv_latent_dim)
        """
        if self.training or self.absorbed_qk_proj is None:
            # Compute absorbed projection: W_q @ W_uk^T
            absorbed = self.W_q.weight @ self.W_uk.weight  # (n_embd, kv_latent_dim)
            
            # Reshape for multi-head processing
            # Split along the first dimension (n_embd) into heads
            self.absorbed_qk_proj = absorbed.view(
                self.n_heads, self.head_dim, self.kv_latent_dim
            )
        return self.absorbed_qk_proj
    
    # def _get_absorbed_projection(self):
    #     """
    #     Get the absorbed query-key projection with smart caching.
        
    #     This is the "absorption trick": instead of computing Q @ K^T where
    #     Q = X @ W_q and K = C_kv @ W_uk, we precompute W_q @ W_uk^T and
    #     compute (X @ [W_q @ W_uk^T]) @ C_kv^T, saving computation.
        
    #     We cache the result and only recompute when weights actually change.
    #     During training, weights change frequently, so we get some caching benefit
    #     within each forward pass. During inference, weights are static, so we get
    #     maximum caching benefit.
        
    #     Shape: (n_heads, head_dim, kv_latent_dim)
    #     """
    #     # Use tensor data pointers as a more reliable change detection
    #     current_q_ptr = self.W_q.weight.data_ptr()
    #     current_uk_ptr = self.W_uk.weight.data_ptr()
        
    #     # Check if this is first time or if weight tensors have been replaced
    #     need_recompute = (
    #         self.absorbed_qk_proj is None or 
    #         self._weights_hash != (current_q_ptr, current_uk_ptr)
    #     )
        
    #     if need_recompute:
    #         # Compute absorbed projection: W_q @ W_uk^T
    #         absorbed = self.W_q.weight @ self.W_uk.weight  # (n_embd, kv_latent_dim)
            
    #         # Reshape for multi-head processing and detach to avoid autograd issues
    #         self.absorbed_qk_proj = absorbed.view(
    #             self.n_heads, self.head_dim, self.kv_latent_dim
    #         ).detach()
            
    #         # Update hash with tensor pointers
    #         self._weights_hash = (current_q_ptr, current_uk_ptr)
        
    #     # Return a fresh tensor that participates in autograd
    #     # This ensures gradients flow back to W_q and W_uk properly
    #     if self.training:
    #         # During training, recompute to maintain autograd graph
    #         absorbed = self.W_q.weight @ self.W_uk.weight
    #         return absorbed.view(self.n_heads, self.head_dim, self.kv_latent_dim)
    #     else:
    #         # During inference, use cached version for maximum speed
    #         return self.absorbed_qk_proj
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through Multi-Head Latent Attention.
        
        Process:
        1. Compress input to latent KV representation (cached in practice)
        2. Compute absorbed queries using precomputed projection
        3. Compute attention scores efficiently
        4. Decompress values and apply attention
        5. Project output
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, n_embd)
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, n_embd)
        """
        batch_size, seq_len, n_embd = x.shape
        
        # Step 1: Compress to latent KV representation
        # This is what we cache (much smaller than standard KV cache!)
        kv_latent = self.ln(self.W_dkv(x))  # (batch_size, seq_len, kv_latent_dim)
        
        # Step 2: Compute absorbed query-key projection
        self._compute_absorbed_projection()
        
        # Step 3: Split input for multi-head processing
        x_split = x.view(batch_size, seq_len, self.n_heads, self.head_dim)
        # (batch_size, seq_len, n_heads, head_dim)
        
        # Step 4: Compute attention scores efficiently using absorption trick
        attn_scores = torch.zeros(
            batch_size, self.n_heads, seq_len, seq_len,
            device=x.device, dtype=x.dtype
        )
        
        for head_idx in range(self.n_heads):
            # Absorbed query for this head
            # (batch_size, seq_len, head_dim) @ (head_dim, kv_latent_dim)
            # -> (batch_size, seq_len, kv_latent_dim)
            absorbed_q = x_split[:, :, head_idx] @ self.absorbed_qk_proj[head_idx]
            
            # Attention scores: absorbed_q @ kv_latent^T
            # (batch_size, seq_len, kv_latent_dim) @ (batch_size, kv_latent_dim, seq_len)
            # -> (batch_size, seq_len, seq_len)
            attn_scores[:, head_idx] = torch.bmm(absorbed_q, kv_latent.transpose(1, 2))
        
        # Step 5: Scale, mask, and softmax
        attn_scores = attn_scores / math.sqrt(self.head_dim)
        
        # Apply causal mask
        attn_scores = attn_scores.masked_fill(
            self.causal_mask[:, :, :seq_len, :seq_len] == 0,
            float('-inf')
        )
        
        # Apply softmax
        attn_weights = F.softmax(attn_scores, dim=-1)  # (batch_size, n_heads, seq_len, seq_len)
        attn_weights = self.attn_dropout(attn_weights)
        
        # Step 6: Decompress values from latent space
        v = self.W_uv(kv_latent)  # (batch_size, seq_len, n_embd)
        
        # Split into heads
        v_split = v.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        # (batch_size, n_heads, seq_len, head_dim)
        
        # Step 7: Apply attention to values
        # (batch_size, n_heads, seq_len, seq_len) @ (batch_size, n_heads, seq_len, head_dim)
        # -> (batch_size, n_heads, seq_len, head_dim)
        attn_output = attn_weights @ v_split
        
        # Step 8: Concatenate heads and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, n_embd
        )
        
        # Final output projection
        output = self.W_o(attn_output)
        output = self.resid_dropout(output)
        
        return output