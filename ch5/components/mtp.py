"""
Multi-Token Prediction (MTP) Module for DeepSeek Training

This module implements the Multi-Token Prediction mechanism used by DeepSeek during pre-training.
MTP predicts multiple future tokens at each position, densifying the training signal and improving
data efficiency, planning capabilities, and inference speed.

Key Concepts:
- Depth (D): Number of future tokens to predict at each position
- Sequential heads: Multiple prediction heads that build on each other's hidden states
- Causal information flow: Each head receives the hidden state from the previous head

Architecture per head:
    Hidden State (from prev head) + Token Embedding (at target position)
        ↓ RMSNorm on each
        ↓ Concatenate
        ↓ Linear Projection (2*d_model -> d_model)
        ↓ Transformer Block
        ↓ Linear Output (d_model -> vocab_size)
        ↓ Logits for next token prediction

Reference: DeepSeek-V3 Technical Report, Section 2.3
"""

from components.common import nn, torch, F
from components.transformer import TransformerBlock


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    
    RMSNorm is used before concatenating hidden states and embeddings in each MTP head.
    It normalizes input by dividing by the RMS (root mean square) of the input elements.
    
    Formula: RMSNorm(x) = x / sqrt(mean(x^2) + eps)
    
    Args:
        d_model (int): Dimension of the input
        eps (float): Small constant for numerical stability
    
    Example:
        >>> norm = RMSNorm(d_model=512)
        >>> x = torch.randn(2, 10, 512)
        >>> normalized = norm(x)
    """
    
    def __init__(self, d_model: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RMSNorm to input tensor.
        
        Args:
            x: Input tensor of any shape (..., d_model)
        
        Returns:
            Normalized tensor of same shape as input
        """
        # Calculate RMS: sqrt of mean of squares
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        # Normalize by dividing by RMS
        return x / rms


class MultiTokenPrediction(nn.Module):
    """
    Multi-Token Prediction mechanism for DeepSeek training.
    
    MTP predicts multiple future tokens (depth D) at each input position using
    sequential transformer blocks. Each head predicts one token into the future,
    with causality maintained by passing hidden states between heads.
    
    For an input sequence of length T:
    - At position i, predict tokens at positions i+1, i+2, ..., i+D
    - Use D sequential heads, each building on the previous head's output
    - Total predictions: (T-D) * D tokens (last D positions have fewer predictions)
    
    Args:
        d_model (int): Hidden dimension (e.g., 512)
        vocab_size (int): Size of vocabulary (e.g., 50257)
        num_heads (int): Number of sequential MTP prediction steps (depth D, e.g., 3)
        nhead (int): Number of attention heads in each Transformer block (e.g., 8)
        cfg (dict): Configuration dictionary for TransformerBlock
    
    Example:
        >>> cfg = {
        ...     "emb_dim": 512, "n_heads": 8, "kv_latent_dim": 128,
        ...     "num_experts": 8, "num_shared_experts": 1, "top_k": 2,
        ...     "expert_hidden_dim": 2048, "drop_rate": 0.1, "context_length": 256
        ... }
        >>> mtp = MultiTokenPrediction(d_model=512, vocab_size=5000, num_heads=3, nhead=8, cfg=cfg)
        >>> token_ids = torch.randint(0, 5000, (2, 10))  # batch=2, seq=10
        >>> logits = mtp(token_ids, embed_layer)
        >>> print(logits.shape)  # (2, T-D, D, vocab_size) = (2, 7, 3, 5000)
    """
    
    def __init__(self, d_model: int, vocab_size: int, num_heads: int, nhead: int, cfg: dict):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.num_heads = num_heads  # Depth D
        
        # Shared normalization layer for hidden states and embeddings
        self.rmsnorm = RMSNorm(d_model)
        
        # Embedding layer (shared with main model through weight tying)
        self.embed = nn.Embedding(vocab_size, d_model)
        
        # Output projection (shared with embedding through weight tying)
        self.unembed = nn.Linear(d_model, vocab_size, bias=False)
        # Weight tying: share weights between embedding and output projection
        self.unembed.weight = self.embed.weight
        
        # One linear projection per head: projects concatenated (2*d_model) back to d_model
        # Each head needs its own projection to learn different future predictions
        self.projections = nn.ModuleList([
            nn.Linear(2 * d_model, d_model) for _ in range(num_heads)
        ])
        
        # One transformer block per head: processes projected features
        # Each transformer learns to predict a different step into the future
        self.transformers = nn.ModuleList([
            TransformerBlock(cfg) for _ in range(num_heads)
        ])
    
    def forward(
        self,
        token_ids: torch.Tensor,
        init_hidden: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Forward pass for Multi-Token Prediction.
        
        Process:
        1. Get token embeddings for entire sequence
        2. For each position i where i + D < T:
            a. Initialize base hidden state (from init_hidden or embeddings)
            b. For each head k (k=0 to D-1):
                - Get future token embedding at position i+(k+1)
                - RMSNorm hidden state and embedding
                - Concatenate and project to d_model
                - Pass through transformer block
                - Generate logits for next token
                - Update hidden state for next head
        3. Stack all predictions into (batch, T-D, D, vocab_size) tensor
        
        Args:
            token_ids: Input token IDs of shape (batch, seq_len)
            init_hidden: Optional initial hidden states of shape (batch, seq_len, d_model)
                        If None, uses token embeddings as base hidden states
        
        Returns:
            logits: Predictions of shape (batch, T-D, D, vocab_size)
                   logits[b, i, k, :] = predicted logits for token at position i+(k+1)
                   where T is seq_len and D is num_heads (depth)
        
        Example:
            If token_ids has shape (2, 8) and num_heads=3:
            - Can predict for positions 0-4 (since 4+3 < 8)
            - Output shape: (2, 5, 3, vocab_size)
            - logits[0, 0, 0, :] predicts token at position 1
            - logits[0, 0, 1, :] predicts token at position 2
            - logits[0, 0, 2, :] predicts token at position 3
        """
        batch_size, seq_len = token_ids.shape
        device = token_ids.device
        
        # Get token embeddings for entire sequence: (batch, seq_len, d_model)
        embeds = self.embed(token_ids)
        
        # Initialize base hidden states
        # If no initial hidden provided, use embeddings as base
        if init_hidden is None:
            h0_seq = embeds  # (batch, seq_len, d_model)
        else:
            h0_seq = init_hidden  # (batch, seq_len, d_model)
        
        # List to collect predictions for each valid position
        outputs = []
        
        # Calculate maximum valid starting position
        # We can only predict from positions where i + D < seq_len
        max_i = seq_len - self.num_heads - 1
        
        # Iterate over positions where we can predict D tokens into the future
        for i in range(0, max_i + 1):
            # Get base hidden state at position i: (batch, d_model)
            h_prev = h0_seq[:, i, :]
            
            # Collect logits for all D prediction heads at this position
            logits_k = []
            
            # For each prediction head k, predict token at position i+(k+1)
            for k in range(self.num_heads):
                # Future position we're predicting
                future_pos = i + (k + 1)
                
                # Get embedding of the token at the target future position
                tok_embed = embeds[:, future_pos, :]  # (batch, d_model)
                
                # Step 1: RMS-normalize both hidden state and token embedding
                h_norm = self.rmsnorm(h_prev)      # (batch, d_model)
                e_norm = self.rmsnorm(tok_embed)   # (batch, d_model)
                
                # Step 2: Concatenate normalized vectors
                merged = torch.cat([h_norm, e_norm], dim=-1)  # (batch, 2*d_model)
                
                # Step 3: Project back to d_model
                proj = self.projections[k](merged)  # (batch, d_model)
                
                # Step 4: Pass through transformer block
                # Add sequence dimension for transformer: (batch, 1, d_model)
                x = proj.unsqueeze(1)
                x = self.transformers[k](x)  # (batch, 1, d_model)
                h_curr = x.squeeze(1)  # (batch, d_model)
                
                # Step 5: Project to vocabulary to get logits
                logits = self.unembed(h_curr)  # (batch, vocab_size)
                logits_k.append(logits)
                
                # Step 6: Update hidden state for next head (causal flow)
                h_prev = h_curr
            
            # Stack predictions for all D heads at position i: (batch, D, vocab_size)
            logits_k = torch.stack(logits_k, dim=1)
            outputs.append(logits_k)
        
        # Stack predictions for all positions: (batch, T-D, D, vocab_size)
        # where T=seq_len
        out = torch.stack(outputs, dim=1)
        
        return out


def compute_mtp_loss(
    logits: torch.Tensor,
    targets: torch.Tensor
) -> torch.Tensor:
    """
    Compute the Multi-Token Prediction loss.
    
    The MTP loss is computed by:
    1. For each position i and each depth k:
       - Compare logits[i, k] with targets[i + k + 1]
       - Compute cross-entropy loss
    2. Sum all losses and normalize by (L * D)
       where L is number of positions, D is depth
    
    Args:
        logits: Model predictions of shape (batch, L, D, vocab_size)
                where L = T-D positions, D = depth
        targets: Target token IDs of shape (batch, T) where T is full sequence length
    
    Returns:
        loss: Scalar tensor containing the averaged MTP loss
    
    Example:
        >>> logits = torch.randn(2, 5, 3, 5000)  # batch=2, L=5, D=3, vocab=5000
        >>> targets = torch.randint(0, 5000, (2, 8))  # batch=2, seq=8
        >>> loss = compute_mtp_loss(logits, targets)
        >>> print(loss.item())  # Scalar loss value
    
    Mathematical formulation:
        loss = (1/(L*D)) * Σ_i Σ_k CrossEntropy(logits[i,k], targets[i+k+1])
    """
    batch_size, L, D, vocab_size = logits.shape
    _, seq_len = targets.shape
    
    # Verify dimensions are compatible
    assert L == seq_len - D - 1, \
        f"Expected L={seq_len - D - 1}, got L={L}"
    
    # Initialize total loss
    loss = 0.0
    
    # Double loop over positions (i) and depth (k)
    for i in range(L):
        for k in range(D):
            # Get predicted logits for position i, depth k
            logit_ik = logits[:, i, k, :]  # (batch, vocab_size)
            
            # Get target token at position i + (k+1)
            target_ik = targets[:, i + (k + 1)]  # (batch,)
            
            # Compute cross-entropy loss for this (i, k) pair
            loss += F.cross_entropy(logit_ik, target_ik)
    
    # Normalize by total number of predictions (L * D)
    loss = loss / (L * D)
    
    return loss


# Example usage and testing
if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Configuration
    batch_size, seq_len, d_model, vocab_size = 1, 8, 8, 5000
    num_heads = 3  # Depth D
    nhead = 2  # Attention heads per transformer
    
    # Minimal config for TransformerBlock
    cfg = {
        "emb_dim": d_model,
        "n_heads": nhead,
        "kv_latent_dim": 4,
        "num_experts": 2,
        "num_shared_experts": 1,
        "top_k": 1,
        "expert_hidden_dim": 16,
        "drop_rate": 0.0,
        "context_length": seq_len
    }
    
    # Create MTP model
    print("="*60)
    print("Multi-Token Prediction Example")
    print("="*60)
    print(f"Configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Hidden dim: {d_model}")
    print(f"  Vocab size: {vocab_size}")
    print(f"  MTP depth (D): {num_heads}")
    print(f"  Attention heads: {nhead}")
    print()
    
    model = MultiTokenPrediction(d_model, vocab_size, num_heads, nhead, cfg)
    
    # Create random input tokens
    tokens = torch.randint(0, vocab_size, (batch_size, seq_len))
    print(f"Input tokens shape: {tokens.shape}")
    print(f"Input tokens: {tokens}")
    print()
    
    # Forward pass
    logits = model(tokens)
    print(f"Output logits shape: {logits.shape}")
    print(f"  Batch size: {logits.shape[0]}")
    print(f"  Positions (L=T-D): {logits.shape[1]} = {seq_len}-{num_heads}")
    print(f"  Depth (D): {logits.shape[2]}")
    print(f"  Vocab size: {logits.shape[3]}")
    print()
    
    # Inspect predictions at position i=0
    print("Predictions at position i=0:")
    print(f"  Head k=0 predicts position 1: {logits[0, 0, 0].argmax().item()}")
    print(f"  Head k=1 predicts position 2: {logits[0, 0, 1].argmax().item()}")
    print(f"  Head k=2 predicts position 3: {logits[0, 0, 2].argmax().item()}")
    print()
    
    # Get all predictions at i=0 across all heads
    pred_ids = logits[0, 0].argmax(dim=-1)
    print(f"All predicted tokens at i=0: {pred_ids}")
    print()
    
    # Compute loss
    targets = tokens  # Use same tokens as targets for demonstration
    loss = compute_mtp_loss(logits, targets)
    print(f"MTP Loss: {loss.item():.4f}")
    print()
    
    print("="*60)
    print("MTP module successfully tested!")
    print("="*60)