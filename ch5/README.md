# DeepSeek Language Model Implementation

A complete from-scratch implementation of a language model combining DeepSeek's two major innovations:
- **Multi-Head Latent Attention (MLA)**: 4-8× reduction in KV cache memory
- **Mixture of Experts (MoE)**: Sparse expert routing for higher capacity

This implementation is designed for educational purposes and follows the same pedagogical approach as Chapter 2's GPT-2 implementation.

## 🌟 Features

### Multi-Head Latent Attention
- Compresses KV cache to low-dimensional latent space
- Implements the "absorption trick" for efficient computation
- 8× memory savings with default configuration (512 dim → 128 latent)
- Maintains multi-head diversity without multi-head cache

### Mixture of Experts
- Sparse top-k routing with load balancing
- Shared experts for common knowledge
- Routed experts for specialization
- Only 33% of parameters active per forward pass (with default config)

### Training Infrastructure
- Clean, modular architecture
- Comprehensive training loop with validation
- Checkpointing and model saving
- Learning rate scheduling with cosine annealing
- Gradient clipping for stability

## 📁 Project Structure

```
deepseek_final/
├── components/
│   ├── __init__.py
│   ├── common.py                        # Common imports
│   ├── expert.py                        # Expert network
│   ├── router.py                        # Top-K router
│   ├── moe.py                          # Sparse MoE layer
│   ├── multi_head_latent_attention.py  # MLA implementation
│   ├── transformer_block.py            # Transformer block
│   └── deepseek_model.py               # Complete model
├── constants.py                         # Model configurations
├── dataset.py                           # Data loading
├── trainer.py                           # Training loop
├── utils.py                             # Helper functions
├── train.py                             # Main training script
├── inference.py                         # Text generation
└── README.md                            # This file
```

## 🚀 Quick Start

### 1. Installation

```bash
# Install dependencies
pip install torch tiktoken datasets tqdm
```

### 2. Training

Train on Paul Graham's essays (same dataset as Chapter 2):

```bash
# Quick test with small config (faster training)
python train.py  # Set USE_SMALL_CONFIG = True in train.py

# Full training with default config
python train.py  # Set USE_SMALL_CONFIG = False in train.py
```

The training script will:
- Load the Paul Graham essays dataset
- Create train/validation splits
- Train the model with progress bars
- Save checkpoints every 2 epochs
- Save the best model based on validation loss

### 3. Text Generation

Generate text using the trained model:

```bash
# Interactive mode (recommended)
python inference.py --model_path checkpoints/best_model.pt --interactive

# Single generation
python inference.py \
    --model_path checkpoints/best_model.pt \
    --prompt "The best way to learn programming is" \
    --max_new_tokens 100 \
    --temperature 0.8 \
    --top_k 40
```

Interactive mode allows you to:
- Enter multiple prompts
- Adjust generation parameters on the fly
- Continuously generate text until you quit

## 📊 Model Configurations

### Small (Quick Testing)
```python
DEEPSEEK_CONFIG_SMALL = {
    "vocab_size": 50257,
    "context_length": 128,
    "emb_dim": 256,
    "n_heads": 4,
    "n_layers": 4,
    "kv_latent_dim": 64,      # 4× compression
    "num_experts": 4,
    "num_shared_experts": 1,
    "top_k": 2,
    "expert_hidden_dim": 1024,
}
```

### Default (Full Training)
```python
DEEPSEEK_CONFIG = {
    "vocab_size": 50257,
    "context_length": 256,
    "emb_dim": 512,
    "n_heads": 8,
    "n_layers": 6,
    "kv_latent_dim": 128,     # 4× compression
    "num_experts": 8,
    "num_shared_experts": 1,
    "top_k": 2,
    "expert_hidden_dim": 2048,
}
```

### Large (Production)
```python
DEEPSEEK_CONFIG_LARGE = {
    "vocab_size": 50257,
    "context_length": 512,
    "emb_dim": 1024,
    "n_heads": 16,
    "n_layers": 12,
    "kv_latent_dim": 128,     # 8× compression
    "num_experts": 16,
    "num_shared_experts": 2,
    "top_k": 4,
    "expert_hidden_dim": 4096,
}
```

## 💡 Key Implementation Details

### Multi-Head Latent Attention

**Memory Savings:**
- Standard MHA: 2 × n_heads × head_dim = 2 × 8 × 64 = 1024 values/token
- MLA: kv_latent_dim = 128 values/token
- **Result: 8× reduction**

**Absorption Trick:**
```python
# Instead of: Q @ K^T where Q = X @ W_q, K = C_kv @ W_uk
# We precompute: W_q @ W_uk^T and compute (X @ [W_q @ W_uk^T]) @ C_kv^T
# This saves computation during inference
```

### Mixture of Experts

**Parameter Efficiency:**
```
Total parameters: 19M
  - Routed experts (8): 16.8M
  - Shared experts (1): 2.1M

Active per token: 6.3M (33%)
  - Routed (top-2): 4.2M
  - Shared (all): 2.1M

Efficiency gain: 19M / 6.3M = 3×
```

**Sparse Execution:**
```python
# Only execute experts that receive at least one token
for i, expert in enumerate(self.experts):
    expert_mask = (expert_indices == i).any(dim=-1)
    if expert_mask.any():  # Skip unused experts!
        # Process only tokens assigned to this expert
```

## 🧪 Experiments

Try these experiments to understand the model better:

### 1. Vary Latent Dimension
```python
# In constants.py, try different compression ratios
"kv_latent_dim": 256,  # 2× compression
"kv_latent_dim": 128,  # 4× compression
"kv_latent_dim": 64,   # 8× compression
```

### 2. Expert Count Scaling
```python
# Keep total parameters constant while varying granularity
# Config A: 8 experts × 2048 dim = 16M params
"num_experts": 8,
"expert_hidden_dim": 2048,

# Config B: 16 experts × 1024 dim = 16M params
"num_experts": 16,
"expert_hidden_dim": 1024,
```

### 3. Shared vs Routed Experts
```python
# Try different shared expert counts
"num_shared_experts": 0,  # No shared experts
"num_shared_experts": 1,  # One shared expert
"num_shared_experts": 2,  # Two shared experts
```

### 4. Top-K Values
```python
# More active experts = more capacity but more compute
"top_k": 1,  # Very sparse
"top_k": 2,  # Balanced (default)
"top_k": 4,  # Less sparse
```

## 📈 Expected Results

With the default configuration on Paul Graham essays:

### Training
- **Initial loss**: ~10-11
- **Final training loss**: ~3-4
- **Final validation loss**: ~4-5
- **Training time**: ~30-60 minutes on GPU

### Generation Quality
The model should generate coherent text in Paul Graham's style:

```
Prompt: "The best way to learn programming is"

Generated: "The best way to learn programming is to write programs. 
You can't learn to program by reading about it. You have to actually 
write code, make mistakes, debug them, and understand why things work 
the way they do. Start with small projects and gradually work your way 
up to more complex ones..."
```

## 🔧 Troubleshooting

### Out of Memory
- Reduce `batch_size` in `train.py`
- Reduce `context_length` in config
- Use `DEEPSEEK_CONFIG_SMALL`

### Slow Training
- Ensure you're using GPU (check device in output)
- Reduce `num_experts` or `expert_hidden_dim`
- Use smaller dataset

### Poor Generation Quality
- Train for more epochs (increase `num_epochs`)
- Adjust generation parameters (temperature, top_k)
- Check validation loss (should be decreasing)

### Expert Collapse
- All tokens routing to same experts
- Increase noise in router (already implemented)
- Reduce learning rate

## 📚 Understanding the Code

### For Students

This implementation is designed to be educational:

1. **Start with simple components**: `expert.py`, `router.py`
2. **Understand MoE**: `moe.py` combines experts and router
3. **Learn MLA**: `multi_head_latent_attention.py` shows compression
4. **See integration**: `transformer_block.py` combines both
5. **Complete model**: `deepseek_model.py` stacks blocks
6. **Training**: `trainer.py` and `train.py` show the full pipeline

Each file is heavily commented with:
- Docstrings explaining purpose
- Shape comments showing tensor dimensions
- Step-by-step explanations of algorithms
- Examples of usage

### Key Files to Study

1. **`components/moe.py`**: See how sparse execution works
2. **`components/multi_head_latent_attention.py`**: Understand the absorption trick
3. **`trainer.py`**: Learn training loop best practices
4. **`utils.py`**: Study text generation algorithms

## 🎯 Learning Objectives

After studying this implementation, you should understand:

1. **MLA Benefits**: Why and how latent compression reduces memory
2. **Absorption Trick**: Mathematical optimization for efficiency
3. **Sparse MoE**: How to route tokens and execute experts efficiently
4. **Load Balancing**: Why noisy routing prevents expert collapse
5. **Training Tricks**: Gradient clipping, learning rate scheduling, checkpointing

## 🔗 Relation to Chapter 5

This code accompanies Chapter 5 of the book, which covers:

1. **Theory** (first half): MLA and MoE concepts
2. **Implementation** (this code): Practical realization
3. **Training** (train.py): End-to-end pipeline
4. **Analysis**: Efficiency calculations and ablations

The code follows the same pedagogical approach as Chapter 2's GPT-2 implementation, making it easy to compare and contrast the architectures.

## 📖 References

- **DeepSeek-V2 Paper**: [Link](./deepseek_v2.pdf)
- **DeepSeek-V3 Paper**: [Link](./deepseek_v3.pdf)
- **DeepSeek-MoE Paper**: [Link](./deepseek_moe.pdf)