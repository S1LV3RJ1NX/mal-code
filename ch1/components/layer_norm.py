from components.common import nn, torch


class LayerNorm(nn.Module):
    """Layer normalization layer that normalizes the input tensor across the last dimension.

    This implementation allows for more flexibility because it learns separate scaling and shifting parameters for each feature.
    This can lead to better performance in scenarios where different features require different levels of normalization.

    Args:
        emb_dim (int): Dimension of the input tensor
        eps (float, optional): Small constant to avoid division by zero
    """

    def __init__(self, emb_dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # gamma is the multiplicative parameter
        self.gamma = nn.Parameter(torch.ones(emb_dim))
        # beta is the additive parameter
        self.beta = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * norm_x + self.beta
