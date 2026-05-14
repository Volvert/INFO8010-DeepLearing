# =============================================================================
# MultiHeadSelfAttention
# =============================================================================
"""
This file implements the multi-head self-attention module used in each
Transformer encoder block.

Self-attention allows every token to communicate with every other token
in the sequence and update its representation accordingly. In the ViT context,
this means every patch token can directly attend to every other patch — global
receptive field from the very first layer, unlike convolutions which are local.

The full operation:

  Q = X W_Q, K = X W_K,  V = X W_V (linear projections)

  H_i = softmax( Q_i K_i^T / sqrt(d_k) ) V_i (scaled dot-product, per head)

  multihead(Q, K, V) = concat(H_1, ..., H_h) W_O

In practice, the three projections W_Q, W_K, W_V are fused into a single
nn.Linear(d_model, 3 * d_model) and split afterward — one GPU call instead
of three, mathematically identical.

Architecture defaults (ViT-Tiny):
  d_model = 192 (token dimension throughout the Transformer)
  num_heads = 8 (parallel attention subspaces)
  d_k = 24 (d_model / num_heads — dimension per head)

See: lec7 pages 19 to 35
See: https://medium.com/@vamsikd219/understanding-transformers-and-multi-head-attention-with-pytorch-008715e6cf88
see: https://medium.com/@heyamit10/implement-self-attention-and-cross-attention-in-pytorch-cfe17ab0b3ee
See: https://docs.pytorch.org/docs/2.11/generated/torch.nn.MultiheadAttention.html
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention module.

    Each token attends to every other token in the sequence simultaneously.
    8 parallel heads each learn a different type of relation between tokens.

    Input  : (batch_size, seq_len, d_model) -> (64, 197, 192)
    Output : (batch_size, seq_len, d_model) -> (64, 197, 192) — same shape as input

    Attributes:
        d_model : int — token dimension
        num_heads : int — number of parallel attention heads
        d_k : int — dimension per head (d_model // num_heads)
        scale : float — 1 / sqrt(d_k) — precomputed, avoids recomputing
        qkv_proj : nn.Linear — fused W_Q, W_K, W_V  (d_model -> 3*d_model)
        out_proj : nn.Linear — output projection W_O (d_model -> d_model)
        attn_drop : nn.Dropout — dropout on attention weights after softmax
    """

    def __init__(self, d_model: int = 192, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model   = d_model
        self.num_heads = num_heads
        self.d_k       = d_model // num_heads  # 192 // 8 = 24
        self.scale     = self.d_k ** -0.5      # 1 / sqrt(24) — stored, not recomputed

        self.qkv_proj = nn.Linear(d_model, 3 * d_model)  # fused W_Q, W_K, W_V
        self.out_proj  = nn.Linear(d_model, d_model)      # W_O
        self.attn_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Full attention pass from input token sequence to updated token sequence.
        Called automatically when you do attention(x).
        Every token queries all others, collects weighted value vectors,
        and returns an enriched sequence of the same shape.

        Step-by-step:
          1. qkv_proj : (B, 197, 192) -> (B, 197, 576)  fused Q K V projection
          2. reshape : (B, 197, 576) -> (B, 197, 8, 72) split into heads
          3. permute : (B, 197, 8, 72) -> (B, 8, 197, 72) heads before sequence
          4. chunk : 3 x (B, 8, 197, 24) separate Q, K, V
          5. scores : Q @ K^T / sqrt(24) -> (B, 8, 197, 197) similarity matrix
          6. softmax : (B, 8, 197, 197) attention weights
          7. dropout : stochastic regularization on weights
          8. AV : weights @ V -> (B, 8, 197, 24) weighted sum of values
          9. concat : (B, 197, 192) merge 8 heads
         10. out_proj : W_O -> (B, 197, 192) final projection

        Args:
            x : (batch_size, seq_len, d_model) — input token sequence

        Returns:
            (batch_size, seq_len, d_model) — updated token sequence
        """
        batch_size, seq_len, _ = x.shape

        # step 1-3 — fused QKV projection + reshape + permute
        qkv = self.qkv_proj(x)  # (B, seq_len, 3*d_model)
        qkv = qkv.reshape(batch_size, seq_len, self.num_heads, 3 * self.d_k)  # (B, seq_len, h, 3*d_k)
        qkv = qkv.permute(0, 2, 1, 3) # (B, h, seq_len, 3*d_k)

        # step 4 — split into Q, K, V
        q, k, v = qkv.chunk(3, dim=-1)  # 3 x (B, h, seq_len, d_k)

        # step 5-7 — scaled dot-product attention
        attn_scores  = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, h, seq_len, seq_len)
        attn_weights = F.softmax(attn_scores, dim=-1)# (B, h, seq_len, seq_len)
        attn_weights = self.attn_drop(attn_weights)

        # step 8-9 — weighted sum + merge heads
        attn_output = torch.matmul(attn_weights, v) # (B, h, seq_len, d_k)
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous() # (B, seq_len, h, d_k)
        attn_output = attn_output.view(batch_size, seq_len, self.d_model) # (B, seq_len, d_model)

        # step 10 — output projection
        return self.out_proj(attn_output) # (B, seq_len, d_model)