import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import math

class GTrXLCell(nn.Module):
    """Transformer-style cell with memory.

    forward(x_t, mem_prev) -> (h_t, mem_next)
    """
    def __init__(self, d_input, d_model, n_head=4, mem_len=16, dropout=0.1, bias=True):
        super().__init__()
        # Store parameters for future use
        self.d_input = d_input
        self.d_model = d_model
        self.n_head = n_head
        assert d_model % n_head == 0, "d_model not divisible by n_head"
        self.mem_len = mem_len
        self.dropout = dropout
        self.bias = bias
        
        self.proj = nn.Linear(d_input, d_model, bias=bias)
        self.attn = nn.MultiheadAttention(
            d_model, n_head, dropout=dropout, bias=bias, batch_first=False
        )
        self.head_dim = d_model // n_head
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model, bias=bias),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model, bias=bias),
        )
        # GRU-style gating parameters
        self.W_r = nn.Linear(d_model, d_model, bias=bias)
        self.U_r = nn.Linear(d_model, d_model, bias=bias)
        self.W_z = nn.Linear(d_model, d_model, bias=bias)
        self.U_z = nn.Linear(d_model, d_model, bias=bias)
        self.W_g = nn.Linear(d_model, d_model, bias=bias)
        self.U_g = nn.Linear(d_model, d_model, bias=bias)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x_t, mem_prev):
        """GTrXL forward with memory tensor.

        Args:
            x_t (Tensor): (B, d_input) - current input
            mem_prev (Tensor): (mem_len, B, d_model) - previous memory

        Returns:
            out (Tensor): (B, d_model) - current output  
            mem_next (Tensor): (mem_len, B, d_model) - updated memory
        """
        B = x_t.size(0)
        
        # Project input and add batch dim at sequence position
        x_proj = self.proj(x_t).unsqueeze(0)  # (1, B, d_model)
        
        # Concatenate memory with current input
        seq = torch.cat([mem_prev, x_proj], dim=0)  # (mem_len+1, B, d_model)
        
        # Self-attention over full sequence (mem + current)
        # MultiheadAttention expects (S, N, E) when batch_first=False
        seq_ln = self.ln1(seq)  # (mem_len+1, B, d_model)
        if torch.isnan(seq_ln).any() or torch.isinf(seq_ln).any():
            raise RuntimeError(
                "NaN/Inf DETECTED: Input to Attention block is invalid.")

        # Built-in multi-head self-attention with clamped weights
        ctx, attn_w = self.attn(seq_ln, seq_ln, seq_ln, need_weights=True)
        attn_w.clamp_(0.0, 1.0)
        if torch.isnan(ctx).any() or torch.isinf(ctx).any():
            raise RuntimeError(
                "NaN/Inf DETECTED: Output of Attention block is invalid.")
        
        # Take the last timestep as current context
        ctx_t = ctx[-1]  # (B, d_model)
        
        # Apply dropout
        ctx_t = F.dropout(ctx_t, p=self.dropout, training=self.training)
        
        # GRU-style gated residual connection
        h_prev = mem_prev[-1]  # (B, d_model) - last memory state
        r = torch.sigmoid(self.W_r(ctx_t) + self.U_r(h_prev))
        z = torch.sigmoid(self.W_z(ctx_t) + self.U_z(h_prev) - 2.0)
        h_hat = torch.tanh(self.W_g(ctx_t) + self.U_g(r * h_prev))
        h_t = (1 - z) * h_prev + z * h_hat
        
        # Feed-forward and residual with dropout
        ff_out = self.ffn(self.ln2(h_t))
        ff_out = F.dropout(ff_out, p=self.dropout, training=self.training)
        out = h_t + ff_out
        
        # Update memory: keep most recent mem_len timesteps
        mem_next = seq[-self.mem_len:]  # (mem_len, B, d_model)
        
        return out, mem_next
