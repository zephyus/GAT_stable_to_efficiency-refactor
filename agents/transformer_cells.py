import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class GTrXLCell(nn.Module):
    """Drop-in replacement for nn.LSTMCell.
    forward(x_t, h_prev) -> (h_new, h_new)
    """
    def __init__(self, d_input, d_model, n_head=4, mem_len=16, dropout=0.1, bias=True):
        super().__init__()
        # Store parameters for future use
        self.d_input = d_input
        self.d_model = d_model
        self.n_head = n_head
        self.mem_len = mem_len
        self.dropout = dropout
        self.bias = bias
        
        self.proj = nn.Linear(d_input, d_model, bias=bias)
        self.attn = nn.MultiheadAttention(
            d_model, n_head, batch_first=False, dropout=dropout, bias=bias
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model, bias=bias),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model, bias=bias),
        )
        # Gating parameters (Parisotto 2019)
        # Initialize so sigmoid(gate_a)≈0 and sigmoid(gate_b)≈1
        self.gate_a = nn.Parameter(torch.full((d_model,), -10.0))
        self.gate_b = nn.Parameter(torch.full((d_model,), 10.0))
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
        mem_cat = torch.cat([mem_prev.clone(), x_proj], dim=0)
        seq = mem_cat  # (mem_len+1, B, d_model)
        
        # Self-attention over full sequence (mem + current)
        # MultiheadAttention expects (S, N, E) when batch_first=False
        seq_ln = self.ln1(seq)  # (mem_len+1, B, d_model)
        ctx, _ = self.attn(seq_ln, seq_ln, seq_ln, need_weights=False)  # (mem_len+1, B, d_model)
        
        # Take the last timestep as current context
        ctx_t = ctx[-1]  # (B, d_model)
        
        # Apply dropout
        ctx_t = F.dropout(ctx_t, p=self.dropout, training=self.training)
        
        # Gated residual connection (Parisotto 2019 style)
        gate_a_val = torch.sigmoid(self.gate_a)
        gate_b_val = torch.sigmoid(self.gate_b)
        
        prev_h = mem_prev[-1]  # (B, d_model) - last memory state
        h_hat = gate_a_val * prev_h + gate_b_val * ctx_t
        
        # Feed-forward and residual with dropout
        ff_out = self.ffn(self.ln2(h_hat))
        ff_out = F.dropout(ff_out, p=self.dropout, training=self.training)
        out = h_hat + ff_out
        
        # Update memory: keep most recent mem_len timesteps
        mem_next = mem_cat[-self.mem_len:]  # (mem_len, B, d_model)
        
        return out, mem_next
