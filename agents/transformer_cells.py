import torch
import torch.nn as nn
import torch.nn.functional as F

class GTrXLCell(nn.Module):
    """Drop-in replacement for nn.LSTMCell.
    forward(x_t, h_prev) -> (h_new, h_new)
    """
    def __init__(self, d_in, d_model, n_head=4, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(d_in, d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_head, batch_first=True, dropout=dropout
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )
        # Gating parameters (Parisotto 2019)
        self.gate_a = nn.Parameter(torch.zeros(d_model))
        self.gate_b = nn.Parameter(torch.ones(d_model))
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x_t, h_prev):
        """Single-step forward.

        Args:
            x_t (Tensor): (B, d_in)
            h_prev (Tensor): (B, d_model)
        """
        x = self.proj(x_t).unsqueeze(1)  # (B,1,d_model)
        q = k = v = self.ln1(x)
        ctx, _ = self.attn(q, k, v, need_weights=False)
        # Gated residual connection
        h_hat = torch.sigmoid(self.gate_a) * h_prev.unsqueeze(1) + \
                torch.sigmoid(self.gate_b) * ctx
        out = h_hat + self.ffn(self.ln2(h_hat))
        h_new = out.squeeze(1)
        return h_new, h_new  # second tensor kept for compatibility
