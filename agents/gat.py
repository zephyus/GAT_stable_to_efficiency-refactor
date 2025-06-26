import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphAttention(nn.Module):
    """
    Graph Attention Network layer implementation as per:
    https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, dropout=0.1, alpha=0.2):
        super(GraphAttention, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha

        # Linear transformation matrix W
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        
        # Attention mechanism parameters
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def _edge_scores(self, Wh):
        N = Wh.size(0)
        Wh1 = Wh.unsqueeze(1).expand(-1, N, -1)
        Wh2 = Wh.unsqueeze(0).expand(N, -1, -1)
        a_input = torch.cat([Wh1, Wh2], dim=-1)                # (N,N,2*C)
        scores = torch.matmul(a_input, self.a).squeeze(-1)      # (N,N)
        return self.leakyrelu(scores)

    def _masked_softmax(self, e, adj):
        e_masked = e.masked_fill(adj == 0, float('-inf'))
        e_masked = torch.clamp(e_masked, -20.0, 20.0)
        return F.softmax(e_masked, dim=1)

    def forward(self, h, adj):
        # h: (N,in), adj: (N,N) with self loops
        N = h.size(0)
        Wh = torch.mm(h, self.W)  # (N,C)

        with torch.no_grad():
            is_lone = (adj.sum(1) == 1)
            keep    = ~is_lone

        final_attn = torch.zeros_like(adj)
        if keep.any():
            Wh_k   = Wh[keep]
            adj_k  = adj[keep][:, keep]
            if Wh_k.size(0) > 0:
                e_k      = self._edge_scores(Wh_k)
                attn_k   = self._masked_softmax(e_k, adj_k)
                attn_drop= F.dropout(attn_k, self.dropout, training=self.training)
                h_prime  = Wh.clone()
                h_prime[keep] = torch.matmul(attn_drop, Wh_k)
                idx      = torch.where(keep)[0]
                final_attn[idx[:,None], idx] = attn_k
                idx_lone = torch.where(is_lone)[0]
                final_attn[idx_lone, idx_lone] = 1.0
            else:
                h_prime = Wh
                final_attn = torch.eye(N, device=h.device, dtype=h.dtype)
        else:
            e        = self._edge_scores(Wh)
            attn     = self._masked_softmax(e, adj)
            attn_drop= F.dropout(attn, self.dropout, training=self.training)
            h_prime  = torch.matmul(attn_drop, Wh)
            final_attn = attn

        return h_prime, final_attn
