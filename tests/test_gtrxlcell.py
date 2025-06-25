import torch
from agents.transformer_cells import GTrXLCell

def test_gtrxl_shapes():
    cell = GTrXLCell(48, 64)
    B = 5
    x = torch.randn(B, 48)
    h = torch.zeros(B, 64)
    y, s = cell(x, h)
    assert y.shape == (B, 64)
    assert torch.equal(y, s)
