import os, sys, torch
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from agents.transformer_cells import GTrXLCell

def test_gtrxl_shapes_and_params():
    cell = GTrXLCell(48, 64)
    B = 5
    x = torch.randn(B, 48, requires_grad=True)
    mem = torch.zeros(cell.mem_len, B, 64)
    y, new_mem = cell(x, mem)
    assert y.shape == (B, 64)
    assert new_mem.shape == (cell.mem_len, B, 64)
    # ensure GRU-style gating weights exist
    for attr in ["W_r", "U_r", "W_z", "U_z", "W_g", "U_g"]:
        assert hasattr(cell, attr)
    assert new_mem.requires_grad
