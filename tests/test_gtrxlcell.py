import os, sys, torch
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from agents.transformer_cells import GTrXLCell

def test_gtrxl_shapes_and_gates():
    cell = GTrXLCell(48, 64)
    B = 5
    x = torch.randn(B, 48, requires_grad=True)
    mem = torch.zeros(cell.mem_len, B, 64)
    y, new_mem = cell(x, mem)
    assert y.shape == (B, 64)
    assert new_mem.shape == (cell.mem_len, B, 64)
    # gate initialization should be around 0.5 and 0.88 after sigmoid
    gate_a_val = torch.sigmoid(cell.gate_a)
    gate_b_val = torch.sigmoid(cell.gate_b)
    assert torch.allclose(gate_a_val, torch.full_like(gate_a_val, 0.5), atol=1e-4)
    assert torch.allclose(gate_b_val, torch.full_like(gate_b_val, 0.88), atol=1e-2)
    assert new_mem.requires_grad
