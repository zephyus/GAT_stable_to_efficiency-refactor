import torch
from agents.transformer_cells import GTrXLCell

def test_gtrxl_shapes_and_gates():
    cell = GTrXLCell(48, 64)
    B = 5
    x = torch.randn(B, 48)
    mem = torch.zeros(cell.mem_len, B, 64)
    y, new_mem = cell(x, mem)
    assert y.shape == (B, 64)
    assert new_mem.shape == (cell.mem_len, B, 64)
    # gate initialization values should yield near-zero/one after sigmoid
    gate_a_val = torch.sigmoid(cell.gate_a)
    gate_b_val = torch.sigmoid(cell.gate_b)
    assert torch.allclose(gate_a_val, torch.zeros_like(gate_a_val), atol=1e-4)
    assert torch.allclose(gate_b_val, torch.ones_like(gate_b_val), atol=1e-4)
