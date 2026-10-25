"""HVQC: Conv1D-LSTM Classical Encoding Module -> VQC (angle enc., RY/RZ, circular CNOT) -> Classical Regression Head."""
import torch, torch.nn as nn
from .qlayer import QuantumBlock

class HVQC(nn.Module):
    def __init__(self, n_feat=5, n_qubits=4, n_layers=2, hidden=32):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(n_feat, hidden, 3, padding=1), nn.ReLU())
        self.lstm = nn.LSTM(hidden, hidden, batch_first=True)
        self.q = QuantumBlock(hidden, n_qubits, n_layers)
        self.head = nn.Sequential(nn.Linear(n_qubits, 16), nn.ReLU(), nn.Linear(16, 1))
    def forward(self, x):
        h = self.conv(x.transpose(1, 2)).transpose(1, 2)
        _, (h, _) = self.lstm(h)
        return self.head(self.q(h[-1])).squeeze(-1)

def freeze_quantum(model, frozen=True):
    """Two-phase strategy: phase 1 trains classical parts, phase 2 fine-tunes all."""
    for p in model.q.q.parameters(): p.requires_grad = not frozen
