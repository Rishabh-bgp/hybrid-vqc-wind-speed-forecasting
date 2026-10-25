"""Train/evaluate HVQC. Example: python train.py --epochs 2   |   python train.py --csv my_scada.csv"""
import argparse, torch
from src.train_common import get_data, fit, evaluate
from src.hvqc import HVQC, freeze_quantum
ap = argparse.ArgumentParser()
ap.add_argument("--csv", default="data/synthetic/synthetic_scada.csv", help="SCADA-style CSV; pass \"\" to generate synthetic data in memory"); ap.add_argument("--n_synth", type=int, default=1500); ap.add_argument("--lookback", type=int, default=24)
ap.add_argument("--horizon", type=int, default=1); ap.add_argument("--epochs", type=int, default=6)
ap.add_argument("--qubits", type=int, default=4); ap.add_argument("--layers", type=int, default=2); ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--quick", action="store_true", help="fast smoke run (<1 min): last 500 rows, few epochs")
a = ap.parse_args()
if a.quick: a.epochs = 2
a.max_rows = 500 if a.quick else None
torch.manual_seed(a.seed)
tr, va, te, sc = get_data(a.csv, a.n_synth, a.lookback, a.horizon, a.max_rows)
m = HVQC(tr[0].shape[2], a.qubits, a.layers)
print("Phase 1: classical warm-up (quantum frozen)"); freeze_quantum(m, True); fit(m, tr, va, max(1, a.epochs // 2), lr=3e-3)
print("Phase 2: joint fine-tuning"); freeze_quantum(m, False); fit(m, tr, va, max(1, a.epochs - a.epochs // 2), lr=1e-3)
evaluate(m, te, sc, name="hvqc")
