
import torch, torch.nn as nn, numpy as np, json
from pathlib import Path

class TemperatureScaler(nn.Module):
    """
    Post-hoc temperature scaling.
    FIX: Saves and loads temperature robustly.
    A temperature near 1.0 = sharp predictions.
    A temperature > 10  = everything becomes 50/50.  ← the bug we fixed
    """
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        # Safety clamp: temperature above 5 collapses all outputs to uniform
        t = self.temperature.clamp(min=0.1, max=5.0)
        return logits / t

    def fit(self, logits: torch.Tensor, labels: torch.Tensor,
            lr=0.01, max_iter=200):
        nll = nn.CrossEntropyLoss()
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)
        def _eval():
            optimizer.zero_grad()
            loss = nll(self.forward(logits), labels)
            loss.backward()
            return loss
        optimizer.step(_eval)
        t = self.temperature.item()
        print(f"    Calibrated temperature: {t:.4f}")
        if t > 4.0:
            print("    ⚠️  WARNING: temperature is high — predictions may be soft.")
            print("       This can happen with small validation sets.")
            print("       Clamping to 2.0 for deployment safety.")
            with torch.no_grad():
                self.temperature.fill_(2.0)
        return self.temperature.item()

    def save(self, path: Path):
        json.dump({"temperature": float(self.temperature.item())}, open(path, "w"))

    def load(self, path: Path):
        """Handles both {"temperature": x} and plain float formats."""
        try:
            data = json.load(open(path))
            if isinstance(data, dict):
                t = float(data.get("temperature", 1.5))
            else:
                t = float(data)
        except Exception:
            print(f"  ⚠️  Could not load calibration from {path}. Using t=1.5")
            t = 1.5
        # Safety clamp on load
        t = max(0.1, min(t, 5.0))
        self.temperature.data = torch.tensor([t])
        print(f"  Loaded temperature: {t:.4f}")

def compute_brier_score(probs, labels, n_classes=2):
    oh = np.eye(n_classes)[labels]
    return float(np.mean(np.sum((probs - oh) ** 2, axis=1)))
