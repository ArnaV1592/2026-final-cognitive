
import torch
import torch.nn as nn
import torch.nn.functional as F

class CognitiveClassifier(nn.Module):
    def __init__(self, fusion, n_classes=2, dropout=0.2):
        super().__init__()
        self.fusion = fusion
        self.head   = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, n_classes),
        )

    def forward(self, wav_emb, xlm_emb, derived):
        return self.head(self.fusion(wav_emb, xlm_emb, derived))

class LabelSmoothingCE(nn.Module):
    def __init__(self, smoothing=0.1, weight=None):
        super().__init__()
        self.smoothing = smoothing
        self.weight    = weight

    def forward(self, logits, targets):
        n    = logits.size(-1)
        lp   = F.log_softmax(logits, dim=-1)
        with torch.no_grad():
            td = torch.zeros_like(lp).fill_(self.smoothing / (n - 1))
            td.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
        loss = -torch.sum(td * lp, dim=-1)
        if self.weight is not None:
            loss = loss * self.weight.to(logits.device)[targets]
        return loss.mean()
