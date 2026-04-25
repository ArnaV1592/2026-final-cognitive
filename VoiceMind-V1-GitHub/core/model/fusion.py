
import torch
import torch.nn as nn

class GatedFusion(nn.Module):
    def __init__(self, wavlm_dim=1024, xlmr_dim=768, derived_dim=28, fused_dim=512):
        super().__init__()
        total      = wavlm_dim + xlmr_dim + derived_dim
        self.proj  = nn.Linear(total, fused_dim)
        self.gate  = nn.Sequential(nn.Linear(total, fused_dim), nn.Sigmoid())
        self.norm  = nn.LayerNorm(fused_dim)

    def forward(self, wav_emb, xlm_emb, derived):
        x = torch.cat([wav_emb, xlm_emb, derived], dim=-1)
        return self.norm(self.proj(x) * self.gate(x))
