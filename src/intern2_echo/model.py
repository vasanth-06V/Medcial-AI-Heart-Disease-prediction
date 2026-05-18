"""
Intern 2 — Echo Encoder + Dual-Branch Base Model
Fuses ECG features (from Intern 1's ResNet1D) with Echo features (EfficientNet-B3)
via multi-head cross-attention.
"""

import torch
import torch.nn as nn
from torchvision import models
from pathlib import Path


class EchoEncoder(nn.Module):
    """EfficientNet-B3 backbone → 512-d projection. Operates on a single frame."""

    def __init__(self, pretrained: bool = True, freeze_backbone: bool = False):
        super().__init__()
        weights = models.EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.efficientnet_b3(weights=weights)
        self.features = backbone.features           # output: (B, 1536, 7, 7) for 224×224
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Sequential(
            nn.Linear(1536, 512), nn.ReLU(), nn.Dropout(0.3)
        )
        if freeze_backbone:
            for p in self.features.parameters():
                p.requires_grad = False

    def forward(self, x):                           # x: (B, 3, 224, 224)
        h = self.features(x)
        h = self.pool(h).flatten(1)                 # (B, 1536)
        return self.proj(h)                         # (B, 512)


class EchoBranch(nn.Module):
    """Combines ED and ES frames into one 512-d vector."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.encoder = EchoEncoder(pretrained=pretrained)
        self.merge = nn.Sequential(
            nn.Linear(1024, 512), nn.ReLU(), nn.Dropout(0.3)
        )

    def forward(self, ed_es):                       # ed_es: (B, 2, 3, 224, 224)
        B = ed_es.size(0)
        flat = ed_es.view(B * 2, 3, 224, 224)
        feat = self.encoder(flat).view(B, 2, 512)   # (B, 2, 512)
        return self.merge(feat.flatten(1))           # (B, 512)


class CardiacBaseModel(nn.Module):
    """
    Dual-branch model: ECG branch (Intern 1 ResNet1D) + Echo branch (EfficientNet-B3).
    Fused via multi-head attention.
    Outputs: (logits [B,3], score [B])
    """

    def __init__(self, ecg_encoder, echo_branch):
        super().__init__()
        self.ecg_encoder = ecg_encoder
        self.echo_branch = echo_branch
        self.attn = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(1024, 256), nn.ReLU(), nn.Dropout(0.3)
        )
        self.cls_head = nn.Linear(256, 3)
        self.score_head = nn.Linear(256, 1)

    def forward(self, ecg, ed_es, return_features: bool = False):
        _, e = self.ecg_encoder(ecg, return_features=True)  # (B, 512)
        v = self.echo_branch(ed_es)                          # (B, 512)
        seq = torch.stack([e, v], dim=1)                     # (B, 2, 512)
        attn_out, _ = self.attn(seq, seq, seq)               # (B, 2, 512)
        h = self.head(attn_out.flatten(1))                   # (B, 256)
        logits = self.cls_head(h)
        score = torch.sigmoid(self.score_head(h)).squeeze(-1)
        feat = attn_out.flatten(1)                           # (B, 1024) for Intern 4
        if return_features:
            return logits, score, feat
        return logits, score

    @classmethod
    def from_checkpoint(cls, ckpt_path: str, ecg_encoder, echo_branch):
        """Load a saved checkpoint."""
        model = cls(ecg_encoder, echo_branch)
        state = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(state["model_state_dict"])
        return model


def encode_base(model, ecg_tensor, ed_es_tensor, device="cpu"):
    """
    Convenience function for Intern 4.
    Returns (score, logits, base_features_dict)
    """
    model.eval()
    with torch.no_grad():
        ecg = ecg_tensor.to(device)
        ed_es = ed_es_tensor.to(device)
        logits, score, feat = model(ecg, ed_es, return_features=True)
    return (
        score.cpu(),
        logits.cpu(),
        {"base_feature_vector": feat.cpu().numpy().tolist()},
    )
