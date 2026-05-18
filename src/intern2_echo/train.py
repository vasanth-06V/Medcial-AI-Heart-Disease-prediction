"""
Intern 2 — Training script for:
  1. EfficientNet-B3 EF regression warm-up (Week 2)
  2. Dual-branch CardiacBaseModel fusion training (Week 3)
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import yaml
from pathlib import Path
from tqdm import tqdm


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ── Week 2: EF Regression Warm-up ─────────────────────────────────────────────

def train_ef_regression(encoder, train_loader, val_loader, cfg: dict):
    """
    Warm-up: regress EF from a single ED frame.
    Loss: MAE (L1). Target: EF / 100 (normalized to [0,1]).
    Quality bar: MAE <= 6 pp on val set.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Replace projection with regression head for warm-up
    reg_head = nn.Linear(512, 1).to(device)
    encoder = encoder.to(device)

    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(reg_head.parameters()),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    criterion = nn.L1Loss()

    best_mae = float("inf")
    for epoch in range(20):
        encoder.train(); reg_head.train()
        for ed_es, ef, _, _ in tqdm(train_loader, desc=f"WarmUp Epoch {epoch+1}"):
            ed_frame = ed_es[:, 0].to(device)    # ED frame only
            ef_norm = (ef / 100.0).to(device)
            feat = encoder(ed_frame)
            pred = reg_head(feat).squeeze(-1)
            loss = criterion(pred, ef_norm)
            optimizer.zero_grad(); loss.backward(); optimizer.step()

        # Validation
        encoder.eval(); reg_head.eval()
        maes = []
        with torch.no_grad():
            for ed_es, ef, _, _ in val_loader:
                pred = reg_head(encoder(ed_es[:, 0].to(device))).squeeze(-1)
                maes.append((pred.cpu() * 100 - ef).abs().mean().item())
        val_mae = np.mean(maes)
        print(f"  Val MAE: {val_mae:.2f} pp")
        if val_mae < best_mae:
            best_mae = val_mae
            torch.save(encoder.state_dict(), "models/echo_encoder_warmup.pt")

    print(f"Best Val MAE: {best_mae:.2f} pp (target ≤ 6 pp)")
    return encoder


# ── Week 3: Dual-Branch Fusion Training ───────────────────────────────────────

def train_base_model(model, train_loader, val_loader, y_train, cfg: dict):
    """
    Train CardiacBaseModel. Combined loss: CE + 0.5 * MSE on normalized EF.
    Quality bar: Balanced accuracy >= 80%.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    weights = compute_class_weight("balanced", classes=[0, 1, 2], y=y_train)
    criterion_ce = nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32, device=device)
    )
    criterion_mse = nn.MSELoss()

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    n_epochs = cfg["training"]["num_epochs"]
    scheduler = OneCycleLR(
        optimizer, max_lr=cfg["training"]["learning_rate"],
        epochs=n_epochs, steps_per_epoch=len(train_loader)
    )

    best_acc = 0.0
    for epoch in range(n_epochs):
        model.train()
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs}"):
            ecg, ed_es, ef, severity = [b.to(device) for b in batch[:4]]
            logits, score = model(ecg, ed_es)
            ef_norm = (ef / 100.0)
            loss = criterion_ce(logits, severity) + \
                   cfg["training"]["loss_mse_weight"] * criterion_mse(score, ef_norm)
            optimizer.zero_grad(); loss.backward()
            optimizer.step(); scheduler.step()

        # Validation
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                ecg, ed_es, _, severity = [b.to(device) for b in batch[:4]]
                logits, _ = model(ecg, ed_es)
                all_preds.extend(logits.argmax(-1).cpu().tolist())
                all_labels.extend(severity.cpu().tolist())

        from sklearn.metrics import balanced_accuracy_score
        bal_acc = balanced_accuracy_score(all_labels, all_preds)
        print(f"  Val Balanced Acc: {bal_acc:.3f}")
        if bal_acc > best_acc:
            best_acc = bal_acc
            torch.save({"model_state_dict": model.state_dict(),
                        "epoch": epoch, "bal_acc": bal_acc},
                       cfg["paths"]["base_model"])

    print(f"Best Balanced Accuracy: {best_acc:.3f} (target >= 0.80)")
    return model
