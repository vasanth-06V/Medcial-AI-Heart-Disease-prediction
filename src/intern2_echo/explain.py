"""
Intern 2 — Grad-CAM heatmaps + base_score.json export.
Week 4 deliverables.
"""

import json
import torch
import numpy as np
import cv2
from pathlib import Path


def generate_gradcam(model, ed_es_tensor: torch.Tensor, out_dir: Path, patient_id: str):
    """
    Generate Grad-CAM heatmap on the ED frame via pytorch-grad-cam library.
    Saves PNG to interim/heatmaps/{patient_id}_ed.png
    """
    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
    except ImportError:
        print("Install: pip install grad-cam")
        return {}

    target_layer = model.echo_branch.encoder.features[-1]
    cam = GradCAM(model=model, target_layers=[target_layer])

    # ED frame only for clarity
    ed_frame = ed_es_tensor[:, 0]                # (B, 3, 224, 224)
    grayscale_cam = cam(input_tensor=ed_frame[:1])
    img_np = ed_frame[0].permute(1, 2, 0).cpu().numpy()
    # Denormalize for visualization
    img_np = img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    img_np = np.clip(img_np, 0, 1).astype(np.float32)
    visualization = show_cam_on_image(img_np, grayscale_cam[0], use_rgb=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = str(out_dir / f"{patient_id}_ed.png")
    cv2.imwrite(out_path, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
    return {"echo_ed": out_path}


def export_base_score(
    patient_id: str,
    score: float,
    logits: torch.Tensor,
    ecg_features: dict,
    echo_features: dict,
    heatmap_paths: dict,
    out_dir: Path,
):
    """
    Write per-patient base_score.json.
    Intern 4 and Intern 5 consume these files.
    """
    softmax = torch.softmax(logits.squeeze(), dim=-1).tolist()
    pred_class = int(logits.argmax(-1).item())
    payload = {
        "patient_id": patient_id,
        "base_score": float(score),
        "class": pred_class,
        "class_label": ["Normal", "Mild", "Severe"][pred_class],
        "softmax": softmax,
        "features": {
            "ecg": ecg_features,    # e.g. {"qrs_duration_ms": {"value": 92, "attribution": 0.31}}
            "echo": echo_features,  # e.g. {"lvef_pct": {"value": 38, "attribution": 0.55}}
        },
        "heatmap_paths": heatmap_paths,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{patient_id}_base.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    return payload
