"""
Intern 2 — EchoNet-Dynamic & CAMUS frame extraction + PyTorch Dataset.
"""

import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd
from pathlib import Path


# ImageNet normalization (EfficientNet is ImageNet-pretrained)
ECHO_TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def ef_to_severity(ef: float) -> int:
    """Convert ejection fraction % → severity class using ASE guidelines."""
    if ef >= 50:
        return 0   # Normal
    elif ef >= 40:
        return 1   # Mild
    else:
        return 2   # Severe


def read_ed_es(video_path: Path, ed_idx: int, es_idx: int) -> torch.Tensor:
    """
    Extract end-diastole (ED) and end-systole (ES) frames from an MP4 echo video.
    Returns shape (2, 3, 224, 224).
    NOTE: OpenCV reads BGR — we convert to RGB before applying transforms.
    """
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    for idx in (ed_idx, es_idx):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            cap.release()
            raise RuntimeError(f"Could not read frame {idx} from {video_path}")
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)   # BGR → RGB is mandatory
        frames.append(ECHO_TRANSFORM(frame_rgb))
    cap.release()
    return torch.stack(frames)   # (2, 3, 224, 224)


class EchoNetDataset(Dataset):
    """
    PyTorch Dataset for EchoNet-Dynamic.
    Yields: (ed_es [2,3,224,224], ef [float], severity [int], patient_id [str])
    """

    def __init__(self, echonet_dir: str, split: str = "train"):
        self.echonet_dir = Path(echonet_dir)
        filelist = pd.read_csv(self.echonet_dir / "FileList.csv")
        self.df = filelist[filelist["Split"].str.lower() == split].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        video_path = self.echonet_dir / "Videos" / row["FileName"]
        ed_es = read_ed_es(video_path, int(row["ED"]), int(row["ES"]))
        ef = float(row["EF"])
        severity = ef_to_severity(ef)
        return ed_es, torch.tensor(ef, dtype=torch.float32), severity, row["FileName"]
