#!/usr/bin/env python3
"""
Training script for ASL recognition model.
Supports both MLP (single-frame) and LSTM (sequence) model types.

Feature extraction is cached to disk on the first run so that MediaPipe
never re-processes the same video twice.  Subsequent epochs and re-runs
load instantly from the cache.

Cache layout:
  <cache_dir>/<video_stem>_f<seq_len>.npy   shape: (seq_len, EXPECTED_FEATURE_SIZE)
  <cache_dir>/<video_stem>_f1.npy           shape: (EXPECTED_FEATURE_SIZE,)  [MLP]
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, Optional

import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from constants import DEFAULT_SEQ_LEN, EXPECTED_FEATURE_SIZE
from model import ASLLSTMModel, ASLModel, AnyASLModel

logger = logging.getLogger(__name__)

_FINGERTIPS = [4, 8, 12, 16, 20]


# -- Feature extraction -------------------------------------------------------


def _extract_frame_features(frame: np.ndarray, hands) -> np.ndarray:
    """Extract a (EXPECTED_FEATURE_SIZE,) feature vector from one frame."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    features: list[float] = []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                features.extend([lm.x, lm.y, lm.z])
            lms = hand_landmarks.landmark
            for i in range(len(_FINGERTIPS)):
                for j in range(i + 1, len(_FINGERTIPS)):
                    a, b = lms[_FINGERTIPS[i]], lms[_FINGERTIPS[j]]
                    features.append(
                        float(
                            np.sqrt(
                                (a.x - b.x) ** 2
                                + (a.y - b.y) ** 2
                                + (a.z - b.z) ** 2
                            )
                        )
                    )

    if len(features) < EXPECTED_FEATURE_SIZE:
        features.extend([0.0] * (EXPECTED_FEATURE_SIZE - len(features)))
    elif len(features) > EXPECTED_FEATURE_SIZE:
        features = features[:EXPECTED_FEATURE_SIZE]

    return np.array(features, dtype=np.float32)


def _extract_video_sequence(
    video_path: str, seq_len: int, hands
) -> np.ndarray:
    """
    Extract seq_len evenly-spaced frames from video_path.
    Returns shape (seq_len, EXPECTED_FEATURE_SIZE).
    Frames that fail to load are zero vectors.
    """
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    result = np.zeros((seq_len, EXPECTED_FEATURE_SIZE), dtype=np.float32)

    if not cap.isOpened() or total == 0:
        cap.release()
        return result

    indices = np.linspace(0, max(total - 1, 0), seq_len, dtype=int)
    for i, frame_idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ret, frame = cap.read()
        if ret:
            result[i] = _extract_frame_features(frame, hands)

    cap.release()
    return result


# -- Feature cache ------------------------------------------------------------


def build_cache(
    data_dir: Path,
    cache_dir: Path,
    seq_len: int,
    hands,
) -> None:
    """
    Extract features for every video not yet cached.
    MLP uses seq_len=1 (middle frame only).
    LSTM uses seq_len=N (evenly-spaced frames).
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    video_paths = list(data_dir.rglob("*.mp4"))
    missing = [
        p for p in video_paths
        if not (cache_dir / f"{p.stem}_f{seq_len}.npy").exists()
    ]

    if not missing:
        print(f"Cache up to date ({len(video_paths)} videos, seq_len={seq_len})")
        return

    print(f"Caching {len(missing)}/{len(video_paths)} videos (seq_len={seq_len})...")
    for video_path in tqdm(missing, desc="Extracting features"):
        seq = _extract_video_sequence(str(video_path), seq_len, hands)
        np.save(cache_dir / f"{video_path.stem}_f{seq_len}.npy", seq)


# -- Datasets -----------------------------------------------------------------


class ASLDataset(Dataset):
    """
    Loads pre-cached feature arrays for MLP training (seq_len=1).
    Each sample is a flat (EXPECTED_FEATURE_SIZE,) vector.
    """

    def __init__(
        self,
        data_dir: Path,
        cache_dir: Path,
        class_mapping: Dict[str, int],
        seq_len: int = 1,
    ):
        self.cache_dir = cache_dir
        self.seq_len = seq_len
        self.samples: list[tuple[str, int]] = []

        for sign_name, class_idx in class_mapping.items():
            sign_dir = data_dir / sign_name
            if sign_dir.exists():
                for vf in sign_dir.glob("*.mp4"):
                    cache_file = cache_dir / f"{vf.stem}_f{seq_len}.npy"
                    if cache_file.exists():
                        self.samples.append((vf.stem, class_idx))

        print(f"Dataset: {len(self.samples)} cached samples from {len(class_mapping)} classes")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        stem, label = self.samples[idx]
        seq = np.load(self.cache_dir / f"{stem}_f{self.seq_len}.npy")
        # MLP: squeeze to (EXPECTED_FEATURE_SIZE,)
        features = seq[self.seq_len // 2] if self.seq_len > 1 else seq[0]
        return torch.from_numpy(features), torch.tensor(label, dtype=torch.long)


class ASLSequenceDataset(Dataset):
    """
    Loads pre-cached feature sequences for LSTM training.
    Each sample is a (seq_len, EXPECTED_FEATURE_SIZE) tensor.
    """

    def __init__(
        self,
        data_dir: Path,
        cache_dir: Path,
        class_mapping: Dict[str, int],
        seq_len: int = DEFAULT_SEQ_LEN,
    ):
        self.cache_dir = cache_dir
        self.seq_len = seq_len
        self.samples: list[tuple[str, int]] = []

        for sign_name, class_idx in class_mapping.items():
            sign_dir = data_dir / sign_name
            if sign_dir.exists():
                for vf in sign_dir.glob("*.mp4"):
                    cache_file = cache_dir / f"{vf.stem}_f{seq_len}.npy"
                    if cache_file.exists():
                        self.samples.append((vf.stem, class_idx))

        print(f"Sequence dataset: {len(self.samples)} cached samples "
              f"(seq_len={seq_len}) from {len(class_mapping)} classes")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        stem, label = self.samples[idx]
        seq = np.load(self.cache_dir / f"{stem}_f{self.seq_len}.npy")
        return torch.from_numpy(seq), torch.tensor(label, dtype=torch.long)


# -- Training loop ------------------------------------------------------------


def train_epoch(
    model: AnyASLModel,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for features, labels in tqdm(dataloader, desc="Training"):
        features = features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += int((predicted == labels).sum().item())

    return total_loss / len(dataloader), 100.0 * correct / total


def validate(
    model: AnyASLModel,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for features, labels in tqdm(dataloader, desc="Validating"):
            features = features.to(device)
            labels = labels.to(device)

            outputs = model(features)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += int((predicted == labels).sum().item())

    return total_loss / len(dataloader), 100.0 * correct / total


# -- Entry point --------------------------------------------------------------


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Train ASL recognition model")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Directory containing videos organized by sign class")
    parser.add_argument("--output-dir", type=str, default="models",
                        help="Directory to save trained model")
    parser.add_argument("--cache-dir", type=str, default="dataset/wlasl_cache",
                        help="Directory for cached feature arrays")
    parser.add_argument("--model-type", choices=["mlp", "lstm"], default="lstm",
                        help="Model architecture (default: lstm)")
    parser.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN,
                        help=f"Frames per video for LSTM (default: {DEFAULT_SEQ_LEN})")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--train-split", type=float, default=0.8)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Model type: {args.model_type.upper()}")

    os.makedirs(args.output_dir, exist_ok=True)
    cache_dir = Path(args.cache_dir)
    data_path = Path(args.data_dir)

    # Discover classes
    class_names = sorted([d.name for d in data_path.iterdir() if d.is_dir()])
    class_mapping = {name: idx for idx, name in enumerate(class_names)}
    num_classes = len(class_mapping)
    print(f"Found {num_classes} sign classes")

    mapping_path = Path(args.output_dir) / "class_mapping.json"
    with open(mapping_path, "w") as f:
        json.dump(class_mapping, f, indent=2)
    print(f"Saved class mapping to {mapping_path}")

    # Build feature cache (skips videos already cached)
    seq_len = args.seq_len if args.model_type == "lstm" else 1
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.5,
    )
    build_cache(data_path, cache_dir, seq_len, hands)
    hands.close()

    # Build dataset
    if args.model_type == "lstm":
        full_dataset: Dataset = ASLSequenceDataset(
            data_path, cache_dir, class_mapping, seq_len
        )
    else:
        full_dataset = ASLDataset(data_path, cache_dir, class_mapping, seq_len=1)

    train_size = int(args.train_split * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=0)

    # Build model
    model: AnyASLModel
    if args.model_type == "lstm":
        model = ASLLSTMModel(
            input_size=EXPECTED_FEATURE_SIZE,
            num_classes=num_classes,
        ).to(device)
    else:
        model = ASLModel(
            input_size=EXPECTED_FEATURE_SIZE,
            num_classes=num_classes,
        ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    best_val_acc = 0.0
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss:   {val_loss:.4f}, Val Acc:   {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_path = Path(args.output_dir) / "best_model.pt"
            torch.save(model.state_dict(), best_path)
            print(f"Saved best model ({val_acc:.2f}%) to {best_path}")

    final_path = Path(args.output_dir) / "final_model.pt"
    torch.save(model.state_dict(), final_path)
    print(f"\nTraining complete. Best val acc: {best_val_acc:.2f}%")


if __name__ == "__main__":
    main()
