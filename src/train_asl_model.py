#!/usr/bin/env python3
"""
Training script for ASL recognition model.
Trains a model to recognize ASL signs from MediaPipe hand landmarks.
"""

import argparse
import json
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

from constants import EXPECTED_FEATURE_SIZE
from model import ASLModel


class ASLDataset(Dataset):
    """Dataset for ASL sign recognition."""

    def __init__(
        self,
        data_dir: str,
        class_mapping: Dict[str, int],
        feature_extractor,
        augment: bool = False,
    ):
        self.data_dir = Path(data_dir)
        self.class_mapping = class_mapping
        self.feature_extractor = feature_extractor
        self.augment = augment

        self.samples: list[tuple[str, int]] = []
        for sign_name, class_idx in class_mapping.items():
            sign_dir = self.data_dir / sign_name
            if sign_dir.exists():
                for video_file in sign_dir.glob("*.mp4"):
                    self.samples.append((str(video_file), class_idx))

        print(f"Loaded {len(self.samples)} samples from {len(class_mapping)} classes")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        video_path, label = self.samples[idx]

        features = self._load_video_features(video_path)

        if features is None or len(features) == 0:
            features = np.zeros((EXPECTED_FEATURE_SIZE,), dtype=np.float32)
        elif len(features) < EXPECTED_FEATURE_SIZE:
            padded = np.zeros((EXPECTED_FEATURE_SIZE,), dtype=np.float32)
            padded[: len(features)] = features
            features = padded
        elif len(features) > EXPECTED_FEATURE_SIZE:
            features = features[:EXPECTED_FEATURE_SIZE]

        return torch.FloatTensor(features), torch.LongTensor([label])[0]

    def _load_video_features(self, video_path: str) -> Optional[np.ndarray]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        middle_frame = total_frames // 2

        cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            return None

        return self.feature_extractor(frame)


def extract_hand_features_mp(frame: np.ndarray, hands) -> np.ndarray:
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    features: list[float] = []

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for landmark in hand_landmarks.landmark:
                features.extend([landmark.x, landmark.y, landmark.z])

            key_points = [4, 8, 12, 16, 20]

            if len(hand_landmarks.landmark) >= 21:
                for i in range(len(key_points)):
                    for j in range(i + 1, len(key_points)):
                        p1 = hand_landmarks.landmark[key_points[i]]
                        p2 = hand_landmarks.landmark[key_points[j]]
                        dist = float(
                            np.sqrt(
                                (p1.x - p2.x) ** 2
                                + (p1.y - p2.y) ** 2
                                + (p1.z - p2.z) ** 2
                            )
                        )
                        features.append(dist)

    if len(features) < EXPECTED_FEATURE_SIZE:
        features.extend([0.0] * (EXPECTED_FEATURE_SIZE - len(features)))
    elif len(features) > EXPECTED_FEATURE_SIZE:
        features = features[:EXPECTED_FEATURE_SIZE]

    return np.array(features, dtype=np.float32)


def train_epoch(
    model: ASLModel,
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
        correct += (predicted == labels).sum().item()

    avg_loss = total_loss / len(dataloader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def validate(
    model: ASLModel,
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
            correct += (predicted == labels).sum().item()

    avg_loss = total_loss / len(dataloader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ASL recognition model")
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory containing video files organized by sign class",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="Directory to save trained model",
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size for training"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=0.001, help="Learning rate"
    )
    parser.add_argument(
        "--train-split",
        type=float,
        default=0.8,
        help="Fraction of data to use for training",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=EXPECTED_FEATURE_SIZE,
        help=f"Size of input feature vector (default: {EXPECTED_FEATURE_SIZE})",
    )

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.5,
    )

    def feature_extractor(frame: np.ndarray) -> np.ndarray:
        return extract_hand_features_mp(frame, hands)

    data_path = Path(args.data_dir)
    class_names = sorted([d.name for d in data_path.iterdir() if d.is_dir()])
    class_mapping = {name: idx for idx, name in enumerate(class_names)}
    num_classes = len(class_mapping)

    print(f"Found {num_classes} sign classes: {class_names}")

    mapping_path = Path(args.output_dir) / "class_mapping.json"
    with open(mapping_path, "w") as f:
        json.dump(class_mapping, f, indent=2)
    print(f"Saved class mapping to {mapping_path}")

    full_dataset = ASLDataset(args.data_dir, class_mapping, feature_extractor)

    train_size = int(args.train_split * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    model = ASLModel(
        input_size=args.input_size,
        num_classes=num_classes,
        hidden_sizes=[256, 128, 64],
    ).to(device)

    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    best_val_acc = 0.0

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        scheduler.step(val_loss)

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model_path = Path(args.output_dir) / "best_model.pt"
            torch.save(model.state_dict(), model_path)
            print(f"Saved best model (val acc: {val_acc:.2f}%) to {model_path}")

    final_model_path = Path(args.output_dir) / "final_model.pt"
    torch.save(model.state_dict(), final_model_path)
    print(f"\nTraining complete! Final model saved to {final_model_path}")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")


if __name__ == "__main__":
    main()
