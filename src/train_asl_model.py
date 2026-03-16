#!/usr/bin/env python3
"""
Training script for ASL recognition model
Trains a model to recognize ASL signs from MediaPipe hand landmarks
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import mediapipe as mp
import os
import json
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm
import argparse

from constants import EXPECTED_FEATURE_SIZE


class ASLDataset(Dataset):
    """Dataset for ASL sign recognition"""
    
    def __init__(self, data_dir: str, class_mapping: Dict[str, int], 
                 feature_extractor, augment: bool = False):
        """
        Initialize ASL dataset
        
        Args:
            data_dir: Directory containing video files organized by sign class
                     Structure: data_dir/sign_name/video1.mp4, video2.mp4, ...
            class_mapping: Dictionary mapping sign names to class indices
            feature_extractor: Function to extract features from frames
            augment: Whether to apply data augmentation
        """
        self.data_dir = Path(data_dir)
        self.class_mapping = class_mapping
        self.feature_extractor = feature_extractor
        self.augment = augment
        
        # Load all video files and their labels
        self.samples = []
        for sign_name, class_idx in class_mapping.items():
            sign_dir = self.data_dir / sign_name
            if sign_dir.exists():
                for video_file in sign_dir.glob("*.mp4"):
                    self.samples.append((str(video_file), class_idx))
        
        print(f"Loaded {len(self.samples)} samples from {len(class_mapping)} classes")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        
        # Load video and extract features
        features = self._load_video_features(video_path)
        
        if features is None or len(features) == 0:
            # Return dummy features if video loading fails
            features = np.zeros((EXPECTED_FEATURE_SIZE,), dtype=np.float32)
        elif len(features) != EXPECTED_FEATURE_SIZE:
            if len(features) < EXPECTED_FEATURE_SIZE:
                padded = np.zeros((EXPECTED_FEATURE_SIZE,), dtype=np.float32)
                padded[: len(features)] = features
                features = padded
            else:
                features = features[:EXPECTED_FEATURE_SIZE]
        
        return torch.FloatTensor(features), torch.LongTensor([label])[0]
    
    def _load_video_features(self, video_path: str) -> Optional[np.ndarray]:
        """Load video and extract features from a representative frame"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        
        # Extract features from middle frame
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        middle_frame = total_frames // 2
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            return None
        
        # Extract features using the feature extractor
        features = self.feature_extractor(frame)
        return features


class ASLModel(nn.Module):
    """Neural network model for ASL sign recognition"""
    
    def __init__(self, input_size: int, num_classes: int, hidden_sizes: List[int] = [256, 128, 64]):
        """
        Initialize ASL model
        
        Args:
            input_size: Size of input feature vector
            num_classes: Number of ASL sign classes
            hidden_sizes: Sizes of hidden layers
        """
        super(ASLModel, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, num_classes))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


def extract_hand_features_mp(frame: np.ndarray, hands) -> np.ndarray:
    """
    Extract hand features from frame using MediaPipe
    This matches the feature extraction in asl_recognizer.py
    """
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    features = []
    
    if results.multi_hand_landmarks:
        # Process all detected hands
        for hand_landmarks in results.multi_hand_landmarks:
            # Extract normalized landmark coordinates
            for landmark in hand_landmarks.landmark:
                features.extend([landmark.x, landmark.y, landmark.z])
            
            # Calculate relative distances between key points
            key_points = [4, 8, 12, 16, 20]  # Fingertips
            
            if len(hand_landmarks.landmark) >= 21:
                for i in range(len(key_points)):
                    for j in range(i + 1, len(key_points)):
                        p1 = hand_landmarks.landmark[key_points[i]]
                        p2 = hand_landmarks.landmark[key_points[j]]
                        dist = np.sqrt(
                            (p1.x - p2.x)**2 + 
                            (p1.y - p2.y)**2 + 
                            (p1.z - p2.z)**2
                        )
                        features.append(dist)
    
    # If no hands detected or only one hand, pad/truncate to expected size
    # Expected: 2 hands * (21 landmarks * 3 coords + 10 distances) = 2 * 73 = 146
    # Or 1 hand: 73 features
    expected_size = 146  # For 2 hands
    if len(features) < expected_size:
        features.extend([0.0] * (expected_size - len(features)))
    elif len(features) > expected_size:
        features = features[:expected_size]
    
    return np.array(features, dtype=np.float32)


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
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
    accuracy = 100 * correct / total
    return avg_loss, accuracy


def validate(model, dataloader, criterion, device):
    """Validate the model"""
    model.eval()
    total_loss = 0
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
    accuracy = 100 * correct / total
    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser(description='Train ASL recognition model')
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Directory containing video files organized by sign class')
    parser.add_argument('--output-dir', type=str, default='models',
                       help='Directory to save trained model')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--train-split', type=float, default=0.8,
                       help='Fraction of data to use for training')
    parser.add_argument('--input-size', type=int, default=146,
                       help='Size of input feature vector (146 for 2 hands)')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize MediaPipe hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.5
    )
    
    # Create feature extractor
    def feature_extractor(frame):
        return extract_hand_features_mp(frame, hands)
    
    # Discover classes from directory structure
    data_path = Path(args.data_dir)
    class_names = sorted([d.name for d in data_path.iterdir() if d.is_dir()])
    class_mapping = {name: idx for idx, name in enumerate(class_names)}
    num_classes = len(class_mapping)
    
    print(f"Found {num_classes} sign classes: {class_names}")
    
    # Save class mapping
    mapping_path = Path(args.output_dir) / 'class_mapping.json'
    with open(mapping_path, 'w') as f:
        json.dump(class_mapping, f, indent=2)
    print(f"Saved class mapping to {mapping_path}")
    
    # Create dataset
    full_dataset = ASLDataset(args.data_dir, class_mapping, feature_extractor)
    
    # Split dataset
    train_size = int(args.train_split * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Create model
    model = ASLModel(
        input_size=args.input_size,
        num_classes=num_classes,
        hidden_sizes=[256, 128, 64]
    ).to(device)
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Training loop
    best_val_acc = 0.0
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        scheduler.step(val_loss)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model_path = Path(args.output_dir) / 'best_model.pt'
            torch.save(model.state_dict(), model_path)
            print(f"Saved best model (val acc: {val_acc:.2f}%) to {model_path}")
    
    # Save final model
    final_model_path = Path(args.output_dir) / 'final_model.pt'
    torch.save(model.state_dict(), final_model_path)
    print(f"\nTraining complete! Final model saved to {final_model_path}")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")


if __name__ == '__main__':
    main()

