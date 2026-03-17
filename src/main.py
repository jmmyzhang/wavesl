#!/usr/bin/env python3
"""
WaveSL - Real-time ASL to Speech Translation
Phase 2, Step 5: Model inference with confidence threshold
"""

import argparse
import json
import os
from pathlib import Path
from typing import Optional

# Suppress verbose MediaPipe/TensorFlow Lite internal warnings
os.environ.setdefault('GLOG_minloglevel', '2')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn

from constants import EXPECTED_FEATURE_SIZE
from device_selector import select_camera

_FINGERTIPS = [4, 8, 12, 16, 20]
CONFIDENCE_THRESHOLD = 0.6


# ── Model ─────────────────────────────────────────────────────────────────────

class ASLModel(nn.Module):
    def __init__(self, input_size: int, num_classes: int,
                 hidden_sizes: list[int] = [256, 128, 64]):
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_size
        for h in hidden_sizes:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(0.3)]
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def load_model(model_path: str) -> tuple[ASLModel, dict[int, str], torch.device]:
    """
    Load a trained ASLModel from a .pt state-dict file.
    Looks for class_mapping.json in the same directory.
    Returns (model, reverse_mapping, device).
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    path = Path(model_path)

    state_dict = torch.load(str(path), map_location=device, weights_only=True)

    # Infer architecture from state dict
    first_key = next(k for k in state_dict if 'weight' in k)
    last_key = next(k for k in reversed(list(state_dict.keys())) if 'weight' in k)
    input_size = state_dict[first_key].shape[1]
    num_classes = state_dict[last_key].shape[0]

    model = ASLModel(input_size=input_size, num_classes=num_classes)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Load class mapping
    mapping_path = path.parent / 'class_mapping.json'
    reverse_mapping: dict[int, str] = {}
    if mapping_path.exists():
        with open(mapping_path) as f:
            class_mapping: dict[str, int] = json.load(f)
        reverse_mapping = {idx: name for name, idx in class_mapping.items()}
        print(f'Loaded model: {num_classes} classes from {path}')
    else:
        print(f'Loaded model: {num_classes} classes (no class_mapping.json found)')

    return model, reverse_mapping, device


# ── Feature extraction ────────────────────────────────────────────────────────

def extract_features(hand_landmarks_list: list) -> np.ndarray:
    features: list[float] = []
    for hand_landmarks in hand_landmarks_list:
        for lm in hand_landmarks.landmark:
            features.extend([lm.x, lm.y, lm.z])
        lms = hand_landmarks.landmark
        for i in range(len(_FINGERTIPS)):
            for j in range(i + 1, len(_FINGERTIPS)):
                a, b = lms[_FINGERTIPS[i]], lms[_FINGERTIPS[j]]
                features.append(float(
                    np.sqrt((a.x-b.x)**2 + (a.y-b.y)**2 + (a.z-b.z)**2)
                ))
    vec = np.array(features, dtype=np.float32)
    if len(vec) < EXPECTED_FEATURE_SIZE:
        vec = np.pad(vec, (0, EXPECTED_FEATURE_SIZE - len(vec)))
    elif len(vec) > EXPECTED_FEATURE_SIZE:
        vec = vec[:EXPECTED_FEATURE_SIZE]
    return vec


# ── Inference ─────────────────────────────────────────────────────────────────

def infer(model: ASLModel, features: np.ndarray,
          reverse_mapping: dict[int, str],
          device: torch.device) -> Optional[tuple[str, float]]:
    """
    Run one inference pass.
    Returns (label, confidence) if confidence >= threshold, else None.
    """
    tensor = torch.from_numpy(features).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)
        confidence, class_idx = torch.max(probs, dim=1)

    conf = confidence.item()
    if conf < CONFIDENCE_THRESHOLD:
        return None

    label = reverse_mapping.get(class_idx.item(), str(class_idx.item()))
    return label, conf


# ── Main loop ─────────────────────────────────────────────────────────────────

def run(camera_index: int, model_path: Optional[str]) -> None:
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f'Could not open camera {camera_index}')

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    model, reverse_mapping, device = (None, {}, torch.device('cpu'))
    if model_path:
        model, reverse_mapping, device = load_model(model_path)
    else:
        print('No model provided — running feature extraction only')

    print('Camera feed open — press q to quit')

    while True:
        ret, frame = cap.read()
        if not ret:
            print('Warning: failed to read frame')
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style(),
                )

            features = extract_features(results.multi_hand_landmarks)

            if model is not None:
                result = infer(model, features, reverse_mapping, device)
                if result:
                    label, conf = result
                    print(f'\r{label}  ({conf:.0%})', end='', flush=True)
                else:
                    print(f'\r{"":30}', end='', flush=True)

        cv2.imshow('WaveSL', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print()
    hands.close()
    cap.release()
    cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description='WaveSL - ASL to Speech')
    parser.add_argument('--camera', type=int, default=None,
                        help='Camera index (skips interactive selection)')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to trained .pt model file')
    args = parser.parse_args()

    camera_index = args.camera if args.camera is not None else select_camera()
    run(camera_index, args.model)


if __name__ == '__main__':
    main()
