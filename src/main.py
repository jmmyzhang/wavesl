#!/usr/bin/env python3
"""
WaveSL - Real-time ASL to Speech Translation
Phase 2, Step 6: Temporal smoothing with majority-vote buffer
"""

import argparse
import contextlib
import os
from pathlib import Path
from typing import Optional


@contextlib.contextmanager
def _suppress_c_stderr():
    """Redirect fd 2 to /dev/null to silence C-level warnings (e.g. MediaPipe glog)."""
    devnull = os.open(os.devnull, os.O_WRONLY)
    saved = os.dup(2)
    os.dup2(devnull, 2)
    try:
        yield
    finally:
        os.dup2(saved, 2)
        os.close(devnull)
        os.close(saved)


import cv2
with _suppress_c_stderr():
    import mediapipe as mp
import numpy as np
import torch

from constants import EXPECTED_FEATURE_SIZE
from device_selector import select_camera
from model import ASLModel, load_model
from prediction_smoother import PredictionSmoother

_FINGERTIPS = [4, 8, 12, 16, 20]
CONFIDENCE_THRESHOLD = 0.6


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
          device: torch.device,
          threshold: float = CONFIDENCE_THRESHOLD) -> tuple[str, float, bool]:
    """
    Run one inference pass.
    Returns (label, confidence, above_threshold).
    Always returns the top prediction so callers can inspect raw output.
    """
    tensor = torch.from_numpy(features).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)
        confidence, class_idx = torch.max(probs, dim=1)

    conf = confidence.item()
    label = reverse_mapping.get(class_idx.item(), str(class_idx.item()))
    return label, conf, conf >= threshold


# ── Main loop ─────────────────────────────────────────────────────────────────

def run(camera_index: int, model_path: Optional[str],
        threshold: float = CONFIDENCE_THRESHOLD) -> None:
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

    smoother = PredictionSmoother()
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
                label, conf, above = infer(model, features, reverse_mapping, device, threshold)
                marker = '✓' if above else '?'

                smoothed = smoother.update(label, conf) if above else None
                stable = f'  [{smoothed.label}]{"*" if smoothed.is_new else " "}' if smoothed else ''
                print(f'\r{marker} {label:<12} {conf:.0%}{stable}   ', end='', flush=True)
        else:
            smoother.clear()

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
    parser.add_argument('--threshold', type=float, default=CONFIDENCE_THRESHOLD,
                        help=f'Confidence threshold (default: {CONFIDENCE_THRESHOLD})')
    args = parser.parse_args()

    model_path = args.model
    if model_path is None:
        default = Path(__file__).parent.parent / 'models' / 'wlasl' / 'best_model.pt'
        if default.exists():
            model_path = str(default)
        else:
            print(f'Warning: no model found at {default}')
            print('Run train_wlasl_model.sh to train one, or pass --model <path>')

    camera_index = args.camera if args.camera is not None else select_camera()
    run(camera_index, model_path, args.threshold)


if __name__ == '__main__':
    main()
