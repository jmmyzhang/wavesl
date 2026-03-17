#!/usr/bin/env python3
"""
WaveSL - Real-time ASL to Speech Translation
Phase 2, Step 4: Feature extraction from MediaPipe hand landmarks
"""

import argparse

import cv2
import mediapipe as mp
import numpy as np

from constants import EXPECTED_FEATURE_SIZE
from device_selector import select_camera

# Landmark indices for fingertips: thumb, index, middle, ring, pinky
_FINGERTIPS = [4, 8, 12, 16, 20]


def extract_features(hand_landmarks_list: list) -> np.ndarray:
    """
    Extract a fixed-size feature vector from detected hand landmarks.

    Per hand: 21 landmarks × 3 coords (x, y, z) = 63 values
              10 inter-fingertip distances              = 10 values
              total per hand                           = 73 values

    For 2 hands: 146 values (EXPECTED_FEATURE_SIZE).
    If only 1 hand is detected the vector is zero-padded to 146.
    """
    features: list[float] = []

    for hand_landmarks in hand_landmarks_list:
        # Normalised landmark coordinates
        for lm in hand_landmarks.landmark:
            features.extend([lm.x, lm.y, lm.z])

        # Euclidean distances between every pair of fingertips (10 pairs)
        lms = hand_landmarks.landmark
        for i in range(len(_FINGERTIPS)):
            for j in range(i + 1, len(_FINGERTIPS)):
                a, b = lms[_FINGERTIPS[i]], lms[_FINGERTIPS[j]]
                dist = np.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)
                features.append(float(dist))

    # Pad or truncate to EXPECTED_FEATURE_SIZE
    vec = np.array(features, dtype=np.float32)
    if len(vec) < EXPECTED_FEATURE_SIZE:
        vec = np.pad(vec, (0, EXPECTED_FEATURE_SIZE - len(vec)))
    elif len(vec) > EXPECTED_FEATURE_SIZE:
        vec = vec[:EXPECTED_FEATURE_SIZE]

    return vec


def run(camera_index: int) -> None:
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

    print('Camera feed open — press q to quit')
    print(f'Feature vector size: {EXPECTED_FEATURE_SIZE}')

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
            hands_detected = len(results.multi_hand_landmarks)
            print(
                f'\rHands: {hands_detected}  '
                f'Features: {features.shape}  '
                f'min={features.min():.3f}  max={features.max():.3f}  '
                f'nonzero={np.count_nonzero(features)}',
                end='',
                flush=True,
            )

        cv2.imshow('WaveSL', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print()  # newline after the inline stats
    hands.close()
    cap.release()
    cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description='WaveSL - ASL to Speech')
    parser.add_argument('--camera', type=int, default=None,
                        help='Camera index (skips interactive selection)')
    args = parser.parse_args()

    camera_index = args.camera if args.camera is not None else select_camera()
    run(camera_index)


if __name__ == '__main__':
    main()
