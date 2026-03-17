#!/usr/bin/env python3
"""
WaveSL - Real-time ASL to Speech Translation
Phase 1: Camera input, display, and MediaPipe hand detection overlay
"""

import argparse
import cv2
import mediapipe as mp
from device_selector import select_camera


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

    while True:
        ret, frame = cap.read()
        if not ret:
            print('Warning: failed to read frame')
            continue

        # MediaPipe requires RGB
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

        cv2.imshow('WaveSL', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

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
