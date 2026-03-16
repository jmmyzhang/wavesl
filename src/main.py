#!/usr/bin/env python3
"""
WaveSL - Real-time ASL to Speech Translation
Phase 1: Camera input and display
"""

import argparse
import cv2
from device_selector import select_camera


def run(camera_index: int) -> None:
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f'Could not open camera {camera_index}')

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    print('Camera feed open — press q to quit')

    while True:
        ret, frame = cap.read()
        if not ret:
            print('Warning: failed to read frame')
            continue

        cv2.imshow('WaveSL', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

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
