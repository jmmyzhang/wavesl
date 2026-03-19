#!/usr/bin/env python3
"""
WaveSL - Real-time ASL to Speech Translation
Phase 3: TTS + audio routing, subtitle overlay
"""

import argparse
import collections
import contextlib
import logging
import os
import queue
import threading
from pathlib import Path
from typing import Optional

import cv2

with contextlib.redirect_stderr(open(os.devnull, "w")):
    import mediapipe as mp

import numpy as np
import torch

from audio_output import AudioOutput
from constants import DEFAULT_SEQ_LEN, EXPECTED_FEATURE_SIZE
from device_selector import run_device_selection
from model import ASLLSTMModel, AnyASLModel, load_model
from prediction_smoother import PredictionSmoother
from tts_engine import TTSEngine

logger = logging.getLogger(__name__)

_FINGERTIPS = [4, 8, 12, 16, 20]
CONFIDENCE_THRESHOLD = 0.6

_WINDOW_NAME = "WaveSL - Camera Feed"


# -- Feature extraction -------------------------------------------------------


def extract_features(hand_landmarks_list: list) -> np.ndarray:
    features: list[float] = []
    for hand_landmarks in hand_landmarks_list:
        for lm in hand_landmarks.landmark:
            features.extend([lm.x, lm.y, lm.z])
        lms = hand_landmarks.landmark
        for i in range(len(_FINGERTIPS)):
            for j in range(i + 1, len(_FINGERTIPS)):
                a, b = lms[_FINGERTIPS[i]], lms[_FINGERTIPS[j]]
                features.append(
                    float(
                        np.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)
                    )
                )
    vec = np.array(features, dtype=np.float32)
    if len(vec) < EXPECTED_FEATURE_SIZE:
        vec = np.pad(vec, (0, EXPECTED_FEATURE_SIZE - len(vec)))
    elif len(vec) > EXPECTED_FEATURE_SIZE:
        vec = vec[:EXPECTED_FEATURE_SIZE]
    return vec


# -- Inference ----------------------------------------------------------------


def infer(
    model: AnyASLModel,
    features: np.ndarray,
    reverse_mapping: dict[int, str],
    device: torch.device,
    threshold: float = CONFIDENCE_THRESHOLD,
) -> tuple[str, float, bool]:
    """
    Run one inference pass.
    features: (EXPECTED_FEATURE_SIZE,) for MLP, or (seq_len, EXPECTED_FEATURE_SIZE) for LSTM.
    Returns (label, confidence, above_threshold).
    """
    tensor = torch.from_numpy(features).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)
        confidence, class_idx = torch.max(probs, dim=1)

    conf = float(confidence.item())
    idx = int(class_idx.item())
    label = reverse_mapping.get(idx, str(idx))
    return label, conf, conf >= threshold


# -- Subtitle overlay ---------------------------------------------------------


def draw_subtitle(frame: np.ndarray, text: str) -> None:
    """Draw a subtitle at the bottom of the frame (shadow + white text)."""
    if not text:
        return
    y = frame.shape[0] - 40
    for color, thickness in [((0, 0, 0), 4), ((255, 255, 255), 2)]:
        cv2.putText(
            frame, text, (20, y),
            cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, thickness, cv2.LINE_AA,
        )


# -- TTS worker ---------------------------------------------------------------


def _tts_worker(
    tts_queue: "queue.Queue[Optional[str]]",
    tts_engine: TTSEngine,
    audio_output: AudioOutput,
) -> None:
    """Daemon thread: drain tts_queue, synthesize, and play each label."""
    while True:
        label = tts_queue.get()
        if label is None:  # shutdown sentinel
            break
        audio = tts_engine.synthesize(label)
        if audio is not None:
            audio_output.play(audio)


# -- Main loop ----------------------------------------------------------------


def run(
    camera_index: int,
    model_path: Optional[str],
    audio_device: Optional[str] = None,
    threshold: float = CONFIDENCE_THRESHOLD,
    tts_enabled: bool = True,
) -> None:
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {camera_index}")

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

    model: Optional[AnyASLModel] = None
    reverse_mapping: dict[int, str] = {}
    device = torch.device("cpu")
    if model_path:
        model, reverse_mapping, device = load_model(model_path)
    else:
        print("No model provided -- running feature extraction only")

    # LSTM rolling feature buffer
    is_lstm = isinstance(model, ASLLSTMModel) if model is not None else False
    seq_len = DEFAULT_SEQ_LEN
    feature_buffer: collections.deque[np.ndarray] = collections.deque(maxlen=seq_len)

    smoother = PredictionSmoother()

    # TTS + audio setup
    tts_queue: "queue.Queue[Optional[str]]" = queue.Queue()
    tts_engine: Optional[TTSEngine] = None
    audio_output: Optional[AudioOutput] = None
    tts_thread: Optional[threading.Thread] = None

    if tts_enabled:
        tts_engine = TTSEngine()
        audio_output = AudioOutput(device_name=audio_device)
        tts_thread = threading.Thread(
            target=_tts_worker,
            args=(tts_queue, tts_engine, audio_output),
            daemon=True,
        )
        tts_thread.start()

    # Track current stable label for subtitle
    current_label = ""

    print(f"Camera feed open -- press q to quit")
    print(f"TTS {'enabled' if tts_enabled else 'disabled'}")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.warning("failed to read frame")
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
                    if is_lstm:
                        feature_buffer.append(features)
                        if len(feature_buffer) < seq_len:
                            print(
                                f"\rBuffering frames {len(feature_buffer)}/{seq_len}...",
                                end="", flush=True,
                            )
                            draw_subtitle(frame, current_label)
                            cv2.imshow(_WINDOW_NAME, frame)
                            if cv2.waitKey(1) & 0xFF == ord("q"):
                                break
                            continue
                        infer_input = np.stack(feature_buffer)
                    else:
                        infer_input = features

                    label, conf, above = infer(
                        model, infer_input, reverse_mapping, device, threshold
                    )
                    marker = "+" if above else "?"

                    smoothed = smoother.update(label, conf) if above else None
                    if smoothed is not None:
                        current_label = smoothed.label
                        if smoothed.is_new and tts_engine is not None:
                            tts_queue.put(smoothed.label)

                    stable = (
                        f"  [{smoothed.label}]{'*' if smoothed.is_new else ' '}"
                        if smoothed
                        else ""
                    )
                    print(
                        f"\r{marker} {label:<12} {conf:.0%}{stable}   ",
                        end="", flush=True,
                    )
            else:
                smoother.clear()
                feature_buffer.clear()

            draw_subtitle(frame, current_label)
            cv2.imshow(_WINDOW_NAME, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        print()
        hands.close()
        cap.release()
        cv2.destroyAllWindows()
        if tts_thread is not None:
            tts_queue.put(None)  # stop sentinel
        if audio_output is not None:
            audio_output.close()


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="WaveSL - ASL to Speech")
    parser.add_argument("--camera", type=int, default=None,
                        help="Camera index (skips interactive selection)")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to trained .pt model file")
    parser.add_argument("--threshold", type=float, default=CONFIDENCE_THRESHOLD,
                        help=f"Confidence threshold (default: {CONFIDENCE_THRESHOLD})")
    parser.add_argument("--no-tts", action="store_true",
                        help="Disable TTS and audio output")
    args = parser.parse_args()

    model_path = args.model
    if model_path is None:
        default = Path(__file__).parent.parent / "models" / "wlasl" / "best_model.pt"
        if default.exists():
            model_path = str(default)
        else:
            print(f"Warning: no model found at {default}")
            print("Run train_wlasl_model.sh to train one, or pass --model <path>")

    if args.camera is not None:
        camera_index = args.camera
        audio_device = None
    else:
        camera_index, audio_device = run_device_selection()

    run(
        camera_index=camera_index,
        model_path=model_path,
        audio_device=audio_device,
        threshold=args.threshold,
        tts_enabled=not args.no_tts,
    )


if __name__ == "__main__":
    main()
