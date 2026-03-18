"""Audio Output Module -- streams synthesized audio to the selected output device."""

import logging
import queue
from typing import Optional

import numpy as np
import sounddevice as sd

logger = logging.getLogger(__name__)

_BLOCK_SIZE = 1024


class AudioOutput:
    """Handles audio output to a sounddevice OutputStream."""

    def __init__(
        self, device_name: Optional[str] = None, sample_rate: int = 22050
    ) -> None:
        self.sample_rate = sample_rate
        self.device_name = device_name
        self._device_id: Optional[int] = self._resolve_device(device_name)
        self._audio_queue: queue.Queue[np.ndarray] = queue.Queue()
        self._stream: Optional[sd.OutputStream] = None
        self._start_stream()

    @staticmethod
    def _resolve_device(device_name: Optional[str]) -> Optional[int]:
        if not device_name:
            print("Using default audio output device")
            return None

        devices = sd.query_devices()
        for i, device in enumerate(devices):
            if (
                device_name.lower() in device["name"].lower()
                and device["max_output_channels"] > 0
            ):
                print(f"Using audio device: {device['name']} (index {i})")
                return i

        print(f"Warning: Could not find audio device '{device_name}'")
        print("Available output devices:")
        for i, device in enumerate(devices):
            if device["max_output_channels"] > 0:
                print(f"  [{i}] {device['name']}")
        print("Using default output device")
        return None

    def _start_stream(self) -> None:
        try:
            self._stream = sd.OutputStream(
                device=self._device_id,
                samplerate=self.sample_rate,
                channels=1,
                dtype=np.float32,
                callback=self._audio_callback,
                blocksize=_BLOCK_SIZE,
            )
            self._stream.start()
            print("Audio output stream started")
        except Exception as e:
            logger.error("Error starting audio stream: %s", e)
            self._stream = None

    def _audio_callback(
        self, outdata: np.ndarray, frames: int, time_info, status
    ) -> None:
        if status:
            logger.warning("Audio callback status: %s", status)

        try:
            audio_data = self._audio_queue.get_nowait()
            if len(audio_data) < frames:
                audio_data = np.concatenate(
                    [audio_data, np.zeros(frames - len(audio_data), dtype=np.float32)]
                )
            elif len(audio_data) > frames:
                audio_data = audio_data[:frames]
            outdata[:, 0] = audio_data
        except queue.Empty:
            outdata.fill(0)

    def play(self, audio_data: np.ndarray) -> None:
        """Queue audio data for playback. audio_data must be float32 in [-1, 1]."""
        if self._stream is None:
            return

        if len(audio_data) == 0:
            return

        audio = (
            audio_data
            if audio_data.dtype == np.float32
            else audio_data.astype(np.float32)
        )

        max_val = np.abs(audio).max()
        if max_val > 1.0:
            audio = audio / max_val

        for i in range(0, len(audio), _BLOCK_SIZE):
            chunk = audio[i : i + _BLOCK_SIZE]
            try:
                self._audio_queue.put_nowait(chunk)
            except queue.Full:
                try:
                    self._audio_queue.get_nowait()
                    self._audio_queue.put_nowait(chunk)
                except queue.Empty:
                    pass

    def close(self) -> None:
        """Stop and close the audio output stream."""
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
            print("Audio output stream closed")
