"""Text-to-Speech Engine using Coqui TTS for high-quality speech synthesis."""

import logging
from typing import Any, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "tts_models/en/ljspeech/tacotron2-DDC"


class TTSEngine:
    """Text-to-speech engine using Coqui TTS."""

    def __init__(self, model_name: str = _DEFAULT_MODEL) -> None:
        self.model_name = model_name
        self.tts: Optional[Any] = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._load_model()

    def _load_model(self) -> None:
        try:
            from TTS.api import TTS

            self.tts = TTS(model_name=self.model_name, progress_bar=False)
            self.tts.to(self._device)
            print(f"Loaded TTS model: {self.model_name}")
        except Exception as e:
            logger.warning("Could not load TTS model: %s -- TTS will be unavailable", e)
            self.tts = None

    def synthesize(self, text: str, sample_rate: int = 22050) -> Optional[np.ndarray]:
        """
        Synthesize speech from text.
        Returns audio as a float32 numpy array normalized to [-1, 1], or None.
        The sample_rate parameter is accepted for API compatibility; Coqui TTS
        produces audio at the model's native rate regardless of this value.
        """
        if not text or not text.strip():
            return None

        if self.tts is None:
            logger.warning("TTS model not loaded, cannot synthesize")
            return None

        try:
            audio = self.tts.tts(text=text)

            if not isinstance(audio, np.ndarray):
                audio = np.array(audio)

            if audio.dtype != np.float32:
                if audio.dtype == np.int16:
                    audio = audio.astype(np.float32) / 32768.0
                elif audio.dtype == np.int32:
                    audio = audio.astype(np.float32) / 2147483648.0
                else:
                    audio = audio.astype(np.float32)

            max_val = np.abs(audio).max()
            if max_val > 1.0:
                audio = audio / max_val

            return audio

        except Exception as e:
            logger.error("Error synthesizing speech: %s", e, exc_info=True)
            return None
