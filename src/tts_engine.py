"""
Text-to-Speech Engine
Uses Coqui TTS for high-quality speech synthesis
"""

import numpy as np
from typing import Optional
import torch


class TTSEngine:
    """Text-to-speech engine using Coqui TTS"""
    
    def __init__(self, model_name: str = "tts_models/en/ljspeech/tacotron2-DDC"):
        """
        Initialize TTS engine
        
        Args:
            model_name: Coqui TTS model name to use
        """
        self.model_name = model_name
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self._load_model()
    
    def _load_model(self):
        """Load TTS model"""
        try:
            from TTS.api import TTS
            
            # Initialize TTS
            self.tts = TTS(model_name=self.model_name, progress_bar=False)
            self.tts.to(self.device)
            
            print(f"Loaded TTS model: {self.model_name}")
        except Exception as e:
            print(f"Warning: Could not load TTS model: {e}")
            print("TTS functionality will be limited")
            self.tts = None
    
    def synthesize(self, text: str, sample_rate: int = 22050) -> Optional[np.ndarray]:
        """
        Synthesize speech from text
        
        Args:
            text: Text to synthesize
            sample_rate: Output sample rate
            
        Returns:
            Audio data as numpy array (normalized to [-1, 1]) or None
        """
        if not text or not text.strip():
            return None
        
        if self.tts is None:
            print("TTS model not loaded, cannot synthesize")
            return None
        
        try:
            # Synthesize speech
            # Coqui TTS returns audio as numpy array
            audio = self.tts.tts(text=text)
            
            # Convert to numpy array if needed
            if isinstance(audio, list):
                audio = np.array(audio)
            elif not isinstance(audio, np.ndarray):
                audio = np.array(audio)
            
            # Normalize to [-1, 1] range if needed
            if audio.dtype != np.float32:
                if audio.dtype == np.int16:
                    audio = audio.astype(np.float32) / 32768.0
                elif audio.dtype == np.int32:
                    audio = audio.astype(np.float32) / 2147483648.0
                else:
                    audio = audio.astype(np.float32)
            
            # Ensure values are in [-1, 1] range
            max_val = np.abs(audio).max()
            if max_val > 1.0:
                audio = audio / max_val
            
            return audio
            
        except Exception as e:
            print(f"Error synthesizing speech: {e}")
            import traceback
            traceback.print_exc()
            return None

