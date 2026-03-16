"""
Audio Output Module
Streams audio to the selected output device (e.g., BlackHole)
"""

import sounddevice as sd
import numpy as np
from typing import Optional
import queue


class AudioOutput:
    """Handles audio output to virtual audio device"""
    
    def __init__(self, device_name: Optional[str] = None, sample_rate: int = 22050):
        """
        Initialize audio output
        
        Args:
            device_name: Name of the audio output device (e.g., "BlackHole 2ch")
                        If None, uses default output device
            sample_rate: Audio sample rate
        """
        self.sample_rate = sample_rate
        self.device_name = device_name
        self.device_id = None
        
        # Find device by name if specified
        if device_name:
            devices = sd.query_devices()
            for i, device in enumerate(devices):
                if device_name.lower() in device['name'].lower() and device['max_output_channels'] > 0:
                    self.device_id = i
                    print(f"Using audio device: {device['name']} (index {i})")
                    break
            
            if self.device_id is None:
                print(f"Warning: Could not find audio device '{device_name}'")
                print("Available output devices:")
                for i, device in enumerate(devices):
                    if device['max_output_channels'] > 0:
                        print(f"  [{i}] {device['name']}")
                print("Using default output device")
        else:
            print("Using default audio output device")
        
        # Audio stream
        self.blocksize = 1024  # Samples per callback invocation; chunk_size in play() must match
        self.stream = None
        self.audio_queue = queue.Queue()
        
        # Start audio stream
        self._start_stream()
    
    def _start_stream(self):
        """Start the audio output stream"""
        try:
            self.stream = sd.OutputStream(
                device=self.device_id,
                samplerate=self.sample_rate,
                channels=1,  # Mono
                dtype=np.float32,
                callback=self._audio_callback,
                blocksize=self.blocksize
            )
            self.stream.start()
            print("Audio output stream started")
        except Exception as e:
            print(f"Error starting audio stream: {e}")
            self.stream = None
    
    def _audio_callback(self, outdata, frames, time_info, status):
        """Callback function for audio stream"""
        if status:
            print(f"Audio callback status: {status}")
        
        try:
            # Get audio data from queue
            audio_data = self.audio_queue.get_nowait()
            
            # Ensure correct shape and length
            if len(audio_data) < frames:
                # Pad with zeros if needed
                padding = np.zeros(frames - len(audio_data), dtype=np.float32)
                audio_data = np.concatenate([audio_data, padding])
            elif len(audio_data) > frames:
                # Truncate if needed
                audio_data = audio_data[:frames]
            
            # Reshape to (frames, channels)
            outdata[:, 0] = audio_data
            
        except queue.Empty:
            # No audio data available, output silence
            outdata.fill(0)
    
    def play(self, audio_data: np.ndarray):
        """
        Queue audio data for playback
        
        Args:
            audio_data: Audio data as numpy array (normalized to [-1, 1])
        """
        if self.stream is None:
            return
        
        if audio_data is None or len(audio_data) == 0:
            return
        
        # Ensure audio is in correct format
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
        
        # Ensure values are in [-1, 1] range
        max_val = np.abs(audio_data).max()
        if max_val > 1.0:
            audio_data = audio_data / max_val
        
        # Split into chunks matching blocksize so the callback consumes exactly
        # one chunk per invocation without truncation or padding
        chunk_size = self.blocksize
        for i in range(0, len(audio_data), chunk_size):
            chunk = audio_data[i:i + chunk_size]
            try:
                self.audio_queue.put_nowait(chunk)
            except queue.Full:
                # Drop oldest chunk if queue is full
                try:
                    self.audio_queue.get_nowait()
                    self.audio_queue.put_nowait(chunk)
                except queue.Empty:
                    pass
    
    def close(self):
        """Close the audio output stream"""
        if self.stream:
            self.stream.stop()
            self.stream.close()
            print("Audio output stream closed")

