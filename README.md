# Smile and Wave!

WaveSL is a real-time American Sign Language (ASL) to speech translation application. It recognizes hand gestures in real-time using deep learning and converts them to spoken text, enabling ASL-to-speech communication over video calls.

**Technology Stack:**
- Python 3.11 | PyTorch LSTM | MediaPipe Hands | Coqui TTS
- Real-time hand landmark detection and sequence classification
- 10-word vocabulary trained on WLASL dataset

## What It Does

WaveSL captures video from your camera, detects hand movements using MediaPipe, feeds sequences of hand landmarks to a trained LSTM neural network, and synthesizes realistic speech from recognized signs. Recognized words are displayed as subtitles on the camera feed and can be output as audio to a virtual audio device for video conferencing platforms.

**Supported Signs:** before, computer, deaf, drink, hot, like, mother, orange, who, yes

**Demo Flow:**
1. Captures video from your physical camera
2. Detects hand landmarks from MediaPipe in real-time
3. Feeds landmark sequences to LSTM model for sign recognition
4. Converts recognized signs to text
5. Synthesizes speech using Coqui TTS
6. Displays subtitles on video feed
7. Outputs synthesized speech to virtual audio device

## Prerequisites

1. **BlackHole** - Virtual audio device for macOS
   - Download from: https://github.com/ExistentialAudio/BlackHole
   - Install the 2ch (2 channel) version
   - This creates a virtual audio device that other applications can use as a microphone input

2. **OBS Studio** - For virtual camera
   - Download from: https://obsproject.com/
   - Install and set up OBS to capture the WaveSL video window as a virtual camera

3. **Python 3.11** with pip
   - **Important**: Coqui TTS requires Python <3.12, so Python 3.11 is required
   - Install using Homebrew if needed:
     ```bash
     brew install python@3.11
     ```

## Installation

1. **Install Python 3.11** (if you don't have it and your system Python is 3.12+):
   ```bash
   # Using Homebrew (recommended for macOS):
   brew install python@3.11
   
   # Verify installation:
   python3.11 --version
   ```

2. Create and activate a virtual environment:
```bash
python3.11 -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

   **Note**: If installation is slow, try installing core packages first:
   ```bash
   pip install torch torchaudio TTS
   pip install -r requirements.txt
   ```

**Note**: If you encounter an "externally-managed-environment" error, you must use a virtual environment as shown above.

The project includes a pre-trained WLASL LSTM model at `models/wlasl/best_model.pt` (requires Git LFS).

## Usage

### Step 1: Set up BlackHole

1. Install BlackHole (see Prerequisites)
2. In System Preferences > Sound > Output, you can optionally set BlackHole as an output
3. The program will automatically route audio to BlackHole

### Step 2: Set up OBS Virtual Camera

1. Open OBS Studio
2. Add a "Window Capture" source
3. Select the "WaveSL - Camera Feed (for OBS)" window
4. Start Virtual Camera in OBS (Tools > Start Virtual Camera)

### Step 3: Run WaveSL

Basic usage (interactive camera/audio selection):
```bash
python src/main.py
```

With options:
```bash
# Use a specific camera index (skip device selection)
python src/main.py --camera 0

# Use a specific audio device (e.g., BlackHole)
python src/main.py --camera 0 --audio-device "BlackHole 2ch"

# Disable TTS (feature extraction and subtitle display only)
python src/main.py --no-tts

# Adjust confidence threshold (default 0.6)
python src/main.py --threshold 0.7

# Specify a different model
python src/main.py --model /path/to/model.pt
```

### Step 4: Configure Zoom/Discord

1. **Video**: Select "OBS Virtual Camera" as your camera
2. **Audio**: Select "BlackHole 2ch" (or your BlackHole device) as your microphone

### Controls

- Press `q` in the WaveSL window to quit the application

## Project Structure

```
src/
  main.py                 # Entry point: video capture, inference, subtitle overlay
  model.py               # LSTM/MLP architecture and model loading
  train_asl_model.py     # Training script with feature caching and augmentation
  prediction_smoother.py # Temporal smoothing with majority voting
  tts_engine.py          # Coqui TTS integration
  audio_output.py        # Virtual audio device output
  device_selector.py     # Interactive camera/audio device selection
  constants.py           # Feature dimensions, LSTM seq_len, fingertip indices
  prepare_wlasl.py       # Dataset preparation script
  prepare_dataset.py     # Generic dataset preparation utilities

models/wlasl/
  best_model.pt          # Pre-trained LSTM model (via Git LFS)
  class_mapping.json     # Sign name to class index mapping

tests/
  test_prediction_smoother.py  # Unit tests for temporal smoothing
```

## Model Architecture & Training

### Pre-trained Model

The repository includes a pre-trained LSTM model (`models/wlasl/best_model.pt`) trained on 10 common ASL signs from the WLASL dataset. To use it:

```bash
git lfs install
git lfs pull  # Download the model (if not already present)
python src/main.py
```

### How It Works

1. **Feature Extraction**: MediaPipe extracts 21 hand landmarks per hand (x, y, z coordinates) plus fingertip-to-fingertip distances = 146-dim feature vector per frame
2. **Sequence Buffering**: LSTM expects sequences of 16 frames (0.5s at 30fps)
3. **Classification**: LSTM processes the sequence and outputs confidence scores for 10 signs
4. **Smoothing**: Temporal smoother applies majority voting to reduce noise and detect transitions

### Model Details

| Aspect | Detail |
|--------|--------|
| **Type** | LSTM (Sequence classifier) |
| **Input** | 16 consecutive frames of 146-dim hand landmarks |
| **Output** | Confidence scores for 10 sign classes |
| **Architecture** | 2-layer LSTM (hidden=256) + dropout(0.3) + linear classifier |
| **Classes** | before, computer, deaf, drink, hot, like, mother, orange, who, yes |
| **Training Data** | WLASL (Word-Level American Sign Language) |

### Training Your Own Model (Optional)

If you want to retrain or use a different sign vocabulary:

1. **Download WLASL dataset**: https://github.com/dxli94/WLASL
2. **Prepare dataset**:
   ```bash
   python src/prepare_wlasl.py \
     --wlasl-dir /path/to/WLASL/start_kit \
     --output-dir dataset/wlasl \
     --class-mapping models/wlasl/class_mapping.json
   ```
3. **Train model**:
   ```bash
   bash train_wlasl_model.sh
   ```
   This trains an LSTM with:
   - Sequence length: 16 frames
   - Top 100 sign classes (by video count)
   - Data augmentation: horizontal flip + noise jitter
   - Output: `models/wlasl/best_model.pt`

4. **Update class mapping** (automatically done by prepare_wlasl.py)
   The class mapping JSON maps sign names to class indices (0-99)

### Training Script Options

```bash
python src/train_asl_model.py \
  --data-dir dataset/wlasl \
  --output-dir models/wlasl \
  --cache-dir dataset/wlasl_cache \
  --model-type lstm \
  --seq-len 16 \
  --top-n 100 \
  --augment \
  --epochs 50 \
  --batch-size 32 \
  --learning-rate 0.001 \
  --train-split 0.8
```

**Feature Caching**: Features are cached to disk on first run, then loaded instantly on subsequent epochs.

## Testing

Run the test suite:
```bash
pytest tests/
```

Tests cover the prediction smoother (temporal smoothing with majority voting). To add more tests, see `tests/test_prediction_smoother.py` for examples.

## Troubleshooting

| Issue | Solution |
|-------|----------|
| **Model not found** | Run `git lfs pull` to download pre-trained model, or train your own with `bash train_wlasl_model.sh` |
| **Camera not found** | Run `python src/main.py --camera N` with different N values (0, 1, 2...) |
| **Audio device not found** | Ensure BlackHole is installed; the app will prompt for device selection |
| **OBS can't capture window** | Make sure the WaveSL window is visible (not minimized or hidden) |
| **Poor recognition accuracy** | Ensure lighting is good, hands are clearly visible, and you're performing the sign cleanly |
| **Slow startup** | First run caches features (~1-2 min); subsequent runs are instant |
| **Import errors** | Ensure you've activated the venv: `source venv/bin/activate` |

## Dependencies

- `torch` / `torchaudio` - PyTorch deep learning
- `mediapipe` - Hand landmark detection
- `opencv-python` - Video capture and processing
- `TTS` (Coqui TTS) - Text-to-speech synthesis
- `sounddevice` - Virtual audio output
- `numpy` - Numerical computing
