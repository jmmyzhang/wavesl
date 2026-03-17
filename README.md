# Smile and Wave!
WaveSL removes the need for sign language interpreters over video call, translating real-time ASL into realistic voice and subtitle outputs.

**Note**: This project uses the WLASL (Word-Level American Sign Language) dataset for training ASL recognition models.

## Overview

WaveSL creates a virtual video device (via OBS) and virtual audio device (via BlackHole) that can be used in Zoom, Discord, or any video conferencing application. The program:
- Captures video from your physical camera
- Processes frames in real-time to recognize ASL signs
- Converts recognized signs to text
- Synthesizes speech from the text
- Outputs the original video feed (for OBS to capture as virtual camera)
- Outputs the synthesized speech (to BlackHole virtual audio device)

## Prerequisites

1. **BlackHole** - Virtual audio device for macOS
   - Download from: https://github.com/ExistentialAudio/BlackHole
   - Install the 2ch (2 channel) version
   - This creates a virtual audio device that other applications can use as a microphone input

2. **OBS Studio** - For virtual camera
   - Download from: https://obsproject.com/
   - Install and set up OBS to capture the WaveSL video window as a virtual camera

3. **Python 3.8-3.11** with pip
   - **Important**: Coqui TTS requires Python <3.12, so Python 3.11 is the latest supported version
   - If you have Python 3.12+, you'll need to install Python 3.11 using pyenv or Homebrew:
     ```bash
     # Using Homebrew:
     brew install python@3.11
     # Then use: python3.11 -m venv venv
     ```

## Installation

1. **Install Python 3.11** (if you don't have it and your system Python is 3.12+):
   ```bash
   # Using Homebrew (recommended for macOS):
   brew install python@3.11
   
   # Verify installation:
   python3.11 --version
   ```

2. Create and activate a virtual environment using Python 3.11 (or 3.8-3.11):
```bash
# Remove old venv if it exists
rm -rf venv

# Create virtual environment with Python 3.11 (or python3.10, python3.9, etc.)
python3.11 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

   **Note**: If installation is taking too long, you can try:
   - Installing packages in stages (torch first, then TTS, then others)
   - Using the legacy resolver: `pip install --use-deprecated=legacy-resolver -r requirements.txt`
   - Or install core packages first: `pip install torch torchaudio TTS`, then the rest

**Note**: If you encounter an "externally-managed-environment" error, you must use a virtual environment as shown above. This is required on newer Python installations to protect system Python.

4. (Optional) Download and place your ASL recognition model in the project directory
   - The model should be a PyTorch model (.pt file)
   - Update the model path in `asl_recognizer.py` or pass it when initializing

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

Basic usage:
```bash
python src/main.py
```

With options:
```bash
# Use a specific camera
python src/main.py --camera 1

# Use a specific audio device (e.g., BlackHole)
python src/main.py --audio-device "BlackHole 2ch"

# List available audio devices
python src/main.py --list-audio
```

### Step 4: Configure Zoom/Discord

1. **Video**: Select "OBS Virtual Camera" as your camera
2. **Audio**: Select "BlackHole 2ch" (or your BlackHole device) as your microphone

### Controls

- Press `q` in the WaveSL window to quit the application

## Architecture

- `main.py` - Main application entry point and orchestration
- `asl_recognizer.py` - ASL sign recognition using MediaPipe and PyTorch
- `tts_engine.py` - Text-to-speech synthesis using Coqui TTS
- `audio_output.py` - Audio streaming to virtual audio device

## Model Setup

### Using Pre-trained WLASL Model

WaveSL uses a **pre-trained model** based on the WLASL (Word-Level American Sign Language) dataset. The model is trained **once** and then used for inference - no training happens when you run the application.

#### Quick Start

The repository includes a pre-trained WLASL model, so you can run immediately:

```bash
# Clone the repository
git clone <repo-url>
cd wavesl

# Install dependencies
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run the application (uses included pre-trained model)
python src/main.py
```

**Note**: If the model file is large, you may need Git LFS:
```bash
git lfs install
git lfs pull  # Download the model file
```

#### Training Your Own Model (Optional)

A pre-trained model is included in the repository. If you want to train your own model or use a different vocabulary size:

See **[SETUP_WLASL.md](SETUP_WLASL.md)** for complete setup instructions.

Quick summary:
1. **Download WLASL dataset** from [WLASL repository](https://github.com/dxli94/WLASL)
2. **Prepare dataset**: `python src/prepare_wlasl.py --wlasl-dir ~/wlasl --output-dir dataset/wlasl`
3. **Train model**: `./train_wlasl_model.sh`
4. **Use your trained model**: `python src/main.py` (will use your model if it exists)

#### Model Details

- **Dataset**: WLASL (Word-Level American Sign Language)
- **Vocabulary**: Up to 2,000 common ASL words
- **Architecture**: Neural network classifier using MediaPipe hand landmarks
- **Input**: Hand landmark features extracted from video frames
- **Output**: Sign class predictions mapped to text
- **Training**: Done once, model saved to `models/wlasl/best_model.pt`
- **Inference**: Fast, real-time recognition using pre-trained model

**Important**: The application uses a **pre-trained model** - it does NOT train on startup. Training is a one-time setup step.

## Troubleshooting

- **Camera not found**: Use `--camera` to specify a different camera index
- **Audio device not found**: Use `--list-audio` to see available devices, then specify with `--audio-device`
- **OBS can't see the window**: Make sure the WaveSL window is visible and not minimized
- **No ASL recognition**: Make sure you have a trained model loaded, or implement the placeholder recognition logic
