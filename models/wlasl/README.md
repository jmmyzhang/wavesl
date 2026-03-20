# Pre-trained WLASL Model

This directory contains the pre-trained WLASL model for WaveSL.

## Files

- `best_model.pt` — Model weights tracked via Git LFS
- `class_mapping.json` — Maps class indices to sign names

## Model Information

- **Vocabulary**: 10 signs (before, computer, deaf, drink, hot, like, mother, orange, who, yes)
- **Validation Accuracy**: 29.03%
- **Architecture**: 2-layer LSTM (hidden=256, dropout=0.3)
- **Input**: 146-dim feature vectors × 16 frames (MediaPipe hand landmarks, 2 hands)
- **Dataset**: WLASL top-10 classes by video count
- **Parameters**: ~943K
- **Model Size**: ~3.8 MB

## Usage

The model is loaded automatically when you run:

```bash
python src/main.py
```

## Retraining

```bash
bash train_wlasl_model.sh
```

See the root README for full retraining instructions.
