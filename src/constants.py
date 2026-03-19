"""
Shared constants for WaveSL training and inference.
Both train_asl_model.py and asl_recognizer.py import from here to
ensure the feature dimensions always stay in sync.
"""

# Feature vector size: 2 hands x (21 landmarks x 3 coords + 10 fingertip distances)
EXPECTED_FEATURE_SIZE = 146

# Number of frames sampled per video for sequence models (LSTM)
DEFAULT_SEQ_LEN = 16
