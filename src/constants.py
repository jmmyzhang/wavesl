"""
Shared constants for WaveSL training and inference.
Both train_asl_model.py and asl_recognizer.py import from here to
ensure the feature dimensions always stay in sync.
"""

# Feature vector size: 2 hands × (21 landmarks × 3 coords + 10 fingertip distances)
EXPECTED_FEATURE_SIZE = 146
