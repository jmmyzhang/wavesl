"""
ASL Recognition Module
Uses MediaPipe for hand tracking and PyTorch model for sign recognition
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import Optional, Dict
import torch
import torch.nn as nn
import json
from pathlib import Path

from constants import EXPECTED_FEATURE_SIZE


class ASLModel(nn.Module):
    """Neural network model for ASL sign recognition (matches training architecture)"""
    
    def __init__(self, input_size: int, num_classes: int, hidden_sizes: list = [256, 128, 64]):
        super(ASLModel, self).__init__()
        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, num_classes))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class ASLRecognizer:
    """Recognizes ASL signs from video frames"""
    
    def __init__(self, model_path: Optional[str] = None, class_mapping_path: Optional[str] = None):
        """
        Initialize ASL recognizer
        
        Args:
            model_path: Path to PyTorch model state dict (.pt file) for ASL recognition.
                       If None, uses placeholder.
            class_mapping_path: Path to JSON file containing class mapping.
                               If None, tries to find it next to model_path.
        """
        # Initialize MediaPipe hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        # Initialize PyTorch model
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.class_mapping: Dict[str, int] = {}
        self.reverse_mapping: Dict[int, str] = {}
        
        if model_path:
            self._load_model(model_path, class_mapping_path)
        
        # State for accumulating signs
        self.sign_buffer = []
        self.buffer_size = 10
        self.last_text = ""
    
    def _load_model(self, model_path: str, class_mapping_path: Optional[str] = None):
        """Load trained model and class mapping"""
        try:
            model_file = Path(model_path)
            
            # Try to find class mapping file
            if class_mapping_path is None:
                # Look for class_mapping.json in same directory as model
                mapping_file = model_file.parent / 'class_mapping.json'
                if mapping_file.exists():
                    class_mapping_path = str(mapping_file)
            
            # Load class mapping
            if class_mapping_path and Path(class_mapping_path).exists():
                with open(class_mapping_path, 'r') as f:
                    self.class_mapping = json.load(f)
                # Create reverse mapping (class_idx -> sign_name)
                self.reverse_mapping = {idx: name for name, idx in self.class_mapping.items()}
                print(f"Loaded class mapping with {len(self.class_mapping)} classes")
            else:
                print("Warning: No class mapping found. Using placeholder mapping.")
            
            # Load model
            # Try loading as state dict first (for models trained with train_asl_model.py)
            try:
                state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
                
                # Determine model architecture from class mapping or state dict
                if self.class_mapping:
                    num_classes = len(self.class_mapping)
                else:
                    # Try to infer from state dict
                    # Look for the last linear layer weight
                    last_layer_key = None
                    for key in reversed(list(state_dict.keys())):
                        if 'weight' in key and 'network' in key:
                            last_layer_key = key
                            break
                    if last_layer_key:
                        num_classes = state_dict[last_layer_key].shape[0]
                    else:
                        raise ValueError("Could not determine num_classes from state dict")
                
                # Determine input size from first layer
                first_layer_key = None
                for key in sorted(state_dict.keys()):
                    if 'weight' in key and 'network' in key:
                        first_layer_key = key
                        break
                if first_layer_key:
                    input_size = state_dict[first_layer_key].shape[1]
                else:
                    input_size = 146  # Default for 2 hands
                
                # Create model with correct architecture
                self.model = ASLModel(input_size=input_size, num_classes=num_classes)
                self.model.load_state_dict(state_dict)
                self.model.to(self.device)
                self.model.eval()
                print(f"✓ Loaded pre-trained WLASL model from {model_path}")
                print(f"  Model architecture: input_size={input_size}, num_classes={num_classes}")
                if self.reverse_mapping:
                    print(f"  Vocabulary size: {len(self.reverse_mapping)} signs")
            except Exception as e1:
                # Try loading as TorchScript model
                try:
                    self.model = torch.jit.load(model_path, map_location=self.device)
                    self.model.eval()
                    print(f"✓ Loaded pre-trained WLASL model (TorchScript) from {model_path}")
                except Exception as e2:
                    print(f"✗ Error: Could not load pre-trained model from {model_path}")
                    print(f"  State dict error: {e1}")
                    print(f"  TorchScript error: {e2}")
                    print("  Using placeholder recognition (no model loaded)")
                    print("  To use a trained model, train one first:")
                    print("    ./train_wlasl_model.sh")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Using placeholder recognition")
        
    def extract_hand_features(self, frame, hand_landmarks) -> np.ndarray:
        """
        Extract features from hand landmarks
        
        Args:
            frame: Video frame
            hand_landmarks: MediaPipe hand landmarks
            
        Returns:
            Feature vector
        """
        features = []

        # Extract normalized landmark coordinates
        for landmark in hand_landmarks.landmark:
            features.extend([landmark.x, landmark.y, landmark.z])
        
        # Calculate relative distances between key points
        # (thumb tip, index tip, middle tip, ring tip, pinky tip)
        key_points = [4, 8, 12, 16, 20]  # Landmark indices for fingertips
        
        if len(hand_landmarks.landmark) >= 21:
            for i in range(len(key_points)):
                for j in range(i + 1, len(key_points)):
                    p1 = hand_landmarks.landmark[key_points[i]]
                    p2 = hand_landmarks.landmark[key_points[j]]
                    dist = np.sqrt(
                        (p1.x - p2.x)**2 + 
                        (p1.y - p2.y)**2 + 
                        (p1.z - p2.z)**2
                    )
                    features.append(dist)
        
        return np.array(features, dtype=np.float32)
    
    def recognize_sign(self, features: np.ndarray) -> Optional[str]:
        """
        Recognize sign from features using PyTorch model
        
        Args:
            features: Feature vector extracted from hand landmarks
            
        Returns:
            Recognized sign text or None
        """
        if self.model is None:
            # Placeholder: return None or simple gesture detection
            # In production, this would use your trained ASL model
            return None
        
        try:
            # Prepare input tensor
            features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                output = self.model(features_tensor)
                probs = torch.softmax(output, dim=1)
                confidence, predicted_class = torch.max(probs, dim=1)

                if confidence.item() < 0.6:
                    return None

                sign_text = self._class_to_text(predicted_class.item())
                return sign_text
        except Exception as e:
            print(f"Error in sign recognition: {e}")
            return None
    
    def _class_to_text(self, class_idx: int) -> str:
        """
        Map class index to sign text
        
        Uses loaded class mapping if available, otherwise placeholder
        """
        if self.reverse_mapping:
            return self.reverse_mapping.get(class_idx, "")
        
        # Placeholder mapping for testing
        placeholder_mapping = {
            0: "hello",
            1: "thank you",
            2: "yes",
            3: "no",
        }
        return placeholder_mapping.get(class_idx, "")
    
    def process_frame(self, frame: np.ndarray) -> Optional[str]:
        """
        Process a video frame to recognize ASL signs
        
        Args:
            frame: Video frame (BGR format)
            
        Returns:
            Recognized text or None
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = self.hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            # Process each detected hand
            all_features = []
            
            for hand_landmarks in results.multi_hand_landmarks:
                features = self.extract_hand_features(frame, hand_landmarks)
                all_features.append(features)
            
            # Combine features from both hands if available
            if len(all_features) == 1:
                combined_features = all_features[0]
            elif len(all_features) == 2:
                combined_features = np.concatenate(all_features)
            else:
                return None

            # Pad or truncate to the fixed size the model expects
            if len(combined_features) < EXPECTED_FEATURE_SIZE:
                padded = np.zeros(EXPECTED_FEATURE_SIZE, dtype=np.float32)
                padded[:len(combined_features)] = combined_features
                combined_features = padded
            elif len(combined_features) > EXPECTED_FEATURE_SIZE:
                combined_features = combined_features[:EXPECTED_FEATURE_SIZE]

            # Recognize sign
            text = self.recognize_sign(combined_features)
            
            # Accumulate signs in buffer for temporal smoothing
            if text:
                self.sign_buffer.append(text)
                if len(self.sign_buffer) > self.buffer_size:
                    self.sign_buffer.pop(0)
                
                # Return most common sign in buffer
                if len(self.sign_buffer) >= 3:
                    from collections import Counter
                    most_common = Counter(self.sign_buffer).most_common(1)[0]
                    if most_common[1] >= len(self.sign_buffer) // 2:
                        return most_common[0]
            
            return text
        
        return None

