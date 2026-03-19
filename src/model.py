"""ASL neural network model definition and loading utilities."""

import json
from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn


class ASLModel(nn.Module):
    """Single-frame MLP classifier."""

    def __init__(
        self,
        input_size: int,
        num_classes: int,
        hidden_sizes: Optional[list[int]] = None,
    ):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [256, 128, 64]
        layers: list[nn.Module] = []
        prev = input_size
        for h in hidden_sizes:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(0.3)]
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class ASLLSTMModel(nn.Module):
    """
    Sequence LSTM classifier over per-frame MediaPipe feature vectors.

    Input shape: (batch, seq_len, input_size)
    Uses the final hidden state of the last LSTM layer for classification.
    """

    def __init__(
        self,
        input_size: int,
        num_classes: int,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_size)
        _, (hidden, _) = self.lstm(x)
        # hidden: (num_layers, batch, hidden_size) — take last layer
        out = self.dropout(hidden[-1])
        return self.classifier(out)


AnyASLModel = Union[ASLModel, ASLLSTMModel]


def load_model(
    model_path: str,
) -> tuple[AnyASLModel, dict[int, str], torch.device]:
    """
    Load a trained ASLModel or ASLLSTMModel from a .pt state-dict file.
    Detects model type from state dict keys.
    Looks for class_mapping.json in the same directory.
    Returns (model, reverse_mapping, device).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path = Path(model_path)

    state_dict = torch.load(str(path), map_location=device, weights_only=True)
    keys = list(state_dict.keys())

    is_lstm = any("lstm" in k for k in keys)

    if is_lstm:
        # Infer dimensions from LSTM weight tensors
        # lstm.weight_ih_l0: (4*hidden_size, input_size)
        input_size = state_dict["lstm.weight_ih_l0"].shape[1]
        hidden_size = state_dict["lstm.weight_ih_l0"].shape[0] // 4
        num_classes = state_dict["classifier.weight"].shape[0]
        num_layers = sum(1 for k in keys if k.startswith("lstm.weight_ih_l"))
        model: AnyASLModel = ASLLSTMModel(
            input_size=input_size,
            num_classes=num_classes,
            hidden_size=hidden_size,
            num_layers=num_layers,
        )
        model_type = "LSTM"
    else:
        first_key = next(k for k in keys if "weight" in k)
        last_key = next(k for k in reversed(keys) if "weight" in k)
        input_size = state_dict[first_key].shape[1]
        num_classes = state_dict[last_key].shape[0]
        model = ASLModel(input_size=input_size, num_classes=num_classes)
        model_type = "MLP"

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    mapping_path = path.parent / "class_mapping.json"
    reverse_mapping: dict[int, str] = {}
    if mapping_path.exists():
        with open(mapping_path) as f:
            class_mapping: dict[str, int] = json.load(f)
        reverse_mapping = {idx: name for name, idx in class_mapping.items()}
        print(f"Loaded {model_type} model: {num_classes} classes from {path}")
    else:
        print(f"Loaded {model_type} model: {num_classes} classes (no class_mapping.json found)")

    return model, reverse_mapping, device
