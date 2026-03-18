"""ASL neural network model definition and loading utilities."""

import json
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn


class ASLModel(nn.Module):
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


def load_model(model_path: str) -> tuple["ASLModel", dict[int, str], torch.device]:
    """
    Load a trained ASLModel from a .pt state-dict file.
    Looks for class_mapping.json in the same directory.
    Returns (model, reverse_mapping, device).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path = Path(model_path)

    state_dict = torch.load(str(path), map_location=device, weights_only=True)

    first_key = next(k for k in state_dict if "weight" in k)
    last_key = next(k for k in reversed(list(state_dict.keys())) if "weight" in k)
    input_size = state_dict[first_key].shape[1]
    num_classes = state_dict[last_key].shape[0]

    model = ASLModel(input_size=input_size, num_classes=num_classes)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    mapping_path = path.parent / "class_mapping.json"
    reverse_mapping: dict[int, str] = {}
    if mapping_path.exists():
        with open(mapping_path) as f:
            class_mapping: dict[str, int] = json.load(f)
        reverse_mapping = {idx: name for name, idx in class_mapping.items()}
        print(f"Loaded model: {num_classes} classes from {path}")
    else:
        print(f"Loaded model: {num_classes} classes (no class_mapping.json found)")

    return model, reverse_mapping, device
