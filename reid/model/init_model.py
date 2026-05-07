# =============================================================================
# build_model
# =============================================================================
"""
This file is the single entry point for model instantiation.
It reads the yaml config, builds VehicleViT, optionally loads a checkpoint,
and moves the model to the available device.

train.py and evaluate.py call build_model() and receive a ready-to-use model.
They never instantiate VehicleViT directly.

build_model workflow:
  1. load config    : read config/tiny_vit.yaml -> dict
  2. instantiate    : VehicleViT(**config) 
  3. load weights   : load checkpoint if checkpoint_path is provided
  4. device         : move model to GPU if available, else CPU
  5. return         : model ready for training or evaluation

See: model/vit.py — VehicleViT
See: config/tiny_vit.yaml — architecture hyperparameters
"""

import torch
import yaml
from model.vit import VehicleViT


def build_model(
    config_path:     str = "config/tiny_vit.yaml",
    checkpoint_path: str = None,
) -> VehicleViT:
    """
    Builds and returns a VehicleViT model ready for training or evaluation.

    Args:
        config_path     : path to the yaml config file
        checkpoint_path : path to a saved checkpoint — None trains from scratch

    Returns:
        model : VehicleViT on the available device (GPU if available, else CPU)
    """
    # 1. load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # 2. instantiate model
    model = VehicleViT(**config["model"])

    # 3. load weights if checkpoint is provided
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
        model.load_state_dict(state_dict)

    # 4. move model to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    return model