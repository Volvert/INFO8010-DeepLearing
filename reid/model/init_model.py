# =============================================================================
# init_model.py — Model builder
# =============================================================================
"""
Single entry point for model instantiation.

Receives the model config dict and device from main.py — never reads
the YAML itself. This keeps the YAML parsing in one place (main.py)
and makes build_model() easier to test in isolation.

build_model workflow:
  1. instantiate  : VehicleViT(**model_cfg)
  2. load weights : load checkpoint if checkpoint_path is provided
  3. device       : move model to device
  4. return       : model ready for training or evaluation

See: model/vit.py        — VehicleViT
See: config/tiny_vit.yaml — cfg["model"] section passed by main.py
"""

import torch
from model.vit import VehicleViT


def build_model(
    model_cfg:       dict,
    device:          torch.device,
    checkpoint_path: str | None = None,
) -> VehicleViT:
    """
    Builds and returns a VehicleViT model ready for training or evaluation.

    Args:
        model_cfg       : dict — cfg["model"] from tiny_vit.yaml
                          passed directly to VehicleViT(**model_cfg)
        device          : torch.device — GPU or CPU, detected once in main.py
        checkpoint_path : path to a saved checkpoint — None = train from scratch

    Returns:
        model : VehicleViT moved to device, ready for .train() or .eval()
    """

    # 1. instantiate model from config dict
    model = VehicleViT(**model_cfg)

    # 2. load checkpoint if provided
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        # support both raw state_dict and {"model": state_dict} checkpoints
        state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
        model.load_state_dict(state_dict)

    # 3. move to device
    model = model.to(device)

    return model