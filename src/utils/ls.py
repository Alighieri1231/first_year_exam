import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils.losses import *  # Import everything from the provided losses file


# Utility to dynamically fetch a loss function
def get_loss_function(loss_name: str):
    """
    Returns a loss function by name.

    Args:
        loss_name (str): Name of the loss function (case-sensitive).

    Returns:
        Callable: The corresponding loss function.

    Raises:
        ValueError: If the loss function is not found.
    """
    # Predefined PyTorch loss functions
    loss_functions = {
        "MSELoss": nn.MSELoss,
        "L1Loss": nn.L1Loss,
        "SmoothL1Loss": nn.SmoothL1Loss,
        "CrossEntropyLoss": nn.CrossEntropyLoss,
        "BCEWithLogitsLoss": nn.BCEWithLogitsLoss,
    }

    # Add custom loss functions from losses.py
    custom_loss_functions = {
        name: obj
        for name, obj in globals().items()
        if callable(obj) and name.startswith("compute_")
    }

    # Merge both dictionaries
    all_losses = {**loss_functions, **custom_loss_functions}

    if loss_name in all_losses:
        return all_losses[loss_name]()
    else:
        raise ValueError(
            f"Loss function '{loss_name}' not found. Available options: {list(all_losses.keys())}"
        )
