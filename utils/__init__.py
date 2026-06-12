"""Utility package exports.

This file makes the utils directory behave like a clean Python package.

Instead of importing helper functions from individual files like:

    from utils.config import load_config, apply_overrides
    from utils.device import get_device
    from utils.data import load_data, get_batch
    from utils.eval import estimate_loss
    from utils.checkpointing import save_checkpoint

we can import directly from the package:

    from utils import load_config, get_device, get_batch, estimate_loss

The actual function definitions still live in their own files:

    utils/config.py
    utils/device.py
    utils/data.py
    utils/eval.py
    utils/checkpointing.py

This keeps train.py cleaner while keeping the utility code organized.
"""


# Config loading and terminal overrides.
from .config import load_config, apply_overrides

# Device selection: "auto", "cuda", "mps", or "cpu".
from .device import get_device

# Data loading and mini-batch sampling.
from .data import load_data, get_batch

# Evaluation helper for train/validation loss.
from .eval import estimate_loss

# Checkpoint save/load helpers.
from .checkpointing import (
    save_checkpoint,
    load_checkpoint,
    load_model_weights,
    load_training_state,
)


# Controls what gets imported when someone writes:
#
#     from utils import *
#
# This is optional, but it makes the public interface explicit.
__all__ = [
    "load_config",
    "apply_overrides",
    "get_device",
    "load_data",
    "get_batch",
    "estimate_loss",
    "save_checkpoint",
    "load_checkpoint",
    "load_model_weights",
    "load_training_state",
]
