import torch


# ============================================================
# Device utilities
# ============================================================

def get_device(device_name):
    """Choose where training runs: CUDA, MPS, or CPU."""

    valid_devices = {"auto", "cuda", "mps", "cpu"}

    if device_name not in valid_devices:
        raise ValueError(
            f"Invalid device '{device_name}'. Use one of: {valid_devices}"
        )

    if device_name == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but CUDA is not available.")

    if device_name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested, but MPS is not available.")

    return device_name