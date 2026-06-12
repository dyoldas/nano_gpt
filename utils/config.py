import yaml


# ============================================================
# Config utilities
# ============================================================

def load_config(config_path):
    """Load a YAML config file into a Python dictionary.

    Example YAML:
        dataset: tiny_cosmo
        model_type: gpt
        batch_size: 64

    becomes:
        {
            "dataset": "tiny_cosmo",
            "model_type": "gpt",
            "batch_size": 64,
        }
    """

    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def apply_overrides(config, overrides):
    """Apply optional terminal overrides to the YAML config.

    Example:
        python train.py --config configs/pretrain.yaml --learning_rate 1e-4

    If learning_rate exists in the YAML file, this terminal value replaces it.

    This gives us two levels of control:
        1. YAML config for normal experiments.
        2. Terminal overrides for quick changes.
    """

    for key, value in overrides.items():

        # argparse gives None when the user did not pass that argument.
        # We only override config values when the user explicitly passed something.
        if value is not None:
            config[key] = value

    return config