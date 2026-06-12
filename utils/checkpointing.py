import os
import torch


# ============================================================
# Checkpointing utilities
# ============================================================

def save_checkpoint(
    model,
    optimizer,
    config,
    vocab_size,
    iter_num,
    best_val_loss,
    out_dir,
    filename="ckpt.pt",
    extra=None,
):
    """Save model/optimizer/config state to disk.

    A checkpoint stores everything needed to:

        1. generate from the model later,
        2. resume training later,
        3. inspect which config created this model.

    model.state_dict():
        learned neural network weights

    optimizer.state_dict():
        AdamW internal state, useful for resuming training smoothly

    config:
        hyperparameters and dataset information

    extra:
        optional extra metadata, useful for finetuning, e.g.
        {"init_from": "checkpoints/tiny_cosmo/ckpt.pt"}
    """

    os.makedirs(out_dir, exist_ok=True)

    ckpt = {
        # Model weights.
        "model": model.state_dict(),

        # Optimizer state.
        "optimizer": optimizer.state_dict(),

        # Full training config.
        "config": config,

        # Metadata.
        "vocab_size": vocab_size,
        "iter_num": iter_num,
        "best_val_loss": best_val_loss,
    }

    # Add optional metadata, for example:
    #
    #     init_from
    #     pretrain_config
    #     notes
    if extra is not None:
        ckpt.update(extra)

    ckpt_path = os.path.join(out_dir, filename)
    torch.save(ckpt, ckpt_path)

    print(f"Saved checkpoint to {ckpt_path}")

    return ckpt_path


def load_checkpoint(ckpt_path, device):
    """Load a checkpoint from disk.

    map_location=device ensures the checkpoint loads onto the selected device:

        "cpu"
        "cuda"
        "mps"
    """

    print(f"Loading checkpoint from {ckpt_path}")

    ckpt = torch.load(
        ckpt_path,
        map_location=device,
    )

    return ckpt


def load_model_weights(model, ckpt_path, device):
    """Load only model weights from a checkpoint.

    This is useful for finetuning.

    For finetuning, we usually want:

        pretrained model weights: yes
        old optimizer state: no

    because finetuning is a new phase with a new learning rate.
    """

    ckpt = load_checkpoint(ckpt_path, device)

    model.load_state_dict(ckpt["model"])

    print("Model weights loaded.")

    return ckpt


def load_training_state(model, optimizer, ckpt_path, device):
    """Load model and optimizer state for true training resume.

    This is different from finetuning.

    Use this when you want to continue the exact same training run.

    It restores:

        model weights
        optimizer state
        previous iteration
        best validation loss
    """

    ckpt = load_checkpoint(ckpt_path, device)

    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])

    iter_num = ckpt.get("iter_num", 0)
    best_val_loss = ckpt.get("best_val_loss", float("inf"))

    print(f"Training state loaded from iteration {iter_num}.")

    return ckpt, iter_num, best_val_loss