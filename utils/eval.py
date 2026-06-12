import torch

from utils.data import get_batch


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def estimate_loss(model, train_data, val_data, config, device):
    """Estimate train and validation loss.

    We do not want to evaluate on just one batch because one batch is noisy.

    Instead, we sample eval_iters batches, compute their losses, and average them.

    @torch.no_grad() means:
        do not store gradients during evaluation.

    This saves memory and makes evaluation faster.
    """

    out = {}

    # model.eval() changes behavior of layers like Dropout.
    #
    # During training:
    #     dropout randomly removes activations.
    #
    # During evaluation:
    #     dropout is disabled for stable loss estimates.
    model.eval()

    for split in ["train", "val"]:

        # Store loss from each evaluation batch.
        losses = torch.zeros(config["eval_iters"])

        for k in range(config["eval_iters"]):

            # Sample a batch from train or validation data.
            xb, yb = get_batch(
                split=split,
                train_data=train_data,
                val_data=val_data,
                config=config,
                device=device,
            )

            # Forward pass only.
            #
            # logits = predictions
            # loss   = cross-entropy against true next tokens
            _, loss = model(xb, yb)

            # loss.item() converts the scalar tensor into a Python float.
            losses[k] = loss.item()

        # Average loss across eval_iters batches.
        out[split] = losses.mean().item()

    # Return model to training mode to re-enable dropout.
    model.train()

    return out