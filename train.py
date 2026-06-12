import os
import pickle
import argparse

import torch

from models import create_model

from utils import (
    load_config,
    apply_overrides,
    get_device,
    load_data,
    get_batch,
    estimate_loss,
    save_checkpoint,
    load_model_weights,
)


# ============================================================
# Main training script
# ============================================================

def main():
    # ------------------------------------------------------------
    # Terminal arguments
    # ------------------------------------------------------------

    parser = argparse.ArgumentParser()

    # Required YAML config.
    # Example:
    #     python train.py --config configs/pretrain.yaml
    #
    # or:
    #     python train.py --config configs/finetune.yaml
    parser.add_argument("--config", type=str, required=True)

    # Optional terminal overrides. These default to None.
    # If the user passes one, it overwrites the YAML value.
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--model_type", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)

    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--block_size", type=int, default=None)
    parser.add_argument("--max_iters", type=int, default=None)
    parser.add_argument("--eval_interval", type=int, default=None)
    parser.add_argument("--eval_iters", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)

    parser.add_argument("--n_embd", type=int, default=None)
    parser.add_argument("--n_head", type=int, default=None)
    parser.add_argument("--n_layer", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)

    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--init_from", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)

    args = parser.parse_args()

    # ------------------------------------------------------------
    # Load config
    # ------------------------------------------------------------

    # Load base experiment settings from YAML.
    config = load_config(args.config)

    # Convert argparse Namespace into a normal dictionary.
    # Example:
    #     args.learning_rate
    #
    # becomes:
    #     {"learning_rate": value}
    overrides = vars(args)

    # Do not put the config file path itself into the training config.
    overrides.pop("config")

    # Apply terminal overrides.
    # Example:
    #     python train.py --config configs/pretrain.yaml --learning_rate 1e-4
    #
    # replaces the YAML learning_rate with 1e-4.
    config = apply_overrides(config, overrides)

    # ------------------------------------------------------------
    # Device and randomness
    # ------------------------------------------------------------

    # device can be:
    #     "auto" -> choose best available
    #     "cuda" -> NVIDIA GPU
    #     "mps"  -> Apple Silicon GPU
    #     "cpu"  -> CPU
    device = get_device(config.get("device", "auto"))

    # Seed controls random initialization and batch sampling.
    # Same seed gives more reproducible training behavior.
    torch.manual_seed(config.get("seed", 1337))

    print(f"Using device: {device}")

    # ------------------------------------------------------------
    # Dataset paths
    # ------------------------------------------------------------

    # Example:
    #     dataset: tiny_cosmo
    #
    # means:
    #     data/tiny_cosmo/train.bin
    #     data/tiny_cosmo/val.bin
    #     data/tiny_cosmo/meta.pkl
    data_dir = os.path.join("data", config["dataset"])

    meta_path = os.path.join(data_dir, "meta.pkl")

    # Checkpoint directory:
    # If out_dir is not provided in YAML, default to:
    #     checkpoints/<dataset_name>
    out_dir = config.get("out_dir")

    if out_dir is None:
        out_dir = os.path.join("checkpoints", config["dataset"])

    # Make directory if missing
    os.makedirs(out_dir, exist_ok=True)

    # ------------------------------------------------------------
    # Load tokenizer metadata
    # ------------------------------------------------------------

    # meta.pkl was created by prepare.py. It contains:
    #     vocab_size
    #     stoi: character -> integer
    #     itos: integer -> character
    #
    # Training only needs vocab_size.
    # Generation will later need stoi/itos.
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)

    vocab_size = meta["vocab_size"]

    print(f"Dataset: {config['dataset']}")
    print(f"Model type: {config['model_type']}")
    print(f"Vocab size: {vocab_size}")

    # ------------------------------------------------------------
    # Load tokenized data
    # ------------------------------------------------------------

    # load_data uses np.memmap internally.
    # That means train.bin and val.bin behave like arrays,
    # but the full files are not loaded into RAM.
    train_data, val_data = load_data(data_dir)

    print(f"Train tokens/chars: {len(train_data):,}")
    print(f"Val tokens/chars: {len(val_data):,}")

    # ------------------------------------------------------------
    # Create model
    # ------------------------------------------------------------

    model = create_model(config, vocab_size)

    # Move model weights to selected device.
    model = model.to(device)

    # ------------------------------------------------------------
    # Optional finetuning initialization
    # ------------------------------------------------------------

    # This is the key unified pretraining/finetuning mechanism.
    # If:
    #     init_from: null
    #
    # then training starts from random initialization.
    #
    # If:
    #     init_from: checkpoints/tiny_cosmo/ckpt.pt
    #
    # then we load pretrained model weights first and continue training
    # on the current dataset. That is finetuning.
    if config.get("init_from") is not None:
        load_model_weights(
            model=model,
            ckpt_path=config["init_from"],
            device=device,
        )

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {num_params / 1e6:.2f}M")

    # ------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------

    # AdamW is the standard optimizer for transformer training.
    # For pretraining:
    #     usually use a larger learning rate, e.g. 3e-4.
    #
    # For finetuning:
    #     usually use a smaller learning rate, e.g. 1e-4 or 5e-5.
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
    )

    # Track the best validation loss so far.
    # We only save the checkpoint when validation improves.
    best_val_loss = float("inf")

    # ------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------

    for iter_num in range(config["max_iters"] + 1):

        # --------------------------------------------------------
        # Evaluation + checkpoint saving
        # --------------------------------------------------------

        if iter_num % config["eval_interval"] == 0:

            # Estimate loss on train and validation sets.
            # estimate_loss samples multiple batches and averages the loss,
            # so it is more stable than checking one random batch.
            losses = estimate_loss(
                model=model,
                train_data=train_data,
                val_data=val_data,
                config=config,
                device=device,
            )

            print(
                f"step {iter_num}: "
                f"train loss {losses['train']:.4f}, "
                f"val loss {losses['val']:.4f}"
            )

            # Save only if validation loss improves.
            # Validation loss matters because it measures performance
            # on data the model is not directly optimizing on.
            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]

                extra = {}

                # If this was a finetuning run, record which checkpoint the model started from.
                if config.get("init_from") is not None:
                    extra["init_from"] = config["init_from"]

                # save_checkpoint stores:
                #     model weights
                #     optimizer state
                #     config
                #     vocab_size
                #     iteration number
                #     best validation loss
                #
                # This makes the checkpoint usable for generation, inspection, and resuming.
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    config=config,
                    vocab_size=vocab_size,
                    iter_num=iter_num,
                    best_val_loss=best_val_loss,
                    out_dir=out_dir,
                    extra=extra,
                )

        # --------------------------------------------------------
        # One gradient update
        # --------------------------------------------------------

        # Sample one mini-batch from training data.
        # xb shape:
        #     (batch_size, block_size)
        #
        # yb shape:
        #     (batch_size, block_size)
        #
        # yb is xb shifted by one token.
        xb, yb = get_batch(
            split="train",
            train_data=train_data,
            val_data=val_data,
            config=config,
            device=device,
        )

        # Forward pass.
        # The model predicts next-token logits and computes cross-entropy loss.
        _, loss = model(xb, yb)

        # Clear gradients from the previous step.
        # PyTorch accumulates gradients by default,
        # so we must reset them before each backward pass.
        optimizer.zero_grad(set_to_none=True)

        # Backward pass computes:
        #     d(loss) / d(parameter)
        #
        # for every trainable parameter in the model.
        loss.backward()

        # Optimizer uses the gradients to update model parameters.
        optimizer.step()

    print("Training done.")
    print(f"Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
