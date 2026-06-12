import os

import numpy as np
import torch


# ============================================================
# Data loading
# ============================================================

def load_data(data_dir):
    """Load train.bin and val.bin as memory-mapped NumPy arrays.

    np.memmap makes the binary files behave like NumPy arrays without
    loading the entire dataset into RAM.

    This is important once train.bin becomes large.
    """

    train_path = os.path.join(data_dir, "train.bin")
    val_path = os.path.join(data_dir, "val.bin")

    train_data = np.memmap(train_path, dtype=np.uint16, mode="r")
    val_data = np.memmap(val_path, dtype=np.uint16, mode="r")

    return train_data, val_data


# ============================================================
# Data batching
# ============================================================

def get_batch(split, train_data, val_data, config, device):
    """Sample a random mini-batch from train.bin or val.bin.

    The dataset is a long 1D array of token IDs:

        data = [14, 51, 9, 20, 8, 99, 3, ...]

    For language modeling, we train the model to predict the next token.

    So if:

        x = [14, 51, 9, 20]

    then:

        y = [51, 9, 20, 8]

    The model sees x and tries to predict y.
    """

    # Choose whether to sample from training data or validation data.
    data = train_data if split == "train" else val_data

    block_size = config["block_size"]
    batch_size = config["batch_size"]

    # Random starting positions.
    #
    # Example:
    #
    #     ix = [100, 830, 12, 91]
    #
    # means:
    #
    #     sequence 1 starts at token 100
    #     sequence 2 starts at token 830
    #     sequence 3 starts at token 12
    #     sequence 4 starts at token 91
    #
    # We subtract block_size + 1 so both x and shifted y fit.
    ix = torch.randint(
        low=0,
        high=len(data) - block_size - 1,
        size=(batch_size,),
    )

    # Build input batch x.
    #
    # Example:
    #
    #     data[i:i+block_size] = [10, 11, 12, 13]
    #
    # becomes one row of x.
    x = torch.stack([
        torch.from_numpy(
            data[i:i + block_size].astype(np.int64)
        )
        for i in ix.tolist()
    ])

    # Build target batch y.
    #
    # Same chunk shifted one token forward:
    #
    #     x = [10, 11, 12, 13]
    #     y = [11, 12, 13, 14]
    #
    # The model learns next-token prediction.
    y = torch.stack([
        torch.from_numpy(
            data[i + 1:i + block_size + 1].astype(np.int64)
        )
        for i in ix.tolist()
    ])

    # Move tensors to CPU, CUDA, or MPS.
    x = x.to(device)
    y = y.to(device)

    return x, y