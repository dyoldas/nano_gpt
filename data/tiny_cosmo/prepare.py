import os
import pickle
import random
import numpy as np
from datasets import load_dataset
from tqdm import tqdm

# Adjust these for dataset size.
# Start small first.
N_TINY = 50_000
N_COSMO = 5_000
MAX_COSMO_CHARS = 3_000
VAL_FRACTION = 0.01
SEED = 1337

out_dir = os.path.dirname(__file__)
input_path = os.path.join(out_dir, "input.txt")

random.seed(SEED)

def clean_text(s):
    """Unix style newline and remove whitespaces"""
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = s.strip()
    return s

def get_tinystories():
    """Import TinyStories dataset"""
    ds = load_dataset(
        "roneneldan/TinyStories",
        split="train",
        streaming=True,
    )

    # Random sampling, a story per buffer of size 10_000
    ds = ds.shuffle(seed=SEED, buffer_size=10_000)

    count = 0
    for ex in ds:
        text = clean_text(ex["text"])

        # Check if the clean text is not empty
        if text:
            # Regular list stores all elements simultaneously in RAM
            # Generator with yield stores only its current state and one element at a time, producing the next element only when requested
            yield text
            count += 1

        if count >= N_TINY:
            break

def get_cosmopedia():
    """Import Cosmopedia dataset"""
    ds = load_dataset(
        "HuggingFaceTB/smollm-corpus",
        "cosmopedia-v2",
        split="train",
        streaming=True,
    )
    ds = ds.shuffle(seed=SEED + 1, buffer_size=10_000)

    count = 0
    for ex in ds:
        text = clean_text(ex["text"])
        if text:
            text = text[:MAX_COSMO_CHARS]
            yield text
            count += 1
        if count >= N_COSMO:
            break


print("Writing pretraining input.txt...")

# Open the output file once. Everything we yield will be written here.
with open(input_path, "w", encoding="utf-8") as f:
    # Create two generators.
    # No stories/articles are loaded yet; these are just "stream handles".
    tiny_gen = get_tinystories()
    cosmo_gen = get_cosmopedia()

    # Write exactly N_TINY + N_COSMO documents, each iteration writes one document to input.txt
    remaining_tiny = N_TINY
    remaining_cosmo = N_COSMO
    for _ in range(remaining_tiny + remaining_cosmo):
        # Choose which dataset to sample from
        # Prefer TinyStories with 70% probability, provided there are TinyStories left
        # If Cosmopedia has already been exhausted, always use TinyStories
        use_tiny = (random.random() < 0.7 and remaining_tiny > 0) or remaining_cosmo == 0

        if use_tiny:
            # Resume the TinyStories generator from its previous yield
            # It runs until the next "yield text" and returns that story.
            t = next(tiny_gen)
            remaining_tiny -= 1

        else:
            # Resume the Cosmopedia generator from its previous yield.
            # It runs until the next "yield text" and returns that article.
            t = next(cosmo_gen)
            remaining_cosmo -= 1

        # Write the current document to disk.
        f.write(t)

        # Separate documents with a blank line.
        # The model can learn that "\n\n" often marks a new document.
        f.write("\n\n")


print("Building char-level tokenizer...")
with open(input_path, "r", encoding="utf-8") as f:
    data = f.read()

# Here are all the unique characters that occur in this text
chars = sorted(list(set(data)))
vocab_size = len(chars)

print("Total characters:", len(data))
print("Vocab size:", vocab_size)

# Create a mapping from characters to integers
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

# Encoder: take a string, output a list of integers
encode = lambda s: [stoi[c] for c in s]

# Decoder: take a list of integers, output a string
decode = lambda l: ''.join([itos[i] for i in l])

# Split data to training and validation splits
n = len(data)
split_idx = int(n * (1 - VAL_FRACTION))
train_data = data[:split_idx]
val_data = data[split_idx:]

# Convert encoded python text into numpy arrays
# Each token ID is stored in 2 bytes, enough for vocabularies up to 65535 tokens
train_ids = np.array(encode(train_data), dtype=np.uint16)
val_ids = np.array(encode(val_data), dtype=np.uint16)

# Write to disk
train_ids.tofile(os.path.join(out_dir, "train.bin"))
val_ids.tofile(os.path.join(out_dir, "val.bin"))

# Save the tokenizer information
meta = {
    "vocab_size": vocab_size,
    "itos": itos,
    "stoi": stoi,
}

with open(os.path.join(out_dir, "meta.pkl"), "wb") as f:
    pickle.dump(meta, f)

print("Done.")