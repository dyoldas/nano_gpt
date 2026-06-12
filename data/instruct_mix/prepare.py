import os
import pickle
import random
import numpy as np
from datasets import load_dataset
from tqdm import tqdm

N_DOLLY = 15_000
N_SMOL = 20_000
VAL_FRACTION = 0.02
SEED = 1337

out_dir = os.path.dirname(__file__)
input_path = os.path.join(out_dir, "input.txt")
pretrain_meta_path = os.path.join("data", "tiny_cosmo", "meta.pkl")

random.seed(SEED)

# Load the tokenizer metadata
with open(pretrain_meta_path, "rb") as f:
    meta = pickle.load(f)

stoi = meta["stoi"]
itos = meta["itos"]
vocab_size = meta["vocab_size"]

def clean(s):
    """Unix style newline and remove whitespaces"""
    if s is None:
        return ""
    return str(s).replace("\r\n", "\n").replace("\r", "\n").strip()

def encode_with_existing_vocab(s):
    """Encode using the pretraining tokenizer"""
    # Important: keep same vocab as pretraining so checkpoint can resume.
    # Unknown chars are skipped.
    return [stoi[c] for c in s if c in stoi]

def format_dolly(ex):
    """Format Dolly data"""
    instruction = clean(ex["instruction"])
    context = clean(ex.get("context", "")) # context if it exists, otherwise uses empty string
    response = clean(ex["response"])

    # <|end|> teaches the model to stop
    if context:
        return f"""### Instruction:
{instruction}

### Context:
{context}

### Response:
{response}

<|end|>
"""
    else:
        return f"""### Instruction:
{instruction}

### Response:
{response}

<|end|>
"""

def format_messages(messages):
    """Format SmolTalk examples"""
    parts = []
    for m in messages:
        role = clean(m.get("role", ""))
        content = clean(m.get("content", ""))

        if not content:
            continue

        if role == "user":
            parts.append(f"### User:\n{content}")
        elif role == "assistant":
            parts.append(f"### Assistant:\n{content}")
        elif role == "system":
            parts.append(f"### System:\n{content}")

    if not parts:
        return ""

    return "\n\n".join(parts) + "\n\n<|end|>\n"


def get_dolly():
    """Load and yield formatted Dolly examples one at a time"""
    print("Loading Dolly...")

    dolly = load_dataset("databricks/databricks-dolly-15k", split="train")
    dolly = dolly.shuffle(seed=SEED)

    for ex in tqdm(dolly.select(range(min(N_DOLLY, len(dolly)))), desc="Dolly"):
        t = format_dolly(ex)
        if t:
            yield t

def get_smoltalk():
    """Load and yield formatted SmolTalk conversations one at a time"""
    print("Loading Smol-SmolTalk...")

    smol = load_dataset(
        "HuggingFaceTB/smol-smoltalk",
        split="train",
        streaming=True,
    )

    smol = smol.shuffle(seed=SEED + 1, buffer_size=10_000)

    count = 0
    for ex in tqdm(smol, total=N_SMOL, desc="Smol-SmolTalk"):
        t = format_messages(ex["messages"])

        if t:
            yield t
            count += 1

        if count >= N_SMOL:
            break


print("Writing finetuning input.txt...")

# Open the output file once. Everything we yield will be written here.
with open(input_path, "w", encoding="utf-8") as f:
    dolly_gen = get_dolly()
    smol_gen = get_smoltalk()

    remaining_dolly = N_DOLLY
    remaining_smol = N_SMOL

    for _ in range(remaining_dolly + remaining_smol):
        use_dolly = (random.random() < 0.4 and remaining_dolly > 0) or remaining_smol == 0

        if use_dolly:
            t = next(dolly_gen)
            remaining_dolly -= 1
        else:
            t = next(smol_gen)
            remaining_smol -= 1

        f.write(t)
        f.write("\n\n")


print("Reading finetuning input.txt...")
with open(input_path, "r", encoding="utf-8") as f:
    data = f.read()

print("Total finetune characters:", len(data))

# Encode with pretraining encoder and convert to numpy array
ids = np.array(encode_with_existing_vocab(data), dtype=np.uint16)

# Split training and validation splits and write to disk
n = len(ids)
split_idx = int(n * (1 - VAL_FRACTION))

train_ids = ids[:split_idx]
val_ids = ids[split_idx:]

train_ids.tofile(os.path.join(out_dir, "train.bin"))
val_ids.tofile(os.path.join(out_dir, "val.bin"))

# Copy same tokenizer metadata from pretraining.
with open(os.path.join(out_dir, "meta.pkl"), "wb") as f:
    pickle.dump(meta, f)

print("Done.")