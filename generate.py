import os
import pickle
import argparse

import torch

from models import create_model
from utils import get_device, load_checkpoint


# ============================================================
# Sampling utilities
# ============================================================


def sample_from_model(
    model,
    start_ids,
    max_new_tokens,
    temperature=1.0,
    top_k=None,
    stop_id=None,
):
    """Generate tokens autoregressively from a trained model.

    start_ids:
        Tensor of starting token IDs with shape (1, T)

    max_new_tokens:
        Maximum number of new tokens to generate

    temperature:
        Controls randomness.

        temperature < 1.0 -> more conservative
        temperature = 1.0 -> normal
        temperature > 1.0 -> more random

    top_k:
        If set, only sample from the top-k most likely tokens.

    stop_id:
        Optional token ID where generation should stop.
        Useful if your model learns an end marker like <|end|>.
    """

    idx = start_ids

    for _ in range(max_new_tokens):
        # Only feed the model the most recent block_size tokens.
        # If the generated sequence is longer than block_size,
        # older tokens fall out of the context window.
        # GPT has model.block_size but Bigram does not
        if hasattr(model, "block_size"):
            idx_cond = idx[:, -model.block_size :]
        else:
            idx_cond = idx

        # Forwards pass gives logits with shape:
        #     (B, T, vocab_size)
        logits, _ = model(idx_cond)

        # We only care about the final time step.
        # This is the model's prediction for the next token with shape:
        #     (B, vocab_size)
        logits = logits[:, -1, :]

        # Temperature scaling.
        # Dividing by smaller temperature sharpens the distribution.
        # Dividing by larger temperature flattens the distribution.
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        logits = logits / temperature

        # Optional top-k filtering.
        # This prevents sampling from very unlikely tokens.
        if top_k is not None:
            top_k = min(top_k, logits.size(-1))
            values, _ = torch.topk(logits, top_k)

            # Anything below the kth largest logit becomes -inf,
            # so softmax gives it probability 0.
            min_value = values[:, [-1]]
            logits = torch.where(
                logits < min_value,
                torch.full_like(logits, float("-inf")),
                logits,
            )

        # Convert logits into probabilities.
        probs = torch.softmax(logits, dim=-1)

        # Sample one next token.
        idx_next = torch.multinomial(probs, num_samples=1)

        # Append sampled token to running sequence.
        idx = torch.cat((idx, idx_next), dim=1)

        # Optional stopping condition.
        if stop_id is not None and idx_next.item() == stop_id:
            break

    return idx


# ============================================================
# Main generation script
# ============================================================


def main():
    parser = argparse.ArgumentParser()

    # Path to trained checkpoint.
    # Examples:
    #     checkpoints/tiny_cosmo/ckpt.pt
    #     checkpoints/instruct_mix/ckpt.pt
    parser.add_argument("--ckpt", type=str, required=True)

    # Optional prompt.
    parser.add_argument("--prompt", type=str, default="")

    # Generation controls.
    parser.add_argument("--max_new_tokens", type=int, default=500)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=None)

    # Device selection.
    parser.add_argument("--device", type=str, default="auto")

    # If true, stop once the model prints <|end|>.
    parser.add_argument("--stop_at_end", action="store_true", default=True)

    args = parser.parse_args()

    # ------------------------------------------------------------
    # Device
    # ------------------------------------------------------------

    device = get_device(args.device)
    print(f"Using device: {device}")

    # ------------------------------------------------------------
    # Load checkpoint
    # ------------------------------------------------------------

    ckpt = load_checkpoint(args.ckpt, device)

    config = ckpt["config"]

    # ------------------------------------------------------------
    # Locate meta.pkl
    # ------------------------------------------------------------

    # The checkpoint config remembers which dataset was used.
    # Example:
    #     dataset: instruct_mix
    #
    # gives:
    #     data/instruct_mix/meta.pkl
    data_dir = os.path.join("data", config["dataset"])
    meta_path = os.path.join(data_dir, "meta.pkl")

    with open(meta_path, "rb") as f:
        meta = pickle.load(f)

    # encode and decode dictionaries
    stoi = meta["stoi"]
    itos = meta["itos"]
    vocab_size = meta["vocab_size"]

    # ------------------------------------------------------------
    # Encoding / decoding
    # ------------------------------------------------------------

    def encode(s):
        """Convert string into token IDs using training vocabulary.

        Unknown characters are skipped.
        This avoids crashing if the prompt contains a character
        not seen during training.
        """

        return [stoi[c] for c in s if c in stoi]

    def decode(ids):
        """Convert token IDs back into a string."""

        return "".join([itos[i] for i in ids])

    # ------------------------------------------------------------
    # Recreate model architecture
    # ------------------------------------------------------------

    # We recreate the same model architecture from the checkpoint config,
    # then load the saved weights.
    model = create_model(config, vocab_size)
    model.load_state_dict(ckpt["model"])
    model = model.to(device)

    # Evaluation mode disables dropout.
    model.eval()

    # ------------------------------------------------------------
    # Prompt formatting
    # ------------------------------------------------------------

    # Keep the original user prompt separate.
    raw_prompt = args.prompt.strip()

    # If the user gave no prompt, start from token 0.
    # This is useful for raw character-level generation.
    if raw_prompt == "":
        start_ids = [0]

    else:
        # For instruction-finetuned models, wrap plain prompts in the same template used during finetuning.
        # But if the user already wrote their own template containing "###", do not modify it.
        if config["dataset"] == "instruct_mix" and "###" not in raw_prompt:
            prompt = f"""### Instruction:
{raw_prompt}

### Response:
"""

        else:
            prompt = raw_prompt

        start_ids = encode(prompt)

    # idx of shape (1, len(start_ids))
    idx = torch.tensor(
        [start_ids],
        dtype=torch.long,
        device=device,
    )

    # ------------------------------------------------------------
    # Optional stop token
    # ------------------------------------------------------------

    # In a character-level tokenizer, <|end|> is not one token.
    # It is several characters, so we cannot stop on a single token ID.
    # We instead generate normally and trim the decoded text afterward.
    stop_id = None

    # ------------------------------------------------------------
    # Generate
    # ------------------------------------------------------------

    with torch.no_grad():
        out_ids = sample_from_model(
            model=model,
            start_ids=idx,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            stop_id=stop_id,
        )[0].tolist()

    text = decode(out_ids)

    # By default, trim output at <|end|>.
    # This works for our current character-level tokenizer because
    # <|end|> is generated as normal characters.
    end_marker = "<|end|>"
    if end_marker in text:
        # Keep everything up to and including the first <|end|>.
        text = text.split(end_marker, 1)[0] + end_marker

    print("\n" + "=" * 80)
    print(text)
    print("=" * 80)


if __name__ == "__main__":
    main()
