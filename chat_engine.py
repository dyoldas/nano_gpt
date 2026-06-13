import os
import pickle

import torch

from models import create_model
from utils import get_device, load_checkpoint
from generate import sample_from_model


class ChatEngine:
    """Backend wrapper for the local nanoGPT chat app.
    Responsibilities:
        1. Load checkpoint.
        2. Load tokenizer metadata.
        3. Format chat prompts.
        4. Call the model generation function.
        5. Clean the model output for UI display.

    The Gradio UI should only call:
        engine.respond(...)
    """

    def __init__(self, ckpt_path, device="auto"):
        # Select device: "mps", "cuda", or "cpu".
        self.device = get_device(device)

        # Load checkpoint.
        self.ckpt = load_checkpoint(ckpt_path, self.device)
        self.config = self.ckpt["config"]

        # Locate tokenizer metadata.
        # Example:
        #   dataset: instruct_mix
        #
        # gives:
        #   data/instruct_mix/meta.pkl
        data_dir = os.path.join("data", self.config["dataset"])
        meta_path = os.path.join(data_dir, "meta.pkl")

        with open(meta_path, "rb") as f:
            meta = pickle.load(f)

        self.stoi = meta["stoi"]
        self.itos = meta["itos"]
        self.vocab_size = meta["vocab_size"]

        # Recreate model architecture from checkpoint config.
        self.model = create_model(self.config, self.vocab_size)

        # Load trained weights.
        self.model.load_state_dict(self.ckpt["model"])

        # Move model to device and disable dropout.
        self.model = self.model.to(self.device)
        self.model.eval()

    def encode(self, text):
        """Convert text into token IDs.
        Unknown characters are skipped because this is a character-level model.
        """
        return [self.stoi[c] for c in text if c in self.stoi]

    def decode(self, ids):
        """Convert token IDs back into text."""
        return "".join(self.itos[i] for i in ids)

    def build_prompt(self, user_message, history=None):
        """Build prompt exactly like generate.py.
        For the current small model, we ignore chat history.
        This makes the app behave like terminal generation.
        """

        if self.config["dataset"] == "instruct_mix":
            return f"""### Instruction:
            {user_message}
            
            ### Response:
            """

        return user_message

    def clean_response(self, text):
        """Cut off unwanted continuation after the response."""

        stop_markers = [
            "<|end|>",
            "\n### User:",
            "\n### Instruction:",
            "\n### System:",
        ]

        for marker in stop_markers:
            if marker in text:
                text = text.split(marker, 1)[0]
                break

        return text.strip()

    def generate_response(
        self,
        prompt,
        max_new_tokens=300,
        temperature=0.8,
        top_k=40,
    ):
        """Generate only the assistant's new response.

        sample_from_model(...) returns prompt + generated continuation.
        We remove the prompt part and keep only the new generated tokens.
        """

        prompt_ids = self.encode(prompt)

        # If prompt is empty after encoding, use token 0.
        if len(prompt_ids) == 0:
            prompt_ids = [0]

        start_ids = torch.tensor(
            [prompt_ids],
            dtype=torch.long,
            device=self.device,
        )

        with torch.no_grad():
            full_ids = sample_from_model(
                model=self.model,
                start_ids=start_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                stop_id=None,
            )[0].tolist()

        # Remove prompt tokens so UI only displays the assistant continuation.
        new_ids = full_ids[len(prompt_ids) :]

        raw_text = self.decode(new_ids)
        clean_text = self.clean_response(raw_text)

        return clean_text

    def respond(
        self,
        user_message,
        history,
        temperature=0.8,
        top_k=40,
        max_new_tokens=300,
    ):
        """Main function called by app.py.

        Input:
            user_message: current user message
            history: previous Gradio chat history

        Output:
            updated history
        """

        if history is None:
            history = []

        user_message = user_message.strip()

        if user_message == "":
            return history

        prompt = self.build_prompt(
            user_message=user_message,
            history=history,
        )

        assistant_message = self.generate_response(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )

        if assistant_message == "":
            assistant_message = "[empty generation]"

        # Gradio Chatbot with type="messages" expects dictionaries.
        history = history + [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_message},
        ]

        return history
