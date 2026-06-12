import torch
import torch.nn as nn
from torch.nn import functional as F


class BigramLanguageModel(nn.Module):
    """Simplest possible language model. This model predicts the next token using only the current token.
    It has no attention, no hidden layers, and no context mixing."""

    def __init__(self, vocab_size):
        super().__init__()

        # Lookup table:
        # input token ID -> logits for next token
        # Shape: (vocab_size, vocab_size)
        # Example: if current token is 10, we read row 10.
        # That row contains scores for every possible next token.
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        """Forward pass.

        idx:
            Tensor of token IDs with shape (B, T)
            B = batch size
            T = sequence length

        targets:
            Optional tensor of correct next-token IDs with shape (B, T)

        returns:
            logits: model predictions
            loss: cross-entropy loss if targets are provided, otherwise None
        """

        # Convert each token ID into prediction scores for the next token.
        # idx shape = (B, T) -> logits shape = (B, T, vocab_size)
        logits = self.token_embedding_table(idx)

        if targets is None:
            loss = None

        else:
            # PyTorch cross_entropy expects:
            # logits shape  = (N, C)
            # targets shape = (N,)
            #
            # But currently:
            # logits shape  = (B, T, C)
            # targets shape = (B, T)
            #
            # So we flatten batch and time together:
            # B*T becomes the total number of predictions.
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)

            # Measures how well the model predicted the true next tokens.
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        """Generate new tokens autoregressively.

        idx:
            Current context of token IDs, shape (B, T)

        max_new_tokens:
            Number of new tokens to generate
        """

        for _ in range(max_new_tokens):

            # Get model predictions for the whole current sequence.
            logits, _ = self(idx)

            # Use only the final time step.
            # That is the model's prediction for the next token.
            #
            # Before: (B, T, vocab_size)
            # After:  (B, vocab_size)
            logits = logits[:, -1, :]

            # Convert raw logits into probabilities.
            probs = F.softmax(logits, dim=-1)

            # Sample one next token from the probability distribution.
            #
            # Shape:
            # (B, 1)
            idx_next = torch.multinomial(probs, num_samples=1)

            # Append the sampled token to the sequence.
            #
            # Old idx: (B, T)
            # New idx: (B, T+1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
