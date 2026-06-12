import torch
import torch.nn as nn
from torch.nn import functional as F


class Head(nn.Module):
    """One head of masked self-attention."""

    def __init__(self, n_embd, head_size, block_size, dropout):
        super().__init__()

        # These linear layers project the input embedding into:
        # key   = "what information do I contain?"
        # query = "what information am I looking for?"
        # value = "what information do I pass forward?"
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        # Lower-triangular causal mask.
        # This prevents each token from attending to future tokens.
        #
        # Example for block_size = 4:
        # [[1, 0, 0, 0],
        #  [1, 1, 0, 0],
        #  [1, 1, 1, 0],
        #  [1, 1, 1, 1]]
        #
        # register_buffer means:
        # - this tensor moves with the model to CPU/GPU/MPS
        # - but it is not a trainable parameter
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

        # Dropout randomly zeroes some attention weights during training.
        # This regularizes the model and helps prevent overfitting.
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape:
        # (B, T, C)
        #
        # B = batch size
        # T = sequence length
        # C = embedding dimension, n_embd
        B, T, C = x.shape

        # Project input embeddings into keys, queries, and values.
        #
        # k, q, v shape:
        # (B, T, head_size)
        k = self.key(x)
        q = self.query(x)

        # Compute attention scores.
        # q @ k.transpose(-2, -1):
        # (B, T, head_size) @ (B, head_size, T) -> (B, T, T)
        #
        # Each position gets a score for every other position.
        #
        # The scaling factor prevents dot products from becoming too large,
        # which keeps softmax numerically stable.
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5

        # Apply causal mask.
        # Future positions receive -inf, so softmax turns them into probability 0.
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))

        # Convert attention scores into attention probabilities.
        # For each token, probabilities over previous/current tokens sum to 1.
        wei = F.softmax(wei, dim=-1)

        # Apply dropout to attention probabilities.
        wei = self.dropout(wei)

        # Project input embeddings into values.
        v = self.value(x)

        # Weighted aggregation of value vectors.
        # wei shape: (B, T, T)
        # v shape:   (B, T, head_size)
        # out shape: (B, T, head_size)
        out = wei @ v

        return out


class MultiHeadAttention(nn.Module):
    """Multiple self-attention heads running in parallel."""

    def __init__(self, n_embd, num_heads, head_size, block_size, dropout):
        super().__init__()

        # Each head learns a different way for tokens to communicate.
        self.heads = nn.ModuleList([
            Head(n_embd, head_size, block_size, dropout)
            for _ in range(num_heads)
        ])

        # After concatenating all heads, project back to n_embd.
        self.proj = nn.Linear(head_size * num_heads, n_embd)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Run every attention head independently and concatenate results.
        #
        # Each head output: (B, T, head_size)
        # Concatenated:     (B, T, head_size * num_heads)
        out = torch.cat([head(x) for head in self.heads], dim=-1)

        # Mix information from all heads and return to embedding dimension.
        out = self.proj(out)
        out = self.dropout(out)

        return out


class FeedForward(nn.Module):
    """Token-wise MLP after attention."""

    def __init__(self, n_embd, dropout):
        super().__init__()

        # Applied independently to each token position.
        #
        # Shape:
        # (B, T, n_embd) -> (B, T, 4*n_embd) -> (B, T, n_embd)
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """One Transformer block: attention + feedforward."""

    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()

        # Each attention head gets an equal slice of the embedding dimension.
        head_size = n_embd // n_head

        # Communication step:
        # tokens exchange information through masked self-attention.
        self.sa = MultiHeadAttention(
            n_embd=n_embd,
            num_heads=n_head,
            head_size=head_size,
            block_size=block_size,
            dropout=dropout,
        )

        # Computation step:
        # each token processes its own updated representation.
        self.ffwd = FeedForward(n_embd, dropout)

        # LayerNorm stabilizes training.
        # This is the "pre-norm" transformer style:
        # normalize before attention/feedforward.
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # Residual connection around attention.
        x = x + self.sa(self.ln1(x))

        # Residual connection around feedforward.
        x = x + self.ffwd(self.ln2(x))

        return x


class GPTLanguageModel(nn.Module):
    """Small decoder-only GPT language model."""

    def __init__(self, vocab_size, block_size, n_embd, n_head, n_layer, dropout):
        super().__init__()

        self.block_size = block_size

        # Token embedding:
        # token ID -> learned vector of size n_embd
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)

        # Positional embedding:
        # position 0, 1, ..., block_size-1 -> learned vector of size n_embd
        #
        # This tells the model where each token is in the sequence.
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        # Stack of Transformer blocks.
        self.blocks = nn.Sequential(*[
            Block(n_embd, n_head, block_size, dropout)
            for _ in range(n_layer)
        ])

        # Final normalization before projecting to vocabulary logits.
        self.ln_f = nn.LayerNorm(n_embd)

        # Language-modeling head:
        # hidden vector -> logits over vocabulary
        self.lm_head = nn.Linear(n_embd, vocab_size)

        # Initialize weights with small random values.
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize Linear and Embedding weights."""

        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        """Forward pass.

        idx:
            Tensor of token IDs with shape (B, T)

        targets:
            Optional tensor of next-token labels with shape (B, T)

        returns:
            logits: shape (B, T, vocab_size)
            loss: cross-entropy loss if targets are provided, otherwise None
        """

        B, T = idx.shape

        # Safety check:
        # this model cannot process sequences longer than block_size.
        if T > self.block_size:
            raise ValueError(
                f"Sequence length {T} exceeds block_size {self.block_size}."
            )

        # Token embeddings:
        # (B, T) -> (B, T, n_embd)
        #
        # Assume B = 1.
        # E is the embedding matrix of shape (vocab_size, n_embd).
        # x_i is the token ID at position i in the sequence.
        # If x_i = j, then the embedding vector at position i is
        # e_i = E[j]
        tok_emb = self.token_embedding_table(idx)

        # Position embeddings:
        # positions shape: (T,)
        # pos_emb shape:  (T, n_embd)
        #
        # idx.device makes this work on CPU, CUDA, or MPS without global device.
        positions = torch.arange(T, device=idx.device)
        pos_emb = self.position_embedding_table(positions)

        # Add token identity and token position information.
        # tok_emb: (B, T, n_embd)
        # pos_emb: (T, n_embd)
        #
        # PyTorch broadcasts pos_emb over the batch dimension.
        x = tok_emb + pos_emb

        # Pass through transformer blocks.
        x = self.blocks(x)

        # Final layer norm.
        x = self.ln_f(x)

        # Convert hidden states into vocabulary logits.
        #
        # logits shape:
        # (B, T, vocab_size)
        logits = self.lm_head(x)

        if targets is None:
            loss = None

        else:
            # PyTorch cross_entropy expects:
            # logits shape  = (N, C)
            # targets shape = (N,)
            #
            # Current shapes:
            # logits  = (B, T, vocab_size)
            # targets = (B, T)
            #
            # Flatten batch and time:
            # N = B*T
            B, T, C = logits.shape
            logits_flat = logits.view(B * T, C)
            targets_flat = targets.view(B * T)

            loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        """Generate tokens autoregressively.

        idx:
            Current context, shape (B, T)

        max_new_tokens:
            Number of new tokens to append
        """

        for _ in range(max_new_tokens):

            # Crop context to the maximum context length.
            #
            # If idx is longer than block_size, the model only sees
            # the most recent block_size tokens.
            idx_cond = idx[:, -self.block_size:]

            # Forward pass on the current context.
            logits, _ = self(idx_cond)

            # Keep only the logits at the final time step.
            #
            # Before:
            # logits shape = (B, T, vocab_size)
            #
            # After:
            # logits shape = (B, vocab_size)
            logits = logits[:, -1, :]

            # Convert logits into probabilities.
            probs = F.softmax(logits, dim=-1)

            # Sample one token from the probability distribution.
            #
            # idx_next shape:
            # (B, 1)
            idx_next = torch.multinomial(probs, num_samples=1)

            # Append sampled token to the running sequence.
            #
            # Old idx shape: (B, T)
            # New idx shape: (B, T+1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
