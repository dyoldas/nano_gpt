from .bigram import BigramLanguageModel
from .nano_gpt import GPTLanguageModel


# ============================================================
# Model creation
# ============================================================

def create_model(config, vocab_size):
    """Create the model specified by the config file.

    model_type can be:

        "bigram" -> simplest baseline model
        "gpt"    -> transformer model
    """

    if config["model_type"] == "bigram":

        # Bigram model only learns:
        #
        #     current token -> next token distribution
        #
        # It has no attention, no context window, and no token mixing.
        return BigramLanguageModel(
            vocab_size=vocab_size,
        )

    if config["model_type"] == "gpt":

        # GPT model learns from a context window of length block_size.
        return GPTLanguageModel(
            vocab_size=vocab_size,
            block_size=config["block_size"],
            n_embd=config["n_embd"],
            n_head=config["n_head"],
            n_layer=config["n_layer"],
            dropout=config["dropout"],
        )

    raise ValueError(f"Unknown model_type: {config['model_type']}")