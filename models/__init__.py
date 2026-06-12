"""Model package exports.

This file makes the models directory behave like a clean Python package.

Instead of importing from individual files like:

    from models.nano_gpt import GPTLanguageModel
    from models.bigram import BigramLanguageModel

we can import directly from the package:

    from models import GPTLanguageModel, BigramLanguageModel

The actual model and helper definitions still live in:

    models/nano_gpt.py
    models/bigram.py
    models/create.py
"""

from .bigram import BigramLanguageModel
from .nano_gpt import GPTLanguageModel
from .create import create_model


# Controls what gets imported when someone writes:
#
#     from models import *
#
# This is optional, but it makes the package interface explicit.

__all__ = [
    "BigramLanguageModel",
    "GPTLanguageModel",
    "create_model",
]
