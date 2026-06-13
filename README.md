# nano_gpt

## Author: Deniz Yoldas
A small educational language-modeling project inspired by Andrej Karpathy's nanoGPT / makemore lectures.

The goal of this project is to build a complete character-level language modeling pipeline from scratch:

1. Start with a simple Bigram model.
2. Build a small decoder-only Transformer.
3. Pretrain it on TinyStories + Cosmopedia.
4. Finetune it on instruction-style datasets.
5. Generate text from trained checkpoints.

This project is mainly for understanding how language models work internally, not for building a production-quality chatbot.

---

## Project Structure

```text
nano_gpt/
├── README.md
├── app.py
├── chat_engine.py
├── train.py
├── generate.py
├── requirements.txt
├── configs/
│   ├── bigram.yaml
│   ├── pretrain.yaml
│   └── finetune.yaml
├── data/
│   ├── shakespeare/
│   │   └── input.txt
│   ├── tiny_cosmo/
│   │   └── prepare.py
│   └── instruct_mix/
│       └── prepare.py
├── models/
│   ├── __init__.py
│   ├── bigram.py
│   ├── nano_gpt.py
│   └── create.py
├── utils/
│   ├── __init__.py
│   ├── config.py
│   ├── device.py
│   ├── data.py
│   ├── eval.py
│   └── checkpointing.py
├── checkpoints/
│   ├── bigram/
│   ├── tiny_cosmo/
│   └── instruct_mix/
└── notebooks/
    └── walkthrough.ipynb
```

---

## Local Desktop Chat Interface

This project includes a minimal native desktop chat interface built with Tkinter.

The interface lets you interact with a trained checkpoint through a simple local window. It is not a real ChatGPT-scale assistant; it is a UI wrapper around the small character-level nanoGPT model.

The desktop app uses:

* `app.py` for the native Tkinter window
* `chat_engine.py` for loading the model, formatting prompts, generating text, and cleaning responses
* `generate.py`'s sampling function for autoregressive generation

For instruction-finetuned checkpoints, the app formats user messages as:

```text
### Instruction:
<user message>

### Response:
```

This matches the instruction finetuning format used by the project.

### Run the Desktop Chat App

Use the finetuned checkpoint:

```bash
python app.py --ckpt checkpoints/instruct_mix/ckpt.pt
```

Use a specific device:

```bash
python app.py --ckpt checkpoints/instruct_mix/ckpt.pt --device mps
```

Use the pretrained checkpoint for raw story/text continuation:

```bash
python app.py --ckpt checkpoints/tiny_cosmo/ckpt.pt
```

***If Tkinter does not work try:***
```bash
conda install -c conda-forge tk
```

### Notes on Model Quality

The chat interface only makes interaction easier. Expected behavior:

* the pretrained model is better for raw text/story continuation
* the finetuned model follows the instruction format better
* the model may still repeat phrases or produce weak answers because it is small, character-level, and trained on limited/noisy data

For best results, use short and simple prompts with conservative sampling settings.

---

## Models

### Bigram Model

The Bigram model is the simplest baseline.

It learns:

```text
current token -> next token probability distribution
```

It has no attention, no context window, and no hidden reasoning. It is useful as a sanity check before training the Transformer.

### GPT / Transformer Model

The GPT model is a small decoder-only Transformer with:

* token embeddings
* positional embeddings
* masked self-attention
* multi-head attention
* feedforward layers
* residual connections
* layer normalization
* autoregressive generation

The model predicts the next character/token given a previous context window.

---

## Datasets

### Pretraining

Pretraining uses a mixture of:

```text
TinyStories + Cosmopedia
```

This is prepared by:

```bash
python data/tiny_cosmo/prepare.py
```

This creates:

```text
data/tiny_cosmo/input.txt
data/tiny_cosmo/train.bin
data/tiny_cosmo/val.bin
data/tiny_cosmo/meta.pkl
```

### Finetuning

Instruction finetuning uses:

```text
Dolly + SmolTalk / Smol-SmolTalk style conversations
```

This is prepared by:

```bash
python data/instruct_mix/prepare.py
```

This creates:

```text
data/instruct_mix/input.txt
data/instruct_mix/train.bin
data/instruct_mix/val.bin
data/instruct_mix/meta.pkl
```

The finetuning tokenizer metadata is copied from pretraining so the pretrained checkpoint can be reused safely.

---

## Setup

Activate the conda environment:

```bash
conda activate ml
```

Install required packages by simpy:

```bash
pip install -r requirements.txt
```

or

```bash
pip install torch numpy datasets tqdm pyyaml
```

---

## Prepare Data

### Pretraining data

```bash
python data/tiny_cosmo/prepare.py
```

Check output:

```bash
ls -lh data/tiny_cosmo
```

Expected files:

```text
input.txt
train.bin
val.bin
meta.pkl
prepare.py
```

### Finetuning data

```bash
python data/instruct_mix/prepare.py
```

Check output:

```bash
ls -lh data/instruct_mix
```

Expected files:

```text
input.txt
train.bin
val.bin
meta.pkl
prepare.py
```

---

## Training

### Bigram baseline

```bash
python train.py --config configs/bigram.yaml
```

### GPT pretraining

Short test run:

```bash
python train.py --config configs/pretrain.yaml --max_iters 100 --eval_interval 20
```

Full pretraining run:

```bash
python train.py --config configs/pretrain.yaml
```

On macOS, to prevent sleep during long training:

```bash
caffeinate -dimsu python train.py --config configs/pretrain.yaml
```

### GPT instruction finetuning

Make sure this checkpoint exists first:

```text
checkpoints/tiny_cosmo/ckpt.pt
```

Then run:

```bash
python train.py --config configs/finetune.yaml
```

or with sleep prevention:

```bash
caffeinate -dimsu python train.py --config configs/finetune.yaml
```

The finetuned checkpoint is saved to:

```text
checkpoints/instruct_mix/ckpt.pt
```

---

## Generation

### Generate from pretrained model

```bash
python generate.py \
--ckpt checkpoints/tiny_cosmo/ckpt.pt \
--prompt "Once upon a time" \
--max_new_tokens 500
```

More conservative generation:

```bash
python generate.py \
--ckpt checkpoints/tiny_cosmo/ckpt.pt \
--prompt "The little dragon" \
--temperature 0.8 \
--top_k 20
```

### Generate from finetuned model

```bash
python generate.py \
--ckpt checkpoints/instruct_mix/ckpt.pt \
--prompt "Explain gravity simply."
```

Creative generation:

```bash
python generate.py \
--ckpt checkpoints/instruct_mix/ckpt.pt \
--prompt "Write a short story about a dragon and a knight." \
--temperature 1.1 \
--top_k 40 \
--max_new_tokens 1000
```

Math-style prompt:

```bash
python generate.py \
--ckpt checkpoints/instruct_mix/ckpt.pt \
--prompt "What is the derivative of x^2?" \
--temperature 0.4 \
--top_k 10 \
--max_new_tokens 200
```

---

## Config Files

Training is controlled by YAML files in `configs/`.

Example:

```yaml
dataset: tiny_cosmo
model_type: gpt
init_from: null
out_dir: checkpoints/tiny_cosmo

device: auto
seed: 1337

batch_size: 64
block_size: 256
max_iters: 10000
eval_interval: 500
eval_iters: 100
learning_rate: 0.0003

n_embd: 384
n_head: 6
n_layer: 6
dropout: 0.2
```

Terminal arguments can override YAML values:

```bash
python train.py --config configs/pretrain.yaml --learning_rate 0.0001 --max_iters 2000
```

For more information

```bash
python train.py --help
```

---

## Pretraining vs Finetuning

The same `train.py` handles both.

For pretraining:

```yaml
init_from: null
```

For finetuning:

```yaml
init_from: checkpoints/tiny_cosmo/ckpt.pt
```

If `init_from` is `null`, the model trains from scratch.

If `init_from` points to a checkpoint, the model loads pretrained weights first and continues training on the new dataset.

---

## Notes

This is a small character-level model, not a real ChatGPT-scale model.

Expected learning progression:

1. Bigram learns local character statistics.
2. GPT pretraining learns spelling, grammar, simple stories, and weak world patterns.
3. Instruction finetuning teaches the model prompt/response format.
4. Reliable reasoning and math are not expected from a tiny 10M char-level model.

The purpose of the project is to understand the full language-modeling pipeline from data preparation to training, checkpointing, finetuning, and generation.