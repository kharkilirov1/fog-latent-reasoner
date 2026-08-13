# Dataset contract and attribution

The training code defaults to pinned revisions and never bundles either full
dataset or trains on the official GSM8K test split.

| stage | dataset | config | train / evaluation split | pinned revision | license |
|---|---|---|---|---|---|
| lexical pretraining | `roneneldan/TinyStories` | `default` | `train` / `validation` | `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64` | CDLA-Sharing-1.0 |
| reasoning SFT | `openai/gsm8k` | `main` | `train` / `test` | `cc7b047b6e5bb11b4f1af84efc572db110a51b3c` | MIT |

TinyStories is streamed by default. Empty text rows are ignored. Its synthetic
stories are useful for a small English model but are not representative of
broad natural-language pretraining.

For GSM8K, 512 deterministic examples are carved out of the official `train`
split for checkpoint selection. The official `test` split is loaded only by
the explicit `evaluate-gsm8k` command. `target_mode=final` extracts the text
after the final `####`; `target_mode=full` is available as a textual-chain-of-
thought ablation.

Before redistributing dataset-derived artifacts, review the upstream licenses:

- <https://huggingface.co/datasets/roneneldan/TinyStories>
- <https://huggingface.co/datasets/openai/gsm8k>

The small files under `examples/` are original synthetic smoke fixtures, not
rows copied from either dataset.
