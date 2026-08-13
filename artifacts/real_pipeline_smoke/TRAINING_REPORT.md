# Real-pipeline smoke report

This is an execution check of the packaged 10M training path, not a model-quality
claim. The complete values are in [metrics.json](metrics.json).

## What passed

- The initialized checkpoint contains exactly **10,035,848** unique trainable
  parameters and reloads outside `train_real.py`.
- Token embedding and language head are tied and occur once in the optimizer.
- A full causal-LM optimizer update at sequence length 64 was finite; initial
  loss was 9.0595, close to `ln(8192) = 9.0109`.
- The saved pretraining state resumed at optimizer step 2 with the dataset
  cursor, AdamW state, scheduler, and RNG restored.
- The native latent SFT route (`R=4`, strict memory-only decoder) completed an
  optimizer step with finite loss and gradients. The learned compressor is
  active at recurrent steps 3 and 4.
- The full automated suite covers padding, masked loss, compression gradients,
  exact parameter count, data collators, and bit-exact checkpoint resume.

## Data access note

The official TinyStories and GSM8K repository IDs, revisions, schemas, and
split contracts were verified. This sandbox did not grant the Python process
outbound download access, so the executable optimizer smoke used the original
synthetic fixtures under `examples/`. The same loader code supports the pinned
Hugging Face sources and local TXT/JSONL without changing the model path.

The bundled tokenizer is a lossless byte fallback padded with reserved IDs to
the exact 8,192-token geometry. It makes the archive runnable offline. A serious
training run should first execute the corpus BPE command in `README_RU.md`; it
is substantially more token-efficient.
