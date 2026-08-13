# FOG Latent Multi-Workspace Reasoner

This package contains runnable legacy and binding-v2 PyTorch research models at
roughly 10M parameters, a real 8,192-token BPE, trained checkpoints,
reproducible local-data training/evaluation scripts, and mechanistic tests.

The primary checkpoint,
`checkpoints/fog_binding_v2_10m_token_lookup_bf16.pt`, is the
10,000,039-parameter query-bound model. Its SHA-256 is
`d9a5523adc85a049d970f51c2bd75c6f88c64a533d8614077334999a1f4c5960`.
It inherits token embeddings and a four-layer backbone from 1,600 optimizer
steps over a pinned 3,000-story TinyStories slice, then trained one address
sharpness scalar for 40 token-lookup steps. It is a validated `R=1` binding
checkpoint, not a production language model or validated reasoner.

The secondary legacy checkpoint,
`checkpoints/fog_10m_tinystories_1_36m_pretrained_bf16.pt`, contains the original
10,035,848-parameter lexical architecture after consuming 1,358,852 target
tokens. Its subsequent reasoning runs did **not** solve GSM8K or addition.

Release checkpoints are lossy BF16 inference exports of the FP32 training
weights and do not contain optimizer state.

A later size-vs-interface matched experiment found that a 69,184-parameter
ordinary Transformer reached 100.00% and 99.98% on unseen lookup tables in two
of three seeds, while the 139,204-parameter FOG full and strict arms remained
near the 12.5% chance level in all six runs. Both a one-pass new-BOS readout
control and a lossless two-pass hidden-state bridge stayed at chance, localizing
the failure above planner compression to the current readout/interface. The
experiment does not separately identify the effects of BOS, a second pass,
position reset, or memory kind. This is evidence about this implementation, not
an impossibility result for latent reasoning. See
[MATCHED_EXPERIMENT_REPORT_RU.md](MATCHED_EXPERIMENT_REPORT_RU.md).

Frozen probes subsequently refined that diagnosis. Permutation-invariant row
pooling stayed at chance, while exact and ordinary dot-product
query-conditioned row selection were 100% linearly decodable. In contrast,
linear, small-MLP, and learned-attention probes found no generalizing target
signal in the legacy query hidden states, proposals, or persistent memory. The
legacy reader also ignored a perfect oracle answer code until reader-side
adaptation. Thus the failed interface had both a writer/binding problem and a
reader problem.

The separate `query_bound_v2` path addresses both: address before pooling,
payload copying without learned V/O rotation, one protected carrier in a
reusable four-slot workspace, and direct first-token readout without a fresh
BOS retrieval hop. Its frozen
exact-code lookup gate scored **4032/4032 in all three seeds** on the locked
test; a four-digit payload gate scored **4096/4096 exact match in all three
seeds**. Target- and query-deranged interventions scored zero. These are
controlled binding results, not evidence of arithmetic or general reasoning.
The mechanism now also passes the full **10,000,039-parameter** token/backbone
gate: three independent seeds each scored **4032/4032** on locked test, while
zero, target-deranged, and query-cyclic controls each scored **0/4032**.
Address hit was 100%, with 98.42%/96.62%/99.83% mean mass on the correct
address. Only one address-softmax sharpness scalar was trained for 40 steps;
migrated lexical embeddings/backbone and the cosine tied 8,192-way head stayed
frozen. Oracle vocabulary copy was 8192/8192.

Normal NLL remains high at about 6.42 despite perfect top-1 accuracy (uniform
8,192-way NLL is 9.01091), so this is not a calibrated language prediction.
The positive run used one lookup hop and **R=1 only**; it does not establish
recurrent reasoning, relation composition, or arithmetic. See
[BINDING_V2_REPORT_RU.md](BINDING_V2_REPORT_RU.md).

Read [README_RU.md](README_RU.md) for the complete commands and
[MODEL_CARD.md](MODEL_CARD.md) before interpreting results.

## Quick verification

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e '.[dev]'
pytest -q
python verify_release.py
python verify_release.py --forward-backward
```

The default verification target is the primary binding-v2 checkpoint and its
matching `tokenizer/tinystories_3k_bpe.json`; it checks the 10,000,039 parameter
count and both tied vocabulary heads. The secondary legacy contract is
`V=8192, d=320, L=4, H=5, FFN=1344, K=4, R=4, N=8, rank=80,
max_seq_len=512`. Learned memory compression runs on recurrent steps 3 and 4.

The separate v2 reference has 10,000,039 parameters, `K=N=4` reusable slots,
`binding_offsets=(2,)`, a protected primary payload, and cosine tied direct
readout. Its successful token-binding checkpoints were evaluated with `R=1`;
the preset's ability to run deeper is not evidence that `R>1` helps.

`train_real.py init-model` accepts an explicit
`--architecture {legacy_v1,query_bound_v2}`. New v2 training should pass
`--architecture query_bound_v2`; legacy experiment reproduction should pass
`--architecture legacy_v1`. The exact migration, 40-step binding calibration,
locked evaluation, and BF16 export commands are listed in `README_RU.md`.

## Measured result

| run | validation signal | greedy exact match |
|---|---:|---:|
| TinyStories, 400 steps | loss 4.4758, PPL 87.86 | — |
| TinyStories, 1,600 cumulative steps | loss 3.7194, PPL 41.24 | — |
| GSM8K strict memory-only/final | token accuracy 40.14% | 1/128 (0.78%) |
| GSM8K full-prompt/full-CoT | token accuracy 31.85% | 0/128 |
| held-out addition 0…19 | token accuracy 68.56% | 5/80 (6.25%) |

An independent pass over all 300 validation stories measured random-to-final
loss `9.0445 → 3.7529`, perplexity `8471.95 → 42.64`, and token accuracy
`0.0016% → 32.96%`. The 3.7194/41.24 values above are the ten-batch
checkpoint-selection result.

On strict GSM8K, normal, zeroed, and shuffled memory all scored 1/128 and the
model mostly emitted `12`. On held-out addition, normal/zero/shuffled memory
scored 6.25%/5.00%/1.25%. The shuffle drop is weak evidence that memory content
matters, but the task is far from solved. Teacher-forced token accuracy is
inflated by EOS and partial-digit prediction; exact match is decisive here.

Only rows 0–1,023 of the official GSM8K **train** split were used, with a
deterministic 896/128 split. The official test split was not downloaded or
touched. Dataset hashes are recorded in `data_cache/*.manifest.json`; greedy
evaluations are under `artifacts/real_training/`.

The release archive does not redistribute the upstream TinyStories/GSM8K
JSONL rows. It includes their content-hash manifests and the downloader needed
to recreate the same row ranges; synthetic addition data is locally generated.

Therefore this is a **GO for continued architecture/training research and a
NO-GO for claiming a validated reasoner**. Exact reproduction commands,
including the real-dataset downloader, two-stage TinyStories pretraining,
strict SFT, addition gate, and memory interventions, are in the Russian guide.
The complete real-data audit is in
[TRAINING_REPORT_REAL_V2.md](TRAINING_REPORT_REAL_V2.md); the controlled
architecture diagnosis is in
[MATCHED_EXPERIMENT_REPORT_RU.md](MATCHED_EXPERIMENT_REPORT_RU.md), and the
binding-v2 follow-up is in [BINDING_V2_REPORT_RU.md](BINDING_V2_REPORT_RU.md).
