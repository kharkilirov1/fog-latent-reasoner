# Latent loop in Qwen3.5-4B (Colab G4, 2026-11-04/05)

Script: `experiments/latent_loop_qwen35_mx6.py` (branch `latent-loop-qwen35`). Colab cells: `colab_cells/`, their full stdout
(and a few per-run logs) in `raw/`. One seed per arm, eval n=64 per depth and style. Task: the pilot's natural-language
modular programs (Z_31), trained depths 1..6. "id" = training phrasings, "held" = held-out phrasing. Chance = 3.2 %.
Optimiser of the decoder linears: AdamW (lr 1e-5) or mx6 latent grid (6-bit, bf16 momentum, sparse+reuse, counter lr 1e-5).

## Bugs found and fixed on the way (results before the fix are NOT evidence)

1. `ll1`: counter lr 3e-4 wrecked the pretrained model (loss 14 at step 20). Fine-tuning needs Adam-scale lr (1e-5);
   3e-5 already collapses late (`ll2` probe).
2. `ll1`/`ll2`: code book initialised from the first token of " v" -- Qwen splits numbers into digits and the first token is
   the space for all 31 values: **31 identical codes, Delta = 0**. Fixed (`orth` codes, Delta = 1, gap logged).
3. `ll2`: CoT eval truncated at depth >= 12 (max_new_tokens). Fixed in `ll3`+.
4. `ll3`: the prompt holds the start value and all operations; the swap control shows the model IGNORES the latent
   (swap_follows_latent 0.03, swap_keeps_true 0.64 at depth 2): it recomputes from the prompt. Fixed by the Markov arms.

## Results that stand

Explicit CoT (text), mx6, fixed eval (`ll5`):

| depth | 1 | 2 | 4 | 8 | 12 | 16 |
|---|---|---|---|---|---|---|
| id | 1.00 | 1.00 | 1.00 | 0.98 | 0.75 | 0.00 |
| held | 0.55 | 0.28 | 0.09 | 0.02 | 0.05 | 0.03 |

Markov latent loop (each step sees ONLY the operations, the step marker and the current latent; start value only as the
latent x_0; text->latent curriculum), AdamW, 800 steps:

| arm | depth 1 | 2 | 4 | 8 | 16 | 32 | swap follows latent (d=2) |
|---|---|---|---|---|---|---|---|
| hard snap (`ll6`) id / held | 0.80 / 0.80 | 0.67 / 0.72 | 0.38 / 0.48 | 0.28 / 0.17 | 0.00 / 0.03 | 0.05 / 0.03 | **0.91 / 0.77** |
| soft capture beta->30 (`ll4`) id | 0.16 | 0.02 | 0.03 | 0.03 | 0.02 | 0.03 | 0.13 |
| hard snap + teacher forcing 0.5, 1200 steps (`ll7`) id | 0.73 | 0.44 | 0.33 | 0.13 | 0.02 | 0.09 | 0.69 |
| hard snap + teacher forcing, **mx6** (`ll7`) id | 0.06 | 0.03 | 0.05 | 0.06 | 0.02 | 0.03 | 0.11 |

Reading:
- Hard snap in the Markov loop is the first arm where the model provably computes THROUGH the latent (swap 0.91).
- Its depth curve follows the one-step envelope p^d (per-step accuracy ~0.85: 0.85^8 = 0.27 vs 0.23-0.28 measured): no
  drift beyond the per-step errors -- the reviewer's reformulation of P2 (capture lifts the curve only to the one-step
  envelope).
- On held-out phrasing the latent loop generalises far better than text CoT (depth 4: 0.48 vs 0.09).
- Soft capture (beta annealed to 30 on cosine read-out logits) does not learn in this setting; snap does.
- Teacher forcing (0.5 -> 0 at 70 %) HURT (one run).
- mx6 in the latent loop: only tested with teacher forcing and one lr; it learned far worse than AdamW. Untested: mx6 + snap
  without teacher forcing, lower counter lr. Not a conclusion about mx6 yet.

## Open

- Raise one-step accuracy (the envelope): longer training, depth-1 emphasis, second seed.
- Mechanistic shaping (Goodfire 2026, expression shaping) as an alternative to the Markov restriction: penalise the start
  value being decodable at thought steps >= 2 in the NON-Markov loop (`--shape-lambda`).
