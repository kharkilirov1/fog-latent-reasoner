# FOG finite-state latent-iteration report

The prompt includes `(operator, start state)` but no hop count. The hop count is supplied only as latent recurrence depth `R`; intermediate states are never decoded or supervised. The final decoder sees only a neutral task marker plus latent memory, not the original prompt. Chance accuracy is 12.5%.

| variant | depths 1–4 | unseen depths 5–8 | parameters | train seconds |
|---|---:|---:|---:|---:|
| one_shot | 25.00% | 8.33% | 60,675 | 61.9 |
| recurrent | 100.00% | 75.00% | 60,675 | 123.3 |

## Memory/depth interventions

- Selected checkpoint: optimizer step 1000 (best exhaustive depths 1–4 score).
- Normal memory, depths 1–4: 100.00%
- Zeroed memory, depths 1–4: 12.50%
- Shuffled-across-example memory, depths 1–4: 0.00%
- Unseen targets 5–8 evaluated with correct R=L: 75.00%
- Same unseen targets forced to R=1: 8.33%
- Same unseen targets capped at R=4: 8.33%
- Correct-depth OOD accuracy by length: L=5: 87.50%, L=6: 83.33%, L=7: 75.00%, L=8: 54.17%

This is a deliberately small mechanistic sanity check. Success on depths 1–4 proves trainability and memory use, but does not by itself establish language reasoning or depth extrapolation.
