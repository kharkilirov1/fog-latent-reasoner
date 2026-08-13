# FOG finite-state latent-iteration report

The prompt includes `(operator, start state)` but no hop count. The hop count is supplied only as latent recurrence depth `R`; intermediate states are never decoded or supervised. Chance accuracy is 12.5%.

| variant | depths 1–4 | unseen depths 5–8 | parameters | train seconds |
|---|---:|---:|---:|---:|
| recurrent | 98.96% | 9.38% | 58,323 | 95.3 |
| one_shot | 25.00% | 8.33% | 58,323 | 42.2 |

## Memory/depth interventions

- Normal memory, depths 1–4: 98.96%
- Zeroed memory, depths 1–4: 25.00%
- Shuffled-across-example memory, depths 1–4: 81.25%
- Unseen targets 5–8 evaluated with correct R=L: 9.38%
- Same unseen targets forced to R=1: 8.33%
- Same unseen targets capped at R=4: 8.33%

This is a deliberately small mechanistic sanity check. Success on depths 1–4 proves trainability and memory use, but does not by itself establish language reasoning or depth extrapolation.
