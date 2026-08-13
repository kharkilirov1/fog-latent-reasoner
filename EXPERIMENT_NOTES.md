# Training notes and honest scope

## What was changed before training

The uploaded repository was an inference/loss skeleton: it had no dataset,
optimizer loop, evaluation, checkpointing, or standard `forward()` method. The
following blocking issues were corrected:

- all embeddings and projections now use a small-model initialization;
  previously the tied unit-variance embedding/head produced CE around 66 for a
  1024-token vocabulary instead of the random baseline `ln(1024) ≈ 6.93`;
- reasoning depth can be supplied per call, total sequence length is checked,
  and a standard `forward()` entry point is available;
- route entropy is computed in FP32, generation restores train/eval mode safely,
  and fully empty compare masks are rejected;
- the planner's learned slots receive a direct plus learned continuous summary
  of `(mean(context), last_context)`. Without that content path, the nominal
  thought queries were almost fixed and learned lookup poorly;
- an optional separate decoder prompt enables a strict latent-memory bottleneck.

## Negative experiment that informed the fix

The first task placed three fresh shuffled 8-state permutation tables and an
operator program in every example. After 800 steps the recurrent model remained
at chance: 12.84% test-ID and 13.43% on longer programs (chance is 12.5%). Zeroing
or shuffling memory did not change its predictions. It could overfit one fixed
batch to 100%, so this was a generalization/interface failure rather than a
broken backward pass.

That failed run exposed two shortcuts/bottlenecks:

1. fixed latent queries did not receive a strong content-dependent problem
   state;
2. the final decoder could ignore memory and read the original prompt directly.

The final experiment therefore asks a narrower mechanistic question and blocks
the direct lexical path.

## Final finite-state experiment

The lexical prompt contains only `(operator, start_state)`. It deliberately omits
hop count. Hop count is supplied solely as recurrent depth `R`, and the final
decoder sees only a neutral task token plus four latent memory slots. Only the
final answer receives supervision; no intermediate state is decoded.

Seed-0 results for the selected 60,675-parameter checkpoints:

| variant | depths 1–4 | unseen depths 5–8 |
|---|---:|---:|
| recurrent `R=L` | **100.00%** | **75.00%** |
| identical one-shot `R=1` | 25.00% | 8.33% |

The one-shot information-theoretic ceiling is 25% because an identical prompt
has four different depth-dependent targets. Memory interventions on the
recurrent checkpoint give:

- normal memory, depths 1–4: 100%;
- zero memory: 12.5%;
- memory shuffled across examples: 0%;
- unseen-depth targets forced to `R=1`: 8.33%.

This demonstrates end-to-end trainability, genuine use of example-specific
latent memory, and partial recurrence-depth extrapolation. It does **not** show
that the model can yet read arbitrary operator tables or outperform text CoT,
Coconut, looping, or a matched Transformer. Accuracy also decays from 87.5% at
unseen depth 5 to 54.17% at depth 8, so the learned transition is not perfectly
stable under indefinite iteration.
