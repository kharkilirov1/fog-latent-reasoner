# Preregistered matched lookup protocol

Date locked: 2026-08-12, before observing any structured-row training result.

## Question

Does the current FOG planner/memory interface lose dynamic key/value bindings,
or did the earlier one-hop failure come from small-model optimization and the
token serialization?

## Data contract

- Each operator is a uniformly sampled permutation of eight states.
- Train, validation, and test are disjoint by the permutation itself, not by
  query, row order, RNG seed, or complete serialized example.
- Table rows are independently reordered; the target is the value paired with
  the queried key.
- Validation is used for calibration. Test remains untouched until the
  configuration, step budget, and seeds are locked.

## Two gates

1. **Token gate.** Explicit token rows with a shared state-token space. This is
   an ecologically closer causal-LM test, but it requires the small decoder to
   discover both within-row binding and content-addressed retrieval.
2. **Structured-row gate.** Each row is presented as one continuous vector made
   from key, value, and role embeddings. This removes token-level binding as a
   confound and directly tests whether the FOG latent bottleneck preserves an
   already formed key/value record.

Both compare a direct Transformer path, a FOG full-context path, and a strict
FOG path whose answer decoder receives only neutral lexical input plus latent
memory. Shared tensors use name-stable paired initialization, and every arm
receives identical examples in identical order.

## Locked decision rules

Chance accuracy is 12.5%.

- A direct arm passes at >=95% exact accuracy on operator-disjoint test tables.
- Strict FOG passes at >=90% test accuracy, with worst-seed accuracy >=85%.
- A successful strict checkpoint must lose >=50 percentage points when its
  memory is replaced by target-deranged cross-example memory; zeroed and
  shuffled memory should be <=20%.
- **Direct pass, full/strict pass:** earlier failure was serialization or
  optimization, not evidence against the latent interface.
- **Direct pass, full pass, strict fail:** the current latent bottleneck loses
  binding information.
- **Direct pass, both FOG arms fail:** the current planner/allocation or its
  optimization is the leading cause.
- **All direct controls fail:** inconclusive about FOG; model/task optimization
  must be repaired before making an architectural claim.

The conclusion applies to this implementation and training protocol, not to
latent reasoning in general.

## Calibration already observed before this lock

The former implicit-position experiment was not truly unseen: 96.19% of its
evaluation examples reused a training permutation. In the corrected token
gate, a direct 105,856-parameter decoder stayed at 12.55% validation accuracy
after 1,000 steps. A fixed-batch control reached approximately 100%, proving
that forward/backward and optimizer plumbing work, but it did not establish
algorithmic generalization. The structured-row gate was introduced because the
token gate therefore cannot by itself distinguish FOG from the baseline.
