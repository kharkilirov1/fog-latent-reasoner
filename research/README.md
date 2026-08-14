# FOG Research Workflow

This directory is the research control plane for `fog-latent-reasoner`.
Production code and released claims must not become a substitute for an experiment log.

## State machine for research work

Every substantial direction moves through the same states:

`IDEA -> PROTOCOL -> RUNNING -> RESULT -> DECISION -> PRODUCTION (optional)`

1. **IDEA** — mechanism, motivation, predicted failure modes. No claim that it works.
2. **PROTOCOL** — frozen task, baselines, metrics, interventions and pass/fail criteria before the locked test is opened.
3. **RUNNING** — only implementation/debugging and validation. Protocol changes must be recorded.
4. **RESULT** — machine-readable metrics plus a short interpretation. Negative results stay in the repository.
5. **DECISION** — promote, revise, pause, or reject, with the reason.
6. **PRODUCTION** — only a mechanism that survived its decisive gate is allowed into the main architecture contract.

## Naming

- Ideas: `IDEA-NNN-short-name.md`
- Experiments: `EXP-NNN-short-name.md`
- Results: `RESULT-NNN-short-name.md`
- Artifacts: `artifacts/research/exp_NNN_*`
- Code may use the same numeric ID in its module docstring.

IDs are never reused even if a direction is rejected.

## Evidence levels

- **E0 hypothesis**: proposed mechanism only.
- **E1 unit**: deterministic/unit-level mechanism works.
- **E2 controlled**: trained controlled task + causal intervention.
- **E3 composition**: multiple latent transitions compose on unseen operators/depths.
- **E4 generated-value**: an intermediate value absent from the prompt is created and reused.
- **E5 semantic**: works on natural-language/code/scientific inputs with matched baselines.
- **E6 scaled**: survives scaling with efficiency/quality evidence.

Do not use a higher-level claim to describe lower-level evidence.

## Locked-test rule

Validation is used for architecture and hyperparameter decisions. A locked `test` split is opened only after the experiment's protocol pass criteria are frozen. If the test is opened, subsequent design changes create a new experiment ID.

## Failure taxonomy

Every failed run is labeled as one of:

- `ENVIRONMENT` — missing dependency, hardware/runtime issue;
- `IMPLEMENTATION` — bug/contract violation;
- `OPTIMIZATION` — trainable mechanism did not fit reliably;
- `GENERALIZATION` — fit train/ID but failed held-out operators/depths;
- `MECHANISM` — intended causal path is not actually used;
- `HYPOTHESIS` — mechanism is correctly implemented but the research claim fails.

This prevents infrastructure failures from being misreported as model failures.

## Current priority

Controlled experiments now cover exact binding, recurrent composition, generated
latent values, PC/HALT, learned stable cyclic charts, operator induction and
mixed non-commuting operations in a shared invariant representation.

The sparse-action question is now **passed in the controlled linear setting**:
EXP-023/025 recover a diagonal/monomial grammar and its finite laws from learned
dense actions, EXP-024 projects noisy operators back to stable recurrent laws,
and EXP-026 compresses a redundant action library to primitive generators.

EXP-036 closes the first high-dimensional JVP/VJP instrumentation bottleneck,
and EXP-037 plus `register_machine_v3` closes the first model-side generated-value
construction gate.  The reference 10.245M model is now mechanically build-ready.

The highest-value open question is now:

> **Can the built register machine learn semantic/contextual binding and
> operator induction from natural-language/code data while preserving the
> closure, hard-routing and recurrent-stability guarantees found in controlled
> tasks?**

EXP-027/028 establish the first partial-observation sample law and EXP-029 shows
that canonical gauges can align independently trained modules when the operator
algebra is the same.
EXP-030 further shows that the compiler can search compositions of degenerate
learned actions and synthesize a missing simple-spectrum primitive anchor.
EXP-031..034 remove simple-spectrum, exact-action, direct-matrix and single-global
operator assumptions through commutants, trajectory identification and local
Jacobian gauge synchronization.  EXP-035 now starts from a nonlinear black-box
map, actively discovers its finite hidden orbit/graph and estimates Jacobians
with perturbation probes before compilation.

Two design branches remain deliberately alive:

1. **invariant shared register** — wider representation, sparse operator actions,
   no CAST (EXP-019/020);
2. **specialized atlas** — compact operator-specific charts plus explicit learned
   CAST/bridge laws (EXP-008/009).

Future experiments should compare these by representation width, action
complexity, recurrent stability and held-out semantic generalization.  The
production 10M checkpoint transfer remains a parallel gate, not a substitute for
this representation-learning question.
