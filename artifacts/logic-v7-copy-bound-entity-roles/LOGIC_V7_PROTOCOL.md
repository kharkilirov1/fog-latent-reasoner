# FOG Logic v7 — Copy-Bound Entity Roles

## Hypothesis
Logic-v6 established that the discrete FOG executor is causal and functional, but the remaining error is concentrated in the semantic arguments that populate relation memory. The strongest signal is BIND: on locked phrases BIND joint accuracy was 69.64%, while FOLLOW and COMPARE were 100%; replacing all arguments with oracle values raised end-to-end accuracy from 46.83% to 88.29%, and full oracle execution was 100%.

v7 tests whether entity identity should be **copied as a protected lexical payload** rather than re-classified from contextual Qwen states. This mirrors the successful Binding-v2 address/payload separation used earlier in FOG.

## Architectural change
v6:

`Qwen hidden states -> learned 12-way entity identity classifier -> e1/e2`

v7:

`Qwen/tokenizer -> detect explicit entity mentions -> COPY identity -> learned role addressing -> e1/e2`

The learned model may choose which mention fills role e1 or e2, but it cannot invent an entity that is absent from the sentence. Entity identity therefore behaves like a symbol-table payload, not a semantic class.

Opcode and relation remain semantic:

- opcode: sentence-level prototype/QK head
- relation: token-level prototype/QK head
- e1/e2: mention-role selectors over frozen Qwen states
- executor: same hard-ST typed register machine as v6

No Qwen parameter is trained.

## Scientific isolation
The program generator, answer balancing, holdouts, train depths and locked test depths remain the same as v6. This run is intended to isolate the entity-binding mechanism rather than win by changing the benchmark.

Train depths: `R = 1,2,3,4`.
Locked test depths: `R = 1,2,3,4,5,6,8`.

Holdouts remain:

- `FOLLOW(calls)`
- `COMPARE(Kai)`
- `SELECT(Iris,Lena)`

## Why this is not an oracle
The evaluator does not supply e1/e2 labels. It only exposes the lexical mentions that are literally present in the input text, analogous to copying a variable/name token into a symbol table. A learned role selector must still determine source vs target, true vs false branch, etc. Relation meaning and opcode meaning are still learned from frozen-Qwen semantics.

## New diagnostics
In addition to the v6 controls, v7 reports:

- `bind_instruction_joint`
- program accuracy by `BIND` count
- `oracle_bind`: only BIND instructions receive oracle opcode/e1/relation/e2
- `oracle_all_args`
- full oracle
- full semantic derangement
- within-opcode semantic derangement

The key mechanistic prediction is:

`BIND joint(v7) >> 69.6%`

and the gap

`oracle_bind - normal`

should become small if copy-bound identity fixes relation-memory ingestion.

## Locked success criteria
Primary target:

- balanced majority baseline = 8.33%
- train program accuracy >= 99%
- locked BIND joint >= 90%
- locked overall instruction joint >= 90%
- locked program accuracy >= 75%
- held-out program accuracy >= 70%
- `COMPARE(Kai)` joint >= 90%
- `FOLLOW(calls)` joint >= 95%
- `SELECT(Iris,Lena)` joint >= 80%
- full shuffle at least 30 percentage points below normal
- oracle program execution = 100%

A result below 75% can still be scientifically positive if BIND improves and the new operation-specific oracle controls localize the remaining error.
