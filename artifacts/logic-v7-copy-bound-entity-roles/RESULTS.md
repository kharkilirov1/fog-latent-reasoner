# Logic-v7 — Copy-Bound Entity Roles

## Executive summary

Logic-v7 tested whether entity mentions copied as protected lexical payloads and routed into learned roles improve relation-memory ingestion compared with learned entity re-classification. The run used the locked FOG v7 protocol, a frozen `Qwen/Qwen2.5-0.5B-Instruct` backbone, and the unchanged discrete typed register-machine executor.

The hypothesis was **partially supported**. Locked BIND joint accuracy increased from approximately 69.6% in Logic-v6 to **72.06%** in Logic-v7, but it remained below the locked 90% gate. Locked end-to-end program accuracy was **50.20%**, below the required 75%. The protocol verdict is:

> **SEMANTIC_INTERFACE_LIMITED**

The executor itself is validated: full oracle program execution reached **100%**. The strong degradation under shuffling and the large recovery under `oracle_bind` localize the main failure to the learned semantic-to-symbolic interface, especially initial BIND facts entering relation memory.

## Protocol and integrity

| Item | Value |
|---|---|
| Backbone | `Qwen/Qwen2.5-0.5B-Instruct` |
| Backbone training | Frozen |
| Train depths | R = 1, 2, 3, 4 |
| Locked test depths | R = 1, 2, 3, 4, 5, 6, 8 |
| Phrase manifest | 2,420 texts; SHA-256 `4e4697299e47b99f7408815509870a184cc8899a3327a01231545541bb6270aa` |
| Holdouts | `FOLLOW(calls)`, `COMPARE(Kai)`, `SELECT(Iris,Lena)` |
| Programs | train 288; dev 96; test 504 |
| Entity mechanism | Protected lexical copy plus learned role addressing |
| Trainable adapter parameters | 1,318,936 |

No test depths, seeds, paraphrase split, thresholds, or baselines were changed after observing test results.

## Locked criteria

| Criterion | Required | Result | Status |
|---|---:|---:|---|
| Balanced majority baseline | 8.33% | 8.33% | PASS |
| Train program accuracy | >= 99% | 100.00% | PASS |
| Locked BIND joint | >= 90% | 72.06% | FAIL |
| Locked instruction joint | >= 90% | 89.10% | FAIL by 0.90 pp |
| Locked program accuracy | >= 75% | 50.20% | FAIL |
| Held-out program accuracy | >= 70% | 45.61% | FAIL |
| `COMPARE(Kai)` phrase joint | >= 90% | 100.00% | PASS |
| `FOLLOW(calls)` phrase joint | >= 95% | 100.00% | PASS |
| `SELECT(Iris,Lena)` phrase joint | >= 80% | 100.00% | PASS |
| Full shuffle decrease | >= 30 pp | 45.63 pp | PASS |
| Oracle program execution | 100% | 100.00% | PASS |

## Main metrics

| Split | Program accuracy | Held-out accuracy | Instruction joint | BIND joint |
|---|---:|---:|---:|---:|
| Train | 100.00% | 0.00% | 100.00% | 100.00% |
| Dev | 69.79% | 0.00% | 92.40% | 76.12% |
| Locked test | 50.20% | 45.61% | 89.10% | 72.06% |

Locked program accuracy by depth: R=1 **62.50%**, R=2 **63.89%**, R=3 **41.67%**, R=4 **62.50%**, R=5 **26.39%**, R=6 **43.06%**, R=8 **51.39%**.

## Interface diagnostics

| Field | Locked accuracy |
|---|---:|
| Opcode | 98.21% |
| Entity role e1 | 84.31% |
| Relation | 97.40% |
| Entity role e2 | 83.20% |
| Full argument/opcode joint | 82.79% |

By opcode, BIND joint was 72.02%, FOLLOW and COMPARE were 100%, and SELECT was 92.68%. All three lexical holdouts were solved at phrase level.

## Causal controls

| Arm | Locked | Held-out |
|---|---:|---:|
| Normal semantic interface | 50.20% | 45.61% |
| Full semantic shuffle | 4.56% | 5.51% |
| Shuffle within opcode | 7.74% | 7.77% |
| Oracle opcode only | 55.56% | 51.38% |
| Oracle all arguments | 92.46% | 91.98% |
| Oracle BIND only | 92.46% | 91.98% |
| Full oracle program | 100.00% | 100.00% |

`oracle_bind - normal` equals **42.26 percentage points**, while full-oracle execution is perfect. This rules out executor failure and points to BIND ingestion and semantic role addressing as the bottleneck.

## Runtime and hardware

| Item | Value |
|---|---|
| Kaggle kernel | `lirovkharki/fog-logic-v7-copy-bound-entity-roles` |
| URL | https://www.kaggle.com/code/lirovkharki/fog-logic-v7-copy-bound-entity-roles |
| Requested accelerator | GPU T4 x2 |
| Internet | Enabled |
| Reported execution device | CPU |
| PyTorch | `2.10.0+cu128` |
| Runtime | 3,027.487 seconds (~50.46 minutes) |

The kernel was configured for T4 x2, but the final result JSON reports `device: cpu` because the submitted script used a CPU-forcing fallback. The measured runtime is therefore a CPU run in a GPU-enabled Kaggle kernel, not a T4 benchmark.

## Artifacts and verdict

The directory contains the report, raw result JSON, full stdout/stderr log, locked protocol snapshot, and final script. The 6.25 MB adapter checkpoint is excluded from Git history and is published as a GitHub Release asset.

> **Verdict: `SEMANTIC_INTERFACE_LIMITED`**

Copy-Bound identity is directionally useful, but the frozen Qwen semantic bridge still does not reliably populate relation memory on the locked multi-step benchmark. The next experiment should preserve the protocol and target role addressing and relation-memory ingestion.

<!-- experiment-id: EXP-038 / result-id: RESULT-037 -->
