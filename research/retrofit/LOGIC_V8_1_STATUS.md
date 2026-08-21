# FOG Logic v8.1 — true joint pair correction

Status: **ready for Kaggle locked run; old v8 draft retired**.

Protocol: `FOG_LOGIC_V8_1_TRUE_JOINT_PAIR_RELATION_CONDITIONED`.

The pre-run audit of the v8 draft found that ordered pair scores were still marginalized back to independent `e1/e2` hard decisions. v8.1 instead uses one hard-ST categorical variable over the ordered `(source,target)` pair, explicitly conditions BIND pair scoring on the predicted relation, and writes relation memory directly from the joint pair tensor rather than reconstructing an outer product of marginals.

The direct joint write is required for final-answer-only learning: a deliberately reversed BIND pair was verified to fail execution while receiving a finite non-zero gradient (`~2.946e-4` L1 gradient in the local structural smoke).

Additional pre-run fixes:

- `--seed` now changes adapter initialization (`20260819 + seed`) while dataset splits remain fixed;
- forced `COMPARE(Kai)=true` endpoint repair can no longer create self-edge BIND facts such as `Kai -> Kai`;
- 1–2 distractor BIND facts are restored after endpoint repair;
- exact 12-way answer balance is enforced;
- default locked program stream has 42 answers/entity and BIND surface order 1419 source-first / 1419 target-first;
- semantic holdouts remain `FOLLOW(calls)`, `COMPARE(Kai)`, `SELECT(Iris,Lena)`;
- train depths remain 1–4 and locked depths 1,2,3,4,5,6,8.

Local smoke: PASS. Oracle structural execution is exact; full semantic shuffle gives ~5.56%, within-opcode shuffle ~9.72%, and the corrected hard pair path has non-zero gradients.

Local package SHA-256: `7137b1cd57d5ebe164fc8e36ae2e5a10afd6fefeba4ff7313572c0365f5ae6f7`.

The next research evidence should come from the predeclared seeds 0/1/2 on the finalized locked stream, without architectural tuning between seed runs.