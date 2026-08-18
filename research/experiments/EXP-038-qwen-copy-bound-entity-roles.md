# EXP-038 — Copy-Bound entity roles in frozen Qwen

## Question

Can protected lexical copying of entity mentions, followed by learned role addressing, improve ingestion of initial BIND facts into the discrete FOG relation memory?

## Locked protocol

The experiment used frozen `Qwen/Qwen2.5-0.5B-Instruct`, train depths R=1,2,3,4, locked evaluation depths R=1,2,3,4,5,6,8, the fixed paraphrase split, fixed seeds, fixed thresholds, and the locked intervention controls. The executor was not changed. The complete protocol snapshot and raw artifacts are in `artifacts/logic-v7-copy-bound-entity-roles/`.

## Result

Copy-Bound improved locked BIND joint accuracy from approximately 69.6% in Logic-v6 to 72.06% in Logic-v7, but did not reach the locked 90% BIND gate. Locked program accuracy was 50.20%, while full oracle execution was 100%. The intervention therefore localizes the remaining limitation to the semantic bridge and relation-memory ingestion rather than the executor.

## Verdict

**SEMANTIC_INTERFACE_LIMITED**.

See [`RESULTS.md`](../../artifacts/logic-v7-copy-bound-entity-roles/RESULTS.md) and [`RESULT-037`](../results/RESULT-037-qwen-copy-bound-entity-roles.md).

<!-- experiment-id: EXP-038 -->
EOF

# Create the companion result entry in a separate file via shell only for portability.
cat > /home/ubuntu/fog-latent-reasoner/research/results/RESULT-037-qwen-copy-bound-entity-roles.md <<'EOF'
# RESULT-037 — Copy-Bound entity roles remain interface-limited

Logic-v7 completed the frozen-Qwen Copy-Bound experiment. Locked end-to-end program accuracy was **50.20%**, held-out accuracy was **45.61%**, and locked BIND joint accuracy was **72.06%**. The full oracle program reached **100%**, while `oracle_bind` reached **92.46%**.

The Copy-Bound mechanism produced a directional improvement over Logic-v6's approximately 69.6% BIND joint score, but not the required 90% gate. The executor is not the failure source; the remaining bottleneck is the semantic bridge that maps natural-language entity and relation mentions into reliable relation-memory writes.

> **Verdict: `SEMANTIC_INTERFACE_LIMITED`**

The complete report, JSON, log, protocol snapshot, final script, and checkpoint hash are available under [`artifacts/logic-v7-copy-bound-entity-roles/`](../../artifacts/logic-v7-copy-bound-entity-roles/). The adapter checkpoint itself is intentionally distributed as a GitHub Release asset rather than committed to repository history.

<!-- result-id: RESULT-037 -->
EOF
