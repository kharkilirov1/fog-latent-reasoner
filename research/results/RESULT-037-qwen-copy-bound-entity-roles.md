# RESULT-037 — Copy-Bound entity roles remain interface-limited

Logic-v7 completed the frozen-Qwen Copy-Bound experiment. Locked end-to-end program accuracy was **50.20%**, held-out accuracy was **45.61%**, and locked BIND joint accuracy was **72.06%**. The full oracle program reached **100%**, while `oracle_bind` reached **92.46%**.

The Copy-Bound mechanism produced a directional improvement over Logic-v6's approximately 69.6% BIND joint score, but not the required 90% gate. The executor is not the failure source; the remaining bottleneck is the semantic bridge that maps natural-language entity and relation mentions into reliable relation-memory writes.

> **Verdict: `SEMANTIC_INTERFACE_LIMITED`**

The complete report, JSON, log, protocol snapshot, final script, and checkpoint hash are available under [`artifacts/logic-v7-copy-bound-entity-roles/`](../../artifacts/logic-v7-copy-bound-entity-roles/). The adapter checkpoint itself is intentionally distributed as a GitHub Release asset rather than committed to repository history.

<!-- result-id: RESULT-037 -->
