# NOTE-016 — Attractor capture replaces the discrete executor

Status: proved (Theorems A, A′, Props B–E). Experimental check is
`experiments/fog_latent_loop_v8.py` (EXP-039).

## Motivation

Logic-v6/v7 and the SmolLM/Qwen retrofits all end the same way: oracle
execution is 100%, end-to-end accuracy is limited by the **semantic → symbolic
interface** (opcode/BIND parsing). That interface exists only because the
computation is handed to a discrete processor. This note asks the inverse
question: what does the LLM's *own* continuous loop need for the FOG stability
guarantees, without symbols?

The answer: the role of discreteness is played by a **capture attractor**
inside the continuous loop. Discreteness becomes a property of the dynamics,
not a separate processor.

## Setting

Codes `m_1..m_N ∈ R^d` (the canonical chart). Each instruction `c` (here
selected softly by the LLM's own attention to the prompt) induces a continuous
transition `F_c : R^d → R^d` with intended symbolic semantics
`P_c : [N] → [N]`. The latent loop is

    h_{t+1} = Π(F_{c_t}(h_t)),

where `Π` is a continuous clean-up map. `Π = id` is the plain continuous loop
(Coconut-like); `Π = argmax-snap` is the discrete processor.

Assumptions, for all `i, c`:

1. closure defect: `‖F_c(m_i) − m_{P_c(i)}‖ ≤ ε`;
2. local Lipschitz: `‖F_c(h) − F_c(m_i)‖ ≤ L ‖h − m_i‖` on `B(m_i, r)`;
3. clean-up: `‖Π(y) − Π(m_j)‖ ≤ κ ‖y − m_j‖` on `B(m_j, Lr+ε)`, `‖Π(m_j) − m_j‖ ≤ η`.

## Theorem A (tube invariance)

If `κ(Lr + ε) + η ≤ r`, then `h_0 ∈ B(m_{i_0}, r)` implies
`h_t ∈ B(m_{i_t}, r)` for **all** `t` and every instruction sequence, where
`i_{t+1} = P_{c_t}(i_t)` is the true symbolic trajectory. If additionally
`r < ρ = ½ min_{i≠j} ‖m_i − m_j‖`, nearest-code decoding is exact at every depth.

*Proof.* `‖F(h) − m_{P(i)}‖ ≤ ‖F(h) − F(m_i)‖ + ε ≤ Lr + ε`; then
`‖Π(F h) − m_{P(i)}‖ ≤ κ(Lr+ε) + η ≤ r`. Induct on `t`. ∎

A radius exists iff `κL < 1`; the minimal one is `r* = (κε + η)/(1 − κL)`.

**Corollary (comparison with FOG eq. 15).** `Π = id` gives `κ = 1, η = 0`:
stability needs `L < 1`, `r* = ε/(1−L)`, which is preprint eq. (15). Capture
replaces `L < 1` with `L < 1/κ`. The "flexible one-step fit" of Figure 1
(p95 gain > 1, collapses by depth 32) is stable under any clean-up with
`κ < 1/L`.

## Proposition B (threshold is sharp)

One-dimensional model: `F(h) = m' + L(h − m) + ε`, `Π(y) = m' + κ(y − m')`.
Then `e_{t+1} = κ(L e_t + ε)`, which is bounded iff `κL < 1` (for `ε > 0`,
linear growth at `κL = 1`, geometric above). So `κL < 1` cannot be weakened
in general.

## Theorem A′ (Hopfield capture: any Lipschitz constant is tolerated)

Let `Π_β(y) = Σ_j softmax_j(β⟨y, m_j⟩) m_j` with equinorm codes `‖m_j‖ = R`
and gap `Δ = R² − max_{i≠j} ⟨m_i, m_j⟩ > 0`.

*Capture lemma.* For `y ∈ B(m_i, s)` with `s < Δ/(2R)`:

    ‖Π_β(y) − m_i‖ ≤ 2R (N−1) exp(−β(Δ − 2Rs)).

(Logit gap `⟨y, m_i − m_j⟩ ≥ Δ − 2Rs`, so `1 − p_i ≤ (N−1)e^{−β(Δ−2Rs)}`, and
`Π(y) − m_i = Σ_{j≠i} p_j(m_j − m_i)`.)

**Theorem.** If `ε < Δ/(2R)`, then for **every** `L` there is a finite

    β* = min_{s ∈ (ε, Δ/2R)}  ln( 2R·L̄(N−1) / (s − ε) ) / (Δ − 2Rs),    L̄ = max(L, 1),

such that for all `β ≥ β*` the loop `h ← Π_β(F_c(h))` decodes exactly at every
depth from any start in `B(m_{i_0}, (s−ε)/L̄)`.

*Proof.* Set `r = (s−ε)/L̄ ≤ s`. Then `F` maps `B(m_i, r)` into `B(m_{P(i)}, s)`,
and the capture lemma maps that into `B(m_{P(i)}, r)` once `β ≥ β*`. Since
`r ≤ s < Δ/2R ≤ ρ`, decoding is exact. ∎

Consequences:

* The only per-step requirement on the learned transition is a **closure
  defect below `Δ/(2R)`** (for orthogonal codes: `R/2`). One-step accuracy plus
  a margin is then sufficient for unbounded depth — exactly the property that
  fails without capture (EXP-014).
* `β → ∞` recovers the discrete processor; **a finite temperature suffices**,
  so the loop stays differentiable (`‖∂Π/∂y‖ = β‖Cov_p(m)‖ ≤ 4βR²(1−p_i)`).
  There is a stability/learnability tension: train near `β*`, not at `∞`.

## Proposition C (soft routing is absorbed)

If the LLM's attention mixes candidate transitions,
`F = Σ_k π_k F_k` with `π_{k*} ≥ 1 − δ`, then the effective closure defect is
`ε_eff ≤ ε + δ · max_k ‖F_k(m_i) − F_{k*}(m_i)‖ ≤ ε + 2R'δ`. Under Theorem A′
soft routing is harmless as long as `ε + 2R'δ < Δ/(2R)`. The straight-through
hard router of FOG §9 was needed because EXP-037 had no capture in the state
loop; with capture, routing may remain the LLM's ordinary soft attention.

## Proposition D (shared readout fixes the gauge)

If intermediate states are supervised through a **single** readout `D` shared
across all steps, with `D` injective on codes, then any gauge `G` invisible to
the loss satisfies `D∘G = D` on codes, hence `G` fixes every code: the
periodic gauges of EXP-003 cannot occur. Per-step heads `D_t` would admit
`D_t = D∘G^{−t}`. In the `attr` arm the readout logits are also the capture
similarities, so the decoding chart and the feedback chart coincide by
construction.

## Proposition E (width–sharpness trade-off; evading the no-free-global-chart bound)

Theorem 1 of the preprint (`d ≥ p−1`) is a statement about **exact linear**
equivariant registers. Under capture the transition need only be `ε`-close on
`N` points and may be nonlinear (the LLM's MLPs), so the linear bound does not
apply. For `N` equinorm codes on `S^{d−1}`, the best gap scales as
`Δ ≈ R²(1 − cos θ_N)` with `θ_N ≍ N^{−1/(d−1)}`, giving

    β* ≍ (ln L + ln N) · N^{2/(d−1)} / R².

So `d = 2` is already sufficient for exact all-depth recurrence of the affine
grammar over `F_31` (codes on a circle), at inverse temperature `β ∝ N²`; at
`d ≥ p−1` codes can be orthogonal and `β* ∝ ln(LN)`. Width can be traded for
capture sharpness. The no-free-global-chart cost is real for linear,
temperature-free registers, and is paid in `β` rather than `d` once capture is allowed.

## Design consequence (EXP-039)

Remove the discrete executor. Let the LLM run latent thought tokens through
thought-mode LoRA. Feed back `x_{t+1} = Σ_v softmax(β D(h_t))_v c_v` (a soft
token over a learned code book), supervise `D(h_t)` with the intermediate value
using one shared `D`, and use coprime/variable depths. Compare against the
continuous feedback `x_{t+1} = W h_t` (`Π = id`) and the hard snap
(`β = ∞`, discrete reference).
