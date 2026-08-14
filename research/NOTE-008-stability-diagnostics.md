# NOTE-008: Stability Diagnostics — Closure Defect and Perturbation Gain

## Abstract
This note formalizes the diagnostic framework developed to evaluate recurrent operators in the FOG architecture prior to long-horizon testing. High cosine similarity or one-step reconstruction accuracy does not guarantee multi-step stability. We introduce the perturbation gain metric $\lambda$ and closure defect $\varepsilon$ to predict recurrent collapse.

## Mathematical Formulation
For a recurrent latent operator $f$, local perturbation dynamics are bounded by:
$$e_{t+1} \lesssim \lambda e_t + \varepsilon$$

Where:
- $\varepsilon$ represents the **closure defect** (deviation from operator grammar invariants).
- $\lambda$ represents the **local perturbation gain** (error amplification factor per step).

### Empirical Observations
- **Unstable Operators (Flexible / Dense):** 
  - $\lambda_{\text{mean}} \approx 1.000$, $\lambda_{p95} \approx 1.139$. 
  - Result: 100% accuracy at $R=8$, but catastrophic collapse to 0% by $R=32$.
- **Stable Structured Operators (Invariant Register):** 
  - $\lambda_{\text{mean}} \approx 0.733$, $\lambda_{p95} \approx 0.961$. 
  - Result: Sustained 100% accuracy up to $R=256$.

## Conclusion
Monitoring $\lambda < 1$ and bounding $\varepsilon$ is a mandatory gating condition for any recurrent latent reasoning operator before scaling depth.
