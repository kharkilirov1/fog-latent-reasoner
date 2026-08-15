# Pretrained LLM → FOG retrofit experiment report

Date: 2026-08-16

Status: **controlled proof-of-concept positive; standard math-benchmark uplift not yet demonstrated.**

## Question

Can a pretrained language model be frozen and used as a semantic front-end for a separate FOG-style recurrent computation engine, rather than retraining the full language model?

The experiment used `HuggingFaceTB/SmolLM2-135M-Instruct` (134,515,008 frozen parameters, hidden width 576). The backbone weights were never updated.

A larger `Qwen/Qwen2.5-0.5B-Instruct` version was prepared (`experiments/fog_retrofit_qwen05b.py`) but the connected Hugging Face Jobs account returned HTTP 402 Payment Required, so no Qwen result is claimed here.

## Test task

The controlled task consists of natural-language arithmetic programs over `Z_31`, e.g.

```
Work modulo 31. Start with 17.
Step 1: Add 8.
Step 2: Multiply by 3.
...
```

Training programs contain depths 1–4. Held-out test programs contain depths 1, 2, 3, 4, 5, 6, 8, 10, and 12 and are generated from independent seeds.

The FOG execution engine uses a hard finite operator grammar for ADD/SUB/MUL. This is intentionally a controlled retrofit test: the operator primitives are structured and exact; the semantic bridge is learned from frozen LLM hidden states.

## 1. Naive final-hidden retrofit: negative

The first implementation attached matched trainable heads to only the final frozen SmolLM hidden state. The arithmetic space had 101 classes; 960 examples were used for training and 720 for testing. Trainable heads were approximately parameter-matched:

- linear: 58,277 trainable parameters;
- MLP: 280,165;
- GRU: 225,413;
- FOG: 268,331.

All heads remained close to chance on held-out depth. FOG did not systematically beat MLP or GRU. For FOG, accuracies by depth were 3.75%, 1.25%, 3.75%, 1.25%, 0%, 1.25%, 0%, 1.25%, and 0% at depths 1,2,3,4,5,6,8,10,12 respectively.

A separate constrained official GSM8K subset (official train/test split; final answers restricted to 0–100; constrained numeric classification) also did **not** show an uplift: linear = 2.5%, MLP = 0%, FOG at R=4 = 2.5%. This is **not** a full GSM8K score.

Conclusion: **`final hidden → small FOG → answer` is not sufficient.**

## 2. Bridge diagnosis

A second experiment used modulo 31 and explicitly measured what machine symbols can be decoded from the frozen final hidden state.

Linear probes on held-out examples gave:

- starting numeric value: **100.0%**;
- operation identity (`ADD_1..8`, `SUB_1..8`, `MUL_2..5`): **57.63%**.

Thus the frozen LLM retained the exact value but the final hidden representation was a poor interface for the operation/control symbol. This predicted the failure of long recurrent execution: a per-step opcode error compounds with depth.

## 3. Layer sweep: operation information exists, but earlier

We then linearly probed every hidden state of SmolLM2 (embedding + 30 transformer layers; width 576) using independent train/validation/test sets.

Two very different semantic taps emerged:

- **Layer 30:** start/value accuracy = **100% train / 100% validation / 100% test**; opcode test accuracy ≈ **51.86%**.
- **Layer 18:** opcode accuracy = **100% train / 100% validation / 100% test**; start/value test accuracy ≈ **94.25%**.
- Layer 17 also retained opcode almost perfectly (≈99.28% test).

This is the key mechanistic result: **the pretrained LLM already contains the exact operation identity in an intermediate layer, but that control information is substantially mixed away by the final layer.**

Therefore the retrofit interface should be typed and multi-tap rather than reading one universal final hidden vector.

## 4. Semantic-engine upper bound

With an explicitly supervised semantic bridge and exact hard ADD/SUB/MUL primitives, the engine itself is not the bottleneck.

Using the final layer for both roles:

- start parser: 100% test;
- opcode parser: 74.25% test;
- oracle engine: **100% at every depth through 12**;
- predicted start + oracle operations: **100% at every depth through 12**;
- oracle start + predicted operations: 100% at R=1, 99% at R=2, 89% at R=3–4, 51% at R=5, 26% at R=6, 10% at R=8, 0% at R=10, 7% at R=12.

The entire long-horizon failure is therefore explained by accumulated semantic opcode errors, not by recurrent arithmetic execution.

A paraphrase-only language-shift split reduced opcode parsing to 9.55%, showing that this particular bridge did not yet generalize linguistically.

## 5. Dual-tap retrofit: positive controlled result

The decisive run selected the two taps **using validation results from the prior layer sweep only**:

- value register input: **layer 30**;
- operation/control input: **layer 18**.

Only two small linear semantic bridges were trained. The SmolLM2 backbone remained completely frozen. The downstream engine was the same hard FOG modular operator grammar.

Bridge accuracy on a fresh test seed:

- value: **100.00%**;
- opcode: **99.36%**.

The model was trained only on programs of depth 1–4. Held-out execution accuracy was:

| Depth | Dual-tap FOG |
|---:|---:|
| 1 | **100.0%** |
| 2 | **100.0%** |
| 3 | **100.0%** |
| 4 | **100.0%** |
| 5 | **100.0%** |
| 6 | **100.0%** |
| 8 | **96.67%** |
| 10 | **88.33%** |
| 12 | **84.17%** |

With oracle operations, the same engine remained **100% through depth 12**, confirming again that residual long-depth errors come from the ~0.64% opcode recognition error rather than arithmetic recurrence.

### Causal intervention

Shuffling operation states across examples destroyed performance:

| Depth | Shuffled opcode states |
|---:|---:|
| 1 | 4.17% |
| 2 | 5.83% |
| 3 | 7.50% |
| 4 | 5.83% |
| 5 | 5.83% |
| 6 | 3.33% |
| 8 | 6.67% |
| 10 | 2.50% |
| 12 | 3.33% |

Chance is `1/31 = 3.23%`. This is strong causal evidence that the computation uses the extracted control states rather than an answer shortcut.

### Raw pretrained-model baseline

The unmodified SmolLM2-135M-Instruct was also evaluated by ordinary autoregressive generation on the same task family, with no adapters. Accuracy was low:

| Depth | Raw SmolLM2 |
|---:|---:|
| 1 | 2% |
| 2 | 8% |
| 3 | 4% |
| 4 | 12% |
| 5 | 2% |
| 6 | 2% |
| 8 | 0% |
| 10 | 4% |
| 12 | 4% |

Therefore the high dual-tap result is not explained by the pretrained model already solving these arithmetic programs by text generation.

## 6. Language generalization remains open

The decisive dual-tap bridge was trained on one explicit operation phrasing. On a held-out paraphrase style, opcode decoding was **66.93%**, and full execution degraded rapidly with depth (50.83% at R=1, 33.33% at R=2, 23.33% at R=3, 19.17% at R=4, and 6.67% at R=12).

This is a useful failure: the arithmetic engine is stable once symbols are correct, but the current semantic compiler is not yet robust to broad natural-language variation.

## 7. What is and is not established

### Established in this controlled experiment

1. A pretrained 135M LLM can remain **fully frozen** while small learned taps expose machine-usable semantic state.
2. Different semantic roles can have different optimal transformer depths: final hidden is not necessarily the correct universal interface.
3. A typed multi-layer bridge plus a hard FOG operator grammar can convert the frozen LLM into a reliable recurrent arithmetic executor.
4. Training only on depths 1–4 gives strong extrapolation through depth 12 when the semantic symbols are recognized correctly.
5. Shuffling control states collapses performance to near chance, providing causal evidence for the FOG path.

### Not established

1. This is **not** yet an uplift on full GSM8K, MATH, AIME, or another standard mathematical benchmark.
2. The current operators are structured exact modular primitives; the experiment does not show that ADD/SUB/MUL were discovered from final-answer supervision alone.
3. The semantic bridge uses explicit symbol supervision and limited operation language.
4. Paraphrase generalization is currently insufficient.
5. The Qwen2.5-0.5B experiment has not yet been executed because the available Hugging Face Jobs account had no billed compute.

## 8. Decision

The retrofit hypothesis should **not** be phrased as “attach FOG to the final hidden state.” The evidence supports a more specific architecture:

```
pretrained frozen transformer
       │
       ├── late semantic tap ─────→ value register
       │
       └── mid-layer control tap ─→ opcode/control register
                                     │
                                     ▼
                            hard FOG operator grammar
                                     │
                                     ▼
                            recurrent machine state
                                     │
                                     ▼
                               answer/readout
```

The next serious benchmark step is therefore to train a richer semantic program compiler (preferably over multiple paraphrases and natural math rationales/program traces), while keeping the stable recurrent engine fixed, and then evaluate official held-out GSM8K/MATH. The experiment suggests that **semantic compilation, not recurrent arithmetic, is now the dominant bottleneck.**
