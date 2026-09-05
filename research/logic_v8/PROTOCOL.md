# Logic v8: copy-equivariant joint role routing

Base: acc05871bfc5b135e8b5e39c86e8c43e8a56c05b (Logic v7).

## Integrity findings
The unmodified v7 generator creates 51 training FOLLOW(calls) instructions after HALT. v7's auxiliary loss supervises those instructions. Thus the published FOLLOW(calls) result is not a clean held-out-operation witness. The executor itself passed all 888 oracle programs in the reference audit.

The v7 verdict checks a held-out threshold of .65, whereas its written protocol requires .70; it also omits several other gates. v8 checks all eleven written criteria and treats missing/nonfinite metrics as failure.

## Changes
Original program generation, texts, phrase split, dataset seeds, train/test depths, answer balance and written thresholds are immutable and hash-checked. All training computation, including final-answer supervision, is restricted to the prefix through the first gold HALT. The test retains the original post-HALT nuisance instructions. The normal inference path receives no gold opcodes, arguments or answers.

Names are replaced with canonical lexical aliases in first-mention order, with original identities copied into separate payload slots. This is not a semantic parser: it does not know which entity is source or target. Repeated mentions share payload identity and all their token occurrences are retained. The semantic reader is therefore invariant to consistent renaming of the twelve supported entities, while the output identity changes equivariantly. A bidirectional role reader over frozen Qwen features predicts a JOINT pair of roles. Hard-ST forward and hard inference use the same joint MAP assignment.

Opcode/relation prototype heads and train/scan-selected layers [0,1,2,22,23] are retained from v7. Frozen backbone: Qwen/Qwen2.5-0.5B-Instruct. The new batched tensor executor implements the same typed-register transitions, tested against v7 oracle states. Undefined reads preserve v7 behavior but are explicitly diagnosed. Import-time pip installation, silent CPU fallback and potentially non-deranged shuffles are removed from the new path.

## Evaluation discipline
Model seed 0, 1600 isolated steps, then R1/R2/R4 curriculum, semantic annealing and five final-only epochs. The adapter seed is now honored explicitly. All hyperparameters and dataset hashes are recorded. No training labels are supplied to normal inference.

The first checkpoint is evaluated on TRAIN and DEV only. Locked evaluation is permitted only after train accuracy >=99%, dev program accuracy >=90%, and dev BIND >=90%. The checkpoint is saved and hashed before any locked text is featurized. Locked evaluation uses that checkpoint without training. All eleven original gates must pass; the old 50.20% result is not treated as a valid clean held-out baseline because of the discovered leak. Multiple architectural changes mean this is not an isolated causal ablation of role attention alone.

## Scope
Passing these tests does not establish a general language model or autonomous reasoning. This remains a closed-domain, twelve-entity/six-relation instruction-following machine with supplied instruction sequence and fixed operator algebra. No full model result is claimed until the training artifacts and locked metrics exist.

## Commands
```bash
python -m pip install torch numpy pytest 'transformers>=4.45,<5' accelerate
python -m pytest research/logic_v8 -v
python research/logic_v8/run.py --audit-only
python research/logic_v8/run.py --device auto --checkpoint fog_logic_v8.pt --output dev-results.json
# Only after dev qualification, without additional training:
python research/logic_v8/run.py --evaluate-checkpoint fog_logic_v8.pt --locked --output locked-results.json
```
