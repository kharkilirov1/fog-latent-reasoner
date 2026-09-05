# Logic v8.2: typed identity-blind local binding

This protocol is committed BEFORE locked evaluation of the selected model.

## Data and selection
The base is the original clean v8 checkpoint, NOT the v8.1 expanded-template checkpoint. Original v7 training wording, program seeds, depths, answers and held-out evaluation remain unchanged. All training computations are cropped through the first gold HALT, removing post-HALT holdout contamination. No extra training phrase or locked label is introduced.

The initial joint-reader model and the declared-template v8.1 repair each gave train 288/288 but dev 66/96. Separate label-free Qwen role probes did not reliably fix roles. These failures are retained, not represented as successes.

A local role reader was explored on TRAIN/DEV only. A shared radius-3 multilayer role head harmed SELECT; it was rejected. Four BIND-only radius-1 variants were tried: linear versus multilayer readout, with versus without pre-first-mention context. Both variants that kept the prefix reached train/dev 100%; the simpler linear variant was selected. This is development-set model selection, not a four-seed replication or independent validation.

## Selected model
Frozen Qwen2.5-0.5B-Instruct revision 7ae557604adf67be50417f59c2c2f167def9a775 supplies features. Existing clean-v8 opcode, relation and non-BIND role heads are frozen. The new BIND head sees only layer-zero lexical features immediately LEFT and RIGHT of each mention, with ALL entity-token features zeroed. It has a learned 32-wide projection and linear source/target readout. There are 30,625 new parameters and 858,196 total adapter parameters.

Role pairs are chosen jointly. Actual identities are copied separately. The model selects the BIND head using its PREDICTED opcode, never an oracle opcode. The code contains no Save/possessive/verb parsing rule, no template matcher and no answer lookup.

The new head uses seed 0, 500 full-batch AdamW steps on 30 canonical BIND examples already present in the original training set. Then it undergoes five epochs with only the final-answer loss (SGD .001, no auxiliary targets, no weight decay). At an already-correct hard solution these epochs have zero loss, so this demonstrates retention, NOT learning from scratch using final rewards alone.

## Freeze and evaluation
The model achieved 288/288 TRAIN and 96/96 DEV, including every dev BIND, before and after final-only epochs. Local independently rerun tensor state digest:

f4d8055dc740fdcc31bd38290be9d3610e6f84d7e0d86bc34a6e6e222a2e18fa

The checkpoint is saved and SHA256-hashed before any locked text is encoded. Train >=99%, dev >=90%, dev BIND >=90% qualify the checkpoint for locked evaluation. The locked set is evaluated once with no training afterward. All eleven original written protocol gates are checked. Post-evaluation assertions verify both saved-checkpoint bytes and in-memory tensors are unchanged.

Passing these finite tests would demonstrate a domain-specific learned instruction interpreter, not general autonomous planning or unrestricted language understanding. The supplied instruction sequence and six-operator algebra remain part of the task definition.
