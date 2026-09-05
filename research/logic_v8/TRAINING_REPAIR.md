# Version 8.1: repair declared training-template reachability

The first v8 experiment retained v7 training text exactly. Its fresh checkpoint achieved train 288/288 but dev 66/96 (68.75%); dev BIND was 74.54%. It DID NOT qualify for locked evaluation. The locked test was not evaluated.

All six failing canonical dev BIND phrases were the `Save A as B's REL destination` form. Source and target were reversed. Inspection located a generator defect: for BIND/FOLLOW, odd variants are redirected to a relation-specific pool. Consequently four of eight generic TRAIN templates are unreachable. The two-element relation-specific BIND pool also always chooses its second entry. This excludes the generic reversed-role TRAIN form `Register B as the REL target of A` and several natural inverse relation forms.

V8.1 explicitly changes TRAIN text exposure: for semantic instruction labels already present in the old isolated training set or active prefixes of the same 288 training programs, enumerate ALL existing TRAIN_TEMPLATES and REL_BIND_TRAIN/REL_FOLLOW_TRAIN entries. It does not read SCAN_TEMPLATES or TEST_TEMPLATES to construct training examples. No new test formulation is inserted, no active holdout is admitted, and no program, answer, depth, dataset seed, or test-rendering choice is changed. There are 802 supervised instruction types and 7,715 phrase examples. The expanded full phrase-manifest hash is `167a0cf0ee1d851d62576a36e7b3db8563591f93c70b1666a1d88cf1ae63d698`; the old manifest is retained in metadata.

This is NOT an isolated architectural ablation and is NOT described as the identical training protocol to v7/v8. It is a versioned data-generator repair. The architecture, seed 0, number of isolated optimization steps, recurrent curriculum and five final-only epochs remain unchanged. The earlier train/dev-only semantic LM probes are a diagnostic experiment, not a component of this v8.1 model.

The development eligibility rule remains train >=99%, dev program >=90%, dev BIND >=90%. Locked evaluation uses the frozen checkpoint without training and applies all eleven original written gates. The standalone predictor consumes only user instruction text, never gold opcodes or expected answers.

Run:
```bash
python research/logic_v8/run.py --training-texts declared --device auto --checkpoint fog_logic_v8.pt --output dev-results.json
python research/logic_v8/run.py --training-texts declared --evaluate-checkpoint fog_logic_v8.pt --locked --output locked-results.json
python research/logic_v8/predict.py --checkpoint fog_logic_v8.pt --input research/logic_v8/example_program.txt --trace
```
