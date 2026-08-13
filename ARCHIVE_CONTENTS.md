# Archive contents

This release bundle contains:

- all source modules, training/evaluation scripts, tests, and reports;
- the primary `query_bound_v2` BF16 checkpoint;
- the secondary lexical-pretrained legacy BF16 checkpoint;
- the two legacy negative-result BF16 checkpoints referenced by the model card;
- the matching 8,192-token BPE;
- dataset manifests, experiment JSON/Markdown evidence, and synthetic-data
  generators.

It intentionally excludes:

- upstream TinyStories and GSM8K JSONL rows (recreate them with
  `download_viewer_subset.py` and verify their hashes against the manifests);
- FP32 and optimizer checkpoints;
- per-seed diagnostic `.pt` files under `artifacts/`;
- caches and Python bytecode.

The expected clean verification result is `62 passed`. Run:

```bash
pip install -e '.[dev]'
pytest -q
python verify_release.py --forward-backward
```
