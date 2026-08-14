import torch

from matched_structured_lookup_experiment import StructuredTaskConfig, make_batch
from production_recurrent_token_composition_experiment import (
    STATE_BASE,
    token_batch_shared_identity,
)


def test_shared_identity_serialization_exposes_only_source_pairs_for_offset_n():
    task = StructuredTaskConfig(table_size=8)
    batch = make_batch(
        task,
        data_seed=20260814,
        split="validation",
        start_index=0,
        batch_size=8,
    )
    prompt, mask = token_batch_shared_identity(batch)
    n = task.table_size
    assert prompt.shape == (8, 3 * n + 1)
    assert mask[:, : 2 * n].all()
    assert not mask[:, 2 * n : 3 * n].any()
    assert mask[:, -1].all()
    pair_mask = mask[:, :-n] & mask[:, n:]
    assert torch.equal(pair_mask.sum(-1), torch.full((8,), n))
    assert pair_mask[:, :n].all()
    assert not pair_mask[:, n:].any()
    assert torch.equal(prompt[:, :n], STATE_BASE + batch.row_sources)
    assert torch.equal(prompt[:, n : 2 * n], STATE_BASE + batch.row_values)
    assert torch.equal(prompt[:, -1], STATE_BASE + batch.query_keys)
