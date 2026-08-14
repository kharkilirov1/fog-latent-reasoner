import math

import torch

from matched_structured_lookup_experiment import StructuredTaskConfig, make_batch
from recurrent_binding_composition_experiment import (
    CompositionConfig,
    RecurrentCompositionBinder,
    target_for_depth,
)


def tiny_model(scale: float = 40.0) -> RecurrentCompositionBinder:
    task = StructuredTaskConfig(table_size=8)
    return RecurrentCompositionBinder(
        task,
        CompositionConfig(d_model=16, initial_scale=scale),
    ).eval()


def test_exp001_trains_only_sharpness():
    model = tiny_model()
    trainable = [name for name, p in model.named_parameters() if p.requires_grad]
    assert trainable == ["bind.logit_scale"]


def test_target_for_depth_matches_explicit_mapping_iteration():
    task = StructuredTaskConfig(table_size=8)
    batch = make_batch(
        task,
        data_seed=20260814,
        split="validation",
        start_index=0,
        batch_size=7,
    )
    target = target_for_depth(batch, 5, torch.device("cpu"))
    expected = []
    for mapping, start in zip(batch.mappings, batch.query_keys.tolist(), strict=True):
        state = start
        for _ in range(5):
            state = mapping[state]
        expected.append(state)
    assert torch.equal(target, torch.tensor(expected))


@torch.inference_mode()
def test_recurrent_feedback_composes_while_static_repeats_first_hop():
    task = StructuredTaskConfig(table_size=8)
    model = tiny_model(scale=80.0)
    batch = make_batch(
        task,
        data_seed=20260814,
        split="validation",
        start_index=13,
        batch_size=64,
    )
    depth = 7
    target = target_for_depth(batch, depth, torch.device("cpu"))
    recurrent, _ = model.reason(batch, depth=depth, mode="recurrent")
    static, _ = model.reason(batch, depth=depth, mode="static")
    recurrent_pred = model.readout_logits(recurrent).argmax(-1)
    static_pred = model.readout_logits(static).argmax(-1)
    one_hop = target_for_depth(batch, 1, torch.device("cpu"))
    assert torch.equal(recurrent_pred, target)
    assert torch.equal(static_pred, one_hop)
    assert not torch.equal(static_pred, target)


@torch.inference_mode()
def test_hard_recurrent_is_exact_diagnostic_ceiling():
    task = StructuredTaskConfig(table_size=8)
    model = tiny_model(scale=2.0)
    batch = make_batch(
        task,
        data_seed=20260814,
        split="validation",
        start_index=101,
        batch_size=32,
    )
    depth = 12
    target = target_for_depth(batch, depth, torch.device("cpu"))
    primary, _ = model.reason(batch, depth=depth, mode="hard_recurrent")
    pred = model.readout_logits(primary).argmax(-1)
    assert torch.equal(pred, target)


def test_initial_scale_is_applied_exactly():
    model = tiny_model(scale=3.25)
    assert math.isclose(float(model.bind.logit_scale.detach().exp()), 3.25, rel_tol=1e-6)
