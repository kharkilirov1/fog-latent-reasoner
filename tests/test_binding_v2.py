import torch

from binding_v2_experiment import build_model
from matched_structured_lookup_experiment import (
    StructuredModelConfig,
    StructuredTaskConfig,
    make_batch,
)


def config(task: StructuredTaskConfig) -> StructuredModelConfig:
    return StructuredModelConfig(
        d_model=16,
        n_heads=4,
        n_layers=1,
        d_ff=32,
        max_seq_len=32,
        latent_slots=2,
        reasoning_steps=2,
        compare_rank=4,
        planner_ff=32,
        memory_slots=2,
        n_reasoning_modes=1,
        fixed_orthogonal_keys=True,
    )


def test_primary_is_payload_only_and_protected_across_depth():
    task = StructuredTaskConfig(table_size=4)
    model = build_model(task, config(task), model_seed=3)
    batch = make_batch(task, data_seed=5, split="train", start_index=0, batch_size=8)
    memory, aux = model.reason(batch, return_history=True)
    assert memory.shape == (8, 2, 16)
    assert len(aux["history"]) == 2
    assert torch.equal(memory[:, 0], aux["primary"])
    assert torch.equal(aux["history"][-1]["primary"], aux["primary"])


def test_query_and_primary_interventions_change_real_logits():
    task = StructuredTaskConfig(table_size=4)
    model = build_model(task, config(task), model_seed=7)
    batch = make_batch(task, data_seed=11, split="train", start_index=0, batch_size=8)
    normal = model.logits(batch)
    zero = model.logits(batch, intervention="zero_primary")
    deranged = model.logits(batch, intervention="target_deranged_primary")
    query_deranged = model.logits(batch, intervention="query_deranged")
    assert not torch.equal(normal, zero)
    assert not torch.equal(normal, deranged)
    assert not torch.equal(normal, query_deranged)


def test_binding_writer_and_readout_receive_gradient_without_bos():
    task = StructuredTaskConfig(table_size=4)
    model = build_model(task, config(task), model_seed=13)
    batch = make_batch(task, data_seed=17, split="train", start_index=0, batch_size=8)
    loss = torch.nn.functional.cross_entropy(model.logits(batch), batch.targets)
    loss.backward()
    for parameter in (
        model.bind.q_proj.weight,
        model.bind.k_proj.weight,
    ):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.norm() > 0
    assert not hasattr(model, "answer_bos")
    assert model.classifier.weight is model.value.weight
    assert model.classifier.weight.requires_grad is False


def test_row_permutation_does_not_change_binding_result():
    task = StructuredTaskConfig(table_size=4)
    model = build_model(task, config(task), model_seed=19)
    batch = make_batch(task, data_seed=23, split="train", start_index=0, batch_size=8)
    permutation = torch.tensor([2, 0, 3, 1])
    permuted = type(batch)(
        row_sources=batch.row_sources[:, permutation],
        row_values=batch.row_values[:, permutation],
        query_keys=batch.query_keys,
        targets=batch.targets,
        sample_indices=batch.sample_indices,
        mappings=batch.mappings,
        signatures=batch.signatures,
    )
    assert torch.allclose(model.logits(batch), model.logits(permuted), atol=1e-6)
