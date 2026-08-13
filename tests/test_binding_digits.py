import torch

from binding_digits_experiment import DigitBatch, DigitTaskConfig, build_model, make_digit_batch


def test_digit_payload_uses_disjoint_frozen_exact_codebook():
    task = DigitTaskConfig(table_size=4, digits=3, radix=10)
    model = build_model(task, d_model=16, rank=4, seed=3)
    assert not model.key.weight.requires_grad
    assert not model.digit.weight.requires_grad
    assert torch.equal(model.key.weight[:, :4], torch.eye(4))
    assert torch.equal(model.digit.weight[:, 4:14], torch.eye(10))
    oracle = model.digit(torch.arange(10)).unsqueeze(0)
    logits = torch.einsum("bld,vd->blv", model.payload_norm(oracle), model.digit.weight)
    assert torch.equal(logits.argmax(-1), torch.arange(10).unsqueeze(0))


def test_multidigit_binding_is_row_permutation_invariant_and_differentiable():
    task = DigitTaskConfig(table_size=4, digits=3, radix=10)
    model = build_model(task, d_model=16, rank=4, seed=5)
    batch = make_digit_batch(task, data_seed=7, split="train", start_index=0, batch_size=8)
    permutation = torch.tensor([2, 0, 3, 1])
    permuted = DigitBatch(
        batch.row_sources[:, permutation],
        batch.row_digits[:, permutation],
        batch.query_keys,
        batch.targets,
    )
    assert torch.allclose(model.logits(batch), model.logits(permuted), atol=1e-6)
    loss = torch.nn.functional.cross_entropy(model.logits(batch).reshape(-1, 10), batch.targets.reshape(-1))
    loss.backward()
    assert model.bind.q_proj.weight.grad is not None
    assert model.bind.q_proj.weight.grad.norm() > 0
    assert model.bind.k_proj.weight.grad is not None
    assert model.bind.k_proj.weight.grad.norm() > 0
