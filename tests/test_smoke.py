import torch

from fog_lmw import (
    FOGReasonerConfig,
    FOGLatentReasoner,
    allocate_equal_cost_discrete,
    compare_frobenius_error,
)


def tiny_model():
    cfg = FOGReasonerConfig(
        vocab_size=256,
        d_model=64,
        n_heads=4,
        n_layers=1,
        d_ff=128,
        max_seq_len=128,
        latent_slots=4,
        reasoning_steps=3,
        compare_rank=16,
        planner_ff=128,
        memory_slots=8,
        n_reasoning_modes=3,
    )
    return cfg, FOGLatentReasoner(cfg)


def test_reasoning_memory_and_loss():
    torch.manual_seed(0)
    cfg, model = tiny_model()
    prompt = torch.randint(3, cfg.vocab_size, (2, 10))
    answer = torch.tensor([[1, 5, 6, 7], [1, 8, 9, 10]])
    loss, aux = model.loss(prompt, answer, return_diagnostics=True)
    assert torch.isfinite(loss)
    assert len(aux["history"]) == cfg.reasoning_steps
    assert aux["history"][0]["latent"].shape == (2, cfg.latent_slots, cfg.d_model)
    # 4 -> 8 -> compress to 8
    assert [h["memory_size"] for h in aux["history"]] == [4, 8, 8]


def test_compare_rank_law_helper():
    sigma = torch.tensor([5.0, 4.0, 3.0, 2.0])
    e = compare_frobenius_error(sigma, rank=2)
    assert torch.allclose(e, torch.tensor((3.0**2 + 2.0**2) ** 0.5))


def test_discrete_allocator():
    # motif 0 initially has best sensitivity-weighted gain; motif 1 wins later
    out = allocate_equal_cost_discrete(
        gain_curves=[[4.0, 1.0], [2.0, 2.0]],
        sensitivity=[1.0, 1.0],
        total_units=3,
    )
    assert out == [1, 2]


def test_backward_reaches_planner_and_compressor():
    torch.manual_seed(1)
    cfg = FOGReasonerConfig(
        vocab_size=32,
        d_model=32,
        n_heads=4,
        n_layers=1,
        d_ff=64,
        max_seq_len=64,
        latent_slots=4,
        reasoning_steps=3,
        compare_rank=8,
        planner_ff=64,
        memory_slots=4,
        n_reasoning_modes=2,
    )
    model = FOGLatentReasoner(cfg)
    prompt = torch.randint(3, cfg.vocab_size, (4, 12))
    answer = torch.tensor([[1, 5], [1, 6], [1, 7], [1, 8]])
    loss, _ = model(prompt, answer, return_diagnostics=False)
    loss.backward()
    assert model.planner.query.grad is not None
    assert torch.isfinite(model.planner.query.grad).all()
    assert model.planner.query.grad.norm() > 0
    assert model.memory.compress.q_proj.weight.grad is not None
    assert torch.isfinite(model.memory.compress.q_proj.weight.grad).all()
    assert model.memory.compress.q_proj.weight.grad.norm() > 0


def test_explicit_reasoning_depth_and_length_guard():
    cfg, model = tiny_model()
    prompt = torch.randint(3, cfg.vocab_size, (2, 10))
    memory, aux = model.reason(
        prompt, reasoning_steps=1, return_diagnostics=True
    )
    assert memory.shape[1] == cfg.latent_slots
    assert len(aux["history"]) == 1

    model.cfg.max_seq_len = 12
    answer = torch.tensor([[1, 5], [1, 6]])
    try:
        model.loss(prompt, answer, reasoning_steps=1)
    except ValueError as exc:
        assert "exceeds max_seq_len" in str(exc)
    else:
        raise AssertionError("expected max_seq_len guard to fail")


def test_separate_decoder_prompt_memory_bottleneck():
    torch.manual_seed(2)
    cfg, model = tiny_model()
    prompt = torch.randint(3, cfg.vocab_size, (3, 10))
    decoder_prompt = prompt[:, :1]
    answer = torch.tensor([[1, 5], [1, 6], [1, 7]])
    loss, aux = model(
        prompt,
        answer,
        decoder_prompt_ids=decoder_prompt,
        reasoning_steps=2,
        return_diagnostics=False,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(aux["ce_loss"])
    assert model.planner.context_query[1].weight.grad is not None
    assert model.planner.context_query[1].weight.grad.norm() > 0
