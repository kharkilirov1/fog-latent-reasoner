import torch

from fog_lmw import (
    FOG_MACHINE_V3_10M_PARAMETER_COUNT,
    FOGLatentReasoner,
    fog_machine_v3_10m_config,
)


def tiny_machine(*, steps: int = 4) -> FOGLatentReasoner:
    cfg = fog_machine_v3_10m_config(
        vocab_size=64, max_seq_len=64, reasoning_steps=steps, dropout=0.0
    )
    cfg.d_model = 32
    cfg.n_heads = 4
    cfg.n_layers = 1
    cfg.d_ff = 64
    cfg.compare_rank = 8
    cfg.planner_ff = 64
    cfg.machine_operator_rank = 8
    cfg.machine_ff = 64
    cfg.n_reasoning_modes = 2
    cfg.binding_offsets = (1,)
    cfg.diversity_weight = 0.0
    return FOGLatentReasoner(cfg)


def test_machine_10m_preset_parameter_count_and_contract():
    cfg = fog_machine_v3_10m_config(dropout=0.0)
    model = FOGLatentReasoner(cfg)
    assert cfg.architecture_version == "register_machine_v3"
    assert cfg.binding_query_update == "primary_recurrent"
    assert cfg.latent_slots == 4
    assert cfg.machine_operator_count == 4
    assert sum(p.numel() for p in model.parameters()) == FOG_MACHINE_V3_10M_PARAMETER_COUNT


def test_machine_first_tick_preserves_proposal_then_routes_generated_candidates():
    torch.manual_seed(10)
    model = tiny_machine(steps=2).eval()
    prompt = torch.tensor([[4, 5, 6, 7], [8, 9, 10, 11]])
    first, first_aux = model.transition_memory(prompt, None, return_diagnostics=True)
    assert first.shape == (2, 4, 32)
    first_machine = first_aux["planner"]["machine"]
    assert first_machine["initialized_from_query"] is True
    assert first_machine["operator_probs"].shape == (2, 7)
    assert torch.equal(first_machine["selected_operator"], torch.zeros(2, dtype=torch.long))

    second, second_aux = model.transition_memory(prompt, first, return_diagnostics=True)
    machine = second_aux["planner"]["machine"]
    assert machine["initialized_from_query"] is False
    probs = machine["operator_probs"]
    assert probs.shape == (2, 7)  # READ + IDENTITY + BLOCK_PRODUCT + 4 generated operators
    torch.testing.assert_close(probs.sum(dim=-1), torch.ones(2))
    assert second.shape == first.shape


def test_machine_loss_backpropagates_into_generated_operator_bank():
    torch.manual_seed(20)
    model = tiny_machine(steps=3)
    prompt = torch.tensor([[4, 5, 6, 7], [8, 9, 10, 11]])
    answers = torch.tensor([[1, 20, 21], [1, 22, 23]])
    loss, aux = model.loss(prompt, answers)
    assert torch.isfinite(loss)
    loss.backward()
    grads = [
        p.grad for p in model.machine_cell.operator_bank.parameters()
        if p.requires_grad and p.grad is not None
    ]
    assert grads
    assert any(torch.isfinite(g).all() and g.abs().max() > 0 for g in grads)
    assert aux["halt_probabilities"].shape == (2, 3)


@torch.inference_mode()
def test_machine_adaptive_halting_freezes_after_min_steps():
    torch.manual_seed(30)
    model = tiny_machine(steps=6).eval()
    # Force a deterministic HALT decision; this tests execution semantics, not training.
    model.machine_cell.halt_head.weight.zero_()
    model.machine_cell.halt_head.bias.fill_(20.0)
    prompt = torch.tensor([[4, 5, 6, 7], [8, 9, 10, 11]])
    memory, aux = model.reason_adaptive(
        prompt, max_steps=6, min_steps=2, halt_threshold=0.9
    )
    assert memory.shape == (2, 4, 32)
    assert torch.equal(aux["steps_used"], torch.tensor([2, 2]))
    assert bool(torch.all(aux["halted"]))
    assert aux["halt_probabilities"].shape[1] == 2


def test_block_product_primitive_is_exact_and_recurrent_in_a_compatible_chart():
    import math
    from fog_lmw.registers import GeneratedOperatorBank

    n_states = 7
    d_model = 8
    x = torch.arange(n_states, dtype=torch.float32)[:, None]
    k = torch.arange(1, d_model // 2 + 1, dtype=torch.float32)[None, :]
    angle = 2 * math.pi * x * k / n_states
    code = torch.stack([torch.cos(angle), torch.sin(angle)], dim=-1).reshape(n_states, d_model)
    code = torch.nn.functional.normalize(code, dim=-1) * math.sqrt(d_model)

    bank = GeneratedOperatorBank(d_model, 1, 4, hard_routing=True).eval()
    with torch.no_grad():
        final = bank.router[-1]
        final.weight.zero_()
        final.bias.fill_(-10.0)
        final.bias[2] = 10.0  # BLOCK_PRODUCT
    state_id = torch.tensor([2, 5])
    value = code[state_id]
    control = torch.zeros_like(value)
    target = state_id.clone()
    for operand_id in (torch.tensor([3, 1]), torch.tensor([6, 4]), torch.tensor([1, 2])):
        value, stats = bank(value, code[operand_id], control)
        target = (target + operand_id) % n_states
        assert torch.equal(stats["selected_operator"], torch.full((2,), 2))
        logits = torch.nn.functional.normalize(value, dim=-1) @ torch.nn.functional.normalize(code, dim=-1).T
        assert torch.equal(logits.argmax(dim=-1), target)
        torch.testing.assert_close(
            torch.nn.functional.cosine_similarity(value, code[target], dim=-1),
            torch.ones(2),
            rtol=1e-5,
            atol=1e-5,
        )
