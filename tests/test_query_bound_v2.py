from dataclasses import asdict

import pytest
import torch

from fog_lmw import FOGReasonerConfig, FOGLatentReasoner


def tiny_v2(
    *,
    steps: int = 3,
    memory_slots: int = 2,
    binding_offsets: tuple[int, ...] = (1, 2),
) -> FOGLatentReasoner:
    return FOGLatentReasoner(
        FOGReasonerConfig(
            vocab_size=64,
            d_model=32,
            n_heads=4,
            n_layers=1,
            d_ff=64,
            max_seq_len=40,
            dropout=0.0,
            latent_slots=2,
            reasoning_steps=steps,
            compare_rank=8,
            planner_ff=64,
            memory_slots=memory_slots,
            n_reasoning_modes=2,
            diversity_weight=0.0,
            architecture_version="query_bound_v2",
            binding_mode="query_conditioned",
            readout_mode="direct_latent",
            protected_binding_slots=1,
            binding_offsets=binding_offsets,
        )
    )


def test_old_config_without_architecture_fields_stays_legacy():
    config = asdict(tiny_v2().cfg)
    for name in (
        "architecture_version",
        "binding_mode",
        "readout_mode",
        "protected_binding_slots",
        "binding_offsets",
    ):
        config.pop(name)
    # Simulate the old format by setting no new keys on a genuinely legacy
    # geometry; dataclass defaults must select legacy semantics.
    restored = FOGReasonerConfig(**config)
    assert restored.architecture_version == "legacy_v1"
    assert restored.binding_mode == "summary_slots"
    assert restored.readout_mode == "bos_decoder"
    assert restored.binding_offsets == ()


def test_unknown_architecture_version_fails_loudly():
    config = asdict(tiny_v2().cfg)
    config["architecture_version"] = "query_bound_v999"
    with pytest.raises(ValueError, match="architecture_version is unsupported"):
        FOGLatentReasoner(FOGReasonerConfig(**config))


def test_v2_reason_has_no_vocab_projection_and_protects_primary():
    torch.manual_seed(1)
    model = tiny_v2(steps=3, memory_slots=2)
    calls = []
    hook = model.lm_head.register_forward_hook(lambda *args: calls.append(1))
    prompt = torch.tensor([[4, 5, 6, 7], [8, 9, 10, 11]])
    memory, aux = model.reason(prompt, return_diagnostics=True)
    hook.remove()
    assert calls == []
    assert memory.shape == (2, 2, 32)
    assert [row["memory_size"] for row in aux["history"]] == [2, 2, 2]
    assert torch.equal(memory[:, 0], aux["primary_latent"])
    assert all(row["memory"]["protected_slots"] == 1 for row in aux["history"])


def test_v2_reuses_exactly_k_slots_across_depth_even_if_configured_n_is_larger():
    torch.manual_seed(11)
    model = tiny_v2(steps=4, memory_slots=7)
    prompt = torch.tensor([[4, 5, 6, 7], [8, 9, 10, 11]])
    memory, aux = model.reason(prompt, return_diagnostics=True)
    assert model.cfg.effective_memory_slots() == model.cfg.latent_slots == 2
    assert memory.shape == (2, model.cfg.latent_slots, model.cfg.d_model)
    assert [row["memory_size"] for row in aux["history"]] == [2, 2, 2, 2]
    assert [row["memory"]["reused"] for row in aux["history"]] == [
        False,
        True,
        True,
        True,
    ]
    assert torch.equal(memory[:, 0], aux["primary_latent"])


def test_first_target_ignores_bos_id_and_gradients_reach_binder_reader():
    torch.manual_seed(2)
    model = tiny_v2(steps=2, memory_slots=2)
    prompt = torch.tensor([[4, 5, 6, 7], [8, 9, 10, 11]])
    answer_a = torch.tensor([[1, 20], [1, 21]])
    answer_b = torch.tensor([[3, 20], [3, 21]])
    loss_a, aux_a = model.loss(prompt, answer_a)
    loss_b, aux_b = model.loss(prompt, answer_b)
    torch.testing.assert_close(loss_a, loss_b, rtol=0, atol=0)
    torch.testing.assert_close(aux_a["readout_state"], aux_b["readout_state"], rtol=0, atol=0)
    loss_a.backward()
    for parameter in (
        model.planner.bind.q_proj.weight,
        model.planner.bind.k_proj.weight,
    ):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.norm() > 0


@torch.inference_mode()
def test_cosine_tied_readout_exactly_copies_every_tiny_vocab_code():
    torch.manual_seed(23)
    model = tiny_v2(steps=1, memory_slots=2).eval()
    token_ids = torch.arange(model.cfg.vocab_size)
    # Exercise the same protected-payload normalization used by the planner,
    # then the real final vocabulary projection.
    oracle_primary = model.planner.bind_norm(model.token(token_ids))
    predicted = model.direct_vocab_logits(oracle_primary).argmax(dim=-1)
    assert torch.equal(predicted, token_ids)


@torch.inference_mode()
def test_v2_padding_invariance_for_memory_and_first_token():
    torch.manual_seed(3)
    model = tiny_v2(steps=2, memory_slots=2).eval()
    variants = (
        (torch.tensor([[5, 6, 7, 8]]), torch.tensor([[1, 1, 1, 1]])),
        (torch.tensor([[5, 6, 7, 8, 0, 0]]), torch.tensor([[1, 1, 1, 1, 0, 0]])),
        (torch.tensor([[0, 0, 5, 6, 7, 8]]), torch.tensor([[0, 0, 1, 1, 1, 1]])),
    )
    outputs = []
    for prompt, mask in variants:
        memory, aux = model.reason(prompt, prompt_attention_mask=mask)
        logits = model.direct_vocab_logits(aux["primary_latent"])
        outputs.append((memory, logits))
    for memory, logits in outputs[1:]:
        torch.testing.assert_close(memory, outputs[0][0], rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(logits, outputs[0][1], rtol=1e-5, atol=1e-6)


@torch.inference_mode()
def test_primary_binding_is_row_permutation_and_masked_separator_invariant():
    """Only valid address/payload pairs may affect the protected carrier.

    Rows are serialized as ``key, value, masked-separator`` and the final
    valid token is the query.  With offset one, permuting complete rows must
    only permute binding candidates; changing masked separator IDs must be a
    semantic no-op.
    """

    torch.manual_seed(44)
    model = tiny_v2(
        steps=3,
        memory_slots=7,
        binding_offsets=(1,),
    ).eval()
    mask = torch.tensor([[1, 1, 0, 1, 1, 0, 1, 1, 0, 1]])
    base = torch.tensor([[10, 20, 3, 11, 21, 3, 12, 22, 3, 10]])
    permuted = torch.tensor([[12, 22, 3, 10, 20, 3, 11, 21, 3, 10]])
    noisy_padding = torch.tensor([[10, 20, 55, 11, 21, 56, 12, 22, 57, 10]])

    def primary_and_logits(prompt: torch.Tensor):
        _, aux = model.reason(prompt, prompt_attention_mask=mask)
        primary = aux["primary_latent"]
        return primary, model.direct_vocab_logits(primary)

    reference_primary, reference_logits = primary_and_logits(base)
    for variant in (permuted, noisy_padding):
        primary, logits = primary_and_logits(variant)
        torch.testing.assert_close(primary, reference_primary, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(logits, reference_logits, rtol=1e-5, atol=1e-6)
