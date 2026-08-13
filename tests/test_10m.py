import torch
import torch.nn.functional as F

from fog_lmw import (
    FOG_BINDING_V2_10M_PARAMETER_COUNT,
    FOGReasonerConfig,
    FOG_10M_PARAMETER_COUNT,
    FOGLatentReasoner,
    fog_10m_config,
    fog_binding_v2_10m_config,
)


def small_masked_model(
    *,
    latent_slots: int = 2,
    memory_slots: int = 4,
    reasoning_steps: int = 3,
) -> FOGLatentReasoner:
    cfg = FOGReasonerConfig(
        vocab_size=64,
        d_model=32,
        n_heads=4,
        n_layers=1,
        d_ff=64,
        max_seq_len=32,
        dropout=0.0,
        latent_slots=latent_slots,
        reasoning_steps=reasoning_steps,
        compare_rank=8,
        planner_ff=64,
        memory_slots=memory_slots,
        n_reasoning_modes=2,
        diversity_weight=0.0,
        route_entropy_weight=0.0,
    )
    return FOGLatentReasoner(cfg)


def test_10m_preset_exact_unique_parameter_count_and_tied_head():
    cfg = fog_10m_config(dropout=0.0)
    model = FOGLatentReasoner(cfg)

    # K=4 is the provisional mechanistically supported slot count. N=8 makes
    # compression active at recurrent steps three and four.
    assert cfg.latent_slots == 4
    assert cfg.memory_slots == 8
    assert FOG_10M_PARAMETER_COUNT == 10_035_848

    parameters = list(model.parameters())
    unique_parameters = {id(parameter): parameter for parameter in parameters}
    assert len(parameters) == len(unique_parameters)
    assert sum(parameter.numel() for parameter in unique_parameters.values()) == (
        FOG_10M_PARAMETER_COUNT
    )

    assert model.lm_head.weight is model.token.weight
    assert model.lm_head.weight.data_ptr() == model.token.weight.data_ptr()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    optimizer_parameters = [
        parameter
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]
    assert len(optimizer_parameters) == len({id(p) for p in optimizer_parameters})
    assert sum(parameter is model.token.weight for parameter in optimizer_parameters) == 1


def test_binding_v2_10m_preset_is_distinct_exact_and_tied():
    cfg = fog_binding_v2_10m_config(dropout=0.0)
    model = FOGLatentReasoner(cfg)
    assert cfg.architecture_version == "query_bound_v2"
    assert cfg.binding_mode == "query_conditioned"
    assert cfg.readout_mode == "direct_latent"
    assert cfg.protected_binding_slots == 1
    assert sum(p.numel() for p in model.parameters()) == FOG_BINDING_V2_10M_PARAMETER_COUNT
    assert FOG_BINDING_V2_10M_PARAMETER_COUNT == 10_000_039
    assert model.lm_head.weight is model.token.weight


@torch.inference_mode()
def test_padding_invariance_for_reasoning_and_decoding():
    torch.manual_seed(10)
    model = small_masked_model().eval()
    answer_prefix = torch.tensor([[1, 13, 14]])

    variants = (
        (torch.tensor([[5, 6, 7, 8]]), torch.tensor([[1, 1, 1, 1]])),
        (
            torch.tensor([[5, 6, 7, 8, 0, 0]]),
            torch.tensor([[1, 1, 1, 1, 0, 0]]),
        ),
        (
            torch.tensor([[0, 0, 5, 6, 7, 8]]),
            torch.tensor([[0, 0, 1, 1, 1, 1]]),
        ),
    )

    outputs = []
    memories = []
    for prompt, mask in variants:
        memory, _ = model.reason(prompt, prompt_attention_mask=mask)
        logits = model.decode(
            prompt,
            memory,
            answer_prefix,
            prompt_attention_mask=mask,
        )
        memories.append(memory)
        outputs.append(logits)

    for memory in memories[1:]:
        torch.testing.assert_close(memory, memories[0], rtol=1e-5, atol=1e-6)
    for logits in outputs[1:]:
        torch.testing.assert_close(logits, outputs[0], rtol=1e-5, atol=1e-6)


def test_answer_only_logits_and_masked_cross_entropy_match_manual_value():
    torch.manual_seed(20)
    model = small_masked_model().eval()
    prompt = torch.tensor([[5, 6, 7, 0], [8, 9, 10, 11]])
    prompt_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]])
    answer = torch.tensor([[1, 12, 13, 0], [1, 14, 15, 16]])
    answer_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]])

    memory, _ = model.reason(prompt, prompt_attention_mask=prompt_mask)
    logits = model.decode(
        prompt,
        memory,
        answer[:, :-1],
        prompt_attention_mask=prompt_mask,
        decoder_attention_mask=answer_mask[:, :-1],
    )
    assert logits.shape == (2, answer.size(1) - 1, model.cfg.vocab_size)

    targets = answer[:, 1:].masked_fill(~answer_mask[:, 1:].bool(), -100)
    manual_ce = F.cross_entropy(
        logits.float().reshape(-1, logits.size(-1)), targets.reshape(-1)
    )
    loss, aux = model.loss(
        prompt,
        answer,
        prompt_attention_mask=prompt_mask,
        answer_attention_mask=answer_mask,
    )
    torch.testing.assert_close(aux["ce_loss"], manual_ce, rtol=0.0, atol=0.0)
    torch.testing.assert_close(loss, manual_ce, rtol=0.0, atol=0.0)


def test_memory_compressor_receives_finite_nonzero_gradient():
    torch.manual_seed(30)
    # 4 -> compress(8 -> 4) at step two, so the final answer necessarily has a
    # differentiable path through the learned compression module.
    model = small_masked_model(
        latent_slots=4,
        memory_slots=4,
        reasoning_steps=3,
    )
    prompt = torch.randint(3, model.cfg.vocab_size, (4, 7))
    answer = torch.tensor([[1, 20], [1, 21], [1, 22], [1, 23]])

    loss, _ = model(prompt, answer)
    loss.backward()

    compressor_parameters = list(model.memory.compress.named_parameters())
    assert compressor_parameters
    for name, parameter in compressor_parameters:
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
    assert any(parameter.grad.abs().max() > 0 for _, parameter in compressor_parameters)
