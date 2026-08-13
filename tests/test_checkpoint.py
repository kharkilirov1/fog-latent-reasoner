import random

import torch

from export_checkpoint import export_checkpoint
from fog_lmw import FOGReasonerConfig, FOGLatentReasoner
from fog_lmw.checkpoint import (
    load_training_checkpoint,
    save_inference_checkpoint,
    save_training_checkpoint,
    sha256_file,
)


def checkpoint_model() -> FOGLatentReasoner:
    return FOGLatentReasoner(
        FOGReasonerConfig(
            vocab_size=48,
            d_model=24,
            n_heads=4,
            n_layers=1,
            d_ff=48,
            max_seq_len=32,
            dropout=0.2,
            latent_slots=2,
            reasoning_steps=2,
            compare_rank=6,
            planner_ff=48,
            memory_slots=2,
            n_reasoning_modes=2,
            diversity_weight=0.0,
            route_entropy_weight=0.0,
        )
    )


def v2_checkpoint_model() -> FOGLatentReasoner:
    return FOGLatentReasoner(
        FOGReasonerConfig(
            vocab_size=48,
            d_model=24,
            n_heads=4,
            n_layers=1,
            d_ff=48,
            max_seq_len=32,
            dropout=0.0,
            latent_slots=2,
            reasoning_steps=2,
            compare_rank=6,
            planner_ff=48,
            memory_slots=2,
            n_reasoning_modes=2,
            diversity_weight=0.0,
            route_entropy_weight=0.0,
            architecture_version="query_bound_v2",
            binding_mode="query_conditioned",
            readout_mode="direct_latent",
            protected_binding_slots=1,
            binding_offsets=(1, 2),
        )
    )


def make_optimizer_and_scheduler(model):
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=1, gamma=0.9
    )
    return optimizer, scheduler


def random_batch(vocab_size: int):
    # Consume both RNG streams covered by the checkpoint contract.
    python_offset = random.randrange(3, vocab_size)
    prompt = torch.randint(3, vocab_size, (3, 6))
    prompt[:, 0] = python_offset
    answer = torch.randint(3, vocab_size, (3, 3))
    answer[:, 0] = 1
    return python_offset, prompt, answer


def train_step(model, optimizer, scheduler):
    python_offset, prompt, answer = random_batch(model.cfg.vocab_size)
    optimizer.zero_grad(set_to_none=True)
    loss, _ = model(prompt, answer)
    loss.backward()
    optimizer.step()
    scheduler.step()
    return {
        "python_offset": python_offset,
        "prompt": prompt.clone(),
        "answer": answer.clone(),
        "loss": loss.detach().clone(),
    }


def test_checkpoint_resume_reproduces_the_exact_next_step(tmp_path):
    random.seed(1234)
    torch.manual_seed(1234)
    reference_model = checkpoint_model()
    reference_optimizer, reference_scheduler = make_optimizer_and_scheduler(
        reference_model
    )

    train_step(reference_model, reference_optimizer, reference_scheduler)
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer_path.write_text('{"version": 1}\n')
    checkpoint_path = tmp_path / "last.pt"
    save_training_checkpoint(
        checkpoint_path,
        model=reference_model,
        config=reference_model.cfg,
        optimizer=reference_optimizer,
        scheduler=reference_scheduler,
        scaler=None,
        global_step=1,
        consumed_tokens=18,
        tokenizer_path=tokenizer_path,
        training_args={"seed": 1234},
        extra={"sampler_position": 3},
    )

    expected_batch = train_step(
        reference_model, reference_optimizer, reference_scheduler
    )
    expected_state = {
        name: tensor.detach().clone()
        for name, tensor in reference_model.state_dict().items()
    }
    expected_lr = reference_optimizer.param_groups[0]["lr"]

    # Deliberately disturb both streams and consume RNG during initialization.
    random.seed(9999)
    torch.manual_seed(9999)
    resumed_model = checkpoint_model()
    resumed_optimizer, resumed_scheduler = make_optimizer_and_scheduler(resumed_model)
    payload = load_training_checkpoint(
        checkpoint_path,
        model=resumed_model,
        optimizer=resumed_optimizer,
        scheduler=resumed_scheduler,
        scaler=None,
        tokenizer_path=tokenizer_path,
        restore_rng=True,
    )
    assert payload["global_step"] == 1
    assert payload["consumed_tokens"] == 18
    assert payload["extra"]["sampler_position"] == 3

    actual_batch = train_step(resumed_model, resumed_optimizer, resumed_scheduler)
    assert actual_batch["python_offset"] == expected_batch["python_offset"]
    assert torch.equal(actual_batch["prompt"], expected_batch["prompt"])
    assert torch.equal(actual_batch["answer"], expected_batch["answer"])
    torch.testing.assert_close(
        actual_batch["loss"], expected_batch["loss"], rtol=0.0, atol=0.0
    )
    assert resumed_optimizer.param_groups[0]["lr"] == expected_lr
    assert resumed_scheduler.last_epoch == reference_scheduler.last_epoch
    for name, actual in resumed_model.state_dict().items():
        torch.testing.assert_close(actual, expected_state[name], rtol=0.0, atol=0.0)


def test_portable_inference_checkpoint_loads_without_training_state(tmp_path):
    torch.manual_seed(55)
    source = checkpoint_model()
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer_path.write_text('{"version": 1}\n')
    path = save_inference_checkpoint(
        tmp_path / "model.pt",
        model=source,
        config=source.cfg,
        global_step=7,
        consumed_tokens=123,
        tokenizer_path=tokenizer_path,
        metadata={"validation_loss": 1.25},
    )
    target = checkpoint_model()
    payload = load_training_checkpoint(
        path,
        model=target,
        tokenizer_path=tokenizer_path,
        restore_rng=False,
    )
    assert payload["checkpoint_kind"] == "inference"
    assert payload["global_step"] == 7
    assert payload["metadata"]["validation_loss"] == 1.25
    for name, actual in target.state_dict().items():
        torch.testing.assert_close(actual, source.state_dict()[name])


def test_format2_checkpoint_without_architecture_metadata_loads_as_legacy(tmp_path):
    """Released format-2 configs predate all v2 architecture fields."""

    torch.manual_seed(56)
    source = checkpoint_model().eval()
    path = save_inference_checkpoint(
        tmp_path / "modern-v1.pt",
        model=source,
        config=source.cfg,
        global_step=3,
        consumed_tokens=24,
        tokenizer_path=None,
    )
    payload = torch.load(path, map_location="cpu", weights_only=True)
    for field in (
        "architecture_version",
        "binding_mode",
        "readout_mode",
        "protected_binding_slots",
        "binding_offsets",
    ):
        payload["model_config"].pop(field)
    legacy_path = tmp_path / "released-format2.pt"
    torch.save(payload, legacy_path)

    restored_config = FOGReasonerConfig(**payload["model_config"])
    assert restored_config.architecture_version == "legacy_v1"
    target = FOGLatentReasoner(restored_config).eval()
    loaded = load_training_checkpoint(
        legacy_path,
        model=target,
        restore_rng=False,
    )
    assert loaded["format_version"] == 2
    for name, actual in target.state_dict().items():
        torch.testing.assert_close(actual, source.state_dict()[name], rtol=0, atol=0)


def test_v2_checkpoint_and_export_preserve_architecture_contract(tmp_path):
    torch.manual_seed(57)
    source = v2_checkpoint_model().eval()
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer_path.write_text('{"version": 2}\n')
    source_path = save_inference_checkpoint(
        tmp_path / "v2-source.pt",
        model=source,
        config=source.cfg,
        global_step=5,
        consumed_tokens=40,
        tokenizer_path=tokenizer_path,
        metadata={"stage": "v2-unit"},
    )
    source_payload = torch.load(source_path, map_location="cpu", weights_only=True)
    expected_contract = {
        "architecture_version": "query_bound_v2",
        "binding_mode": "query_conditioned",
        "readout_mode": "direct_latent",
        "protected_binding_slots": 1,
        "binding_offsets": (1, 2),
    }
    for field, expected in expected_contract.items():
        assert source_payload["model_config"][field] == expected

    output_path = tmp_path / "v2-exported.pt"
    parameters = sum(parameter.numel() for parameter in source.parameters())
    result = export_checkpoint(
        source_path,
        output_path,
        dtype_name="fp32",
        tokenizer_path=tokenizer_path,
        expected_parameters=parameters,
        max_logit_error=0.0,
    )
    assert result["ok"]
    exported = torch.load(output_path, map_location="cpu", weights_only=True)
    for field, expected in expected_contract.items():
        assert exported["model_config"][field] == expected
    assert exported["metadata"]["stage"] == "v2-unit"
    assert exported["metadata"]["export"]["lossy"] is False

    restored = v2_checkpoint_model().eval()
    load_training_checkpoint(
        output_path,
        model=restored,
        tokenizer_path=tokenizer_path,
        restore_rng=False,
    )
    for name, actual in restored.state_dict().items():
        torch.testing.assert_close(actual, source.state_dict()[name], rtol=0, atol=0)


def test_bf16_export_is_loadable_small_and_reproducible(tmp_path):
    torch.manual_seed(66)
    source = checkpoint_model().eval()
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer_path.write_text('{"version": 1}\n')
    source_path = save_inference_checkpoint(
        tmp_path / "source.pt",
        model=source,
        config=source.cfg,
        global_step=9,
        consumed_tokens=456,
        tokenizer_path=tokenizer_path,
        metadata={"validation_loss": 1.0},
    )
    output_path = tmp_path / "exported.pt"
    parameters = sum(parameter.numel() for parameter in source.parameters())
    first = export_checkpoint(
        source_path,
        output_path,
        tokenizer_path=tokenizer_path,
        expected_parameters=parameters,
    )
    first_hash = sha256_file(output_path)
    second = export_checkpoint(
        source_path,
        output_path,
        tokenizer_path=tokenizer_path,
        expected_parameters=parameters,
        force=True,
    )

    assert first["ok"] and second["ok"]
    assert first["parameters"] == parameters
    assert first["state_dict_dtype"] == "bfloat16"
    assert first["output_bytes"] < first["source_bytes"]
    assert first["max_logit_error"] <= 0.05
    assert sha256_file(output_path) == first_hash == second["output_sha256"]

    payload = torch.load(output_path, map_location="cpu", weights_only=True)
    floating_dtypes = {
        tensor.dtype
        for tensor in payload["model_state_dict"].values()
        if tensor.is_floating_point()
    }
    assert floating_dtypes == {torch.bfloat16}
    assert payload["global_step"] == 9
    assert payload["consumed_tokens"] == 456
    assert payload["metadata"]["validation_loss"] == 1.0
    assert payload["metadata"]["export"]["lossy"] is True
