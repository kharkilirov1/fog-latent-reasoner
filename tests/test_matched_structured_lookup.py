import torch

from fog_lmw.checkpoint import atomic_torch_save
from matched_structured_lookup_experiment import (
    DirectBOSStructuredLookup,
    DirectHiddenMemoryStructuredLookup,
    DirectStructuredLookup,
    FOGStructuredLookup,
    StructuredModelConfig,
    StructuredTaskConfig,
    build_model,
    common_initialization_report,
    evaluate_checkpoint,
    evaluate_model,
    make_batch,
    make_example,
    model_logits,
    parameter_report,
    target_deranged_indices,
    verify_mapping_holdout,
)


def tiny_config(task: StructuredTaskConfig) -> StructuredModelConfig:
    return StructuredModelConfig(
        d_model=16,
        n_heads=4,
        n_layers=1,
        d_ff=32,
        max_seq_len=16,
        latent_slots=2,
        reasoning_steps=1,
        compare_rank=4,
        planner_ff=32,
        memory_slots=2,
        n_reasoning_modes=2,
    )


def test_mapping_rows_are_one_vector_and_query_is_last():
    task = StructuredTaskConfig(table_size=5)
    cfg = tiny_config(task)
    example = make_example(task, data_seed=11, split="train", sample_index=7)
    again = make_example(task, data_seed=11, split="train", sample_index=7)
    assert example == again
    assert len(example.row_sources) == task.table_size
    assert sorted(example.row_sources) == list(range(task.table_size))
    assert example.row_values == tuple(
        example.mapping[source] for source in example.row_sources
    )
    assert example.target_state == example.mapping[example.query_key]

    batch = make_batch(task, data_seed=11, split="train", start_index=7, batch_size=3)
    model = build_model("direct", task, cfg, model_seed=13)
    assert isinstance(model, DirectStructuredLookup)
    prompt = model.encode_prompt(batch)
    assert prompt.shape == (3, task.table_size + 1, cfg.d_model)
    expected_rows = (
        model.key(batch.row_sources)
        + model.value(batch.row_values)
        + model.row_type
    )
    expected_query = model.key(batch.query_keys) + model.query_type.squeeze(1)
    assert torch.equal(prompt[:, :-1], expected_rows)
    assert torch.equal(prompt[:, -1], expected_query)


def test_mapping_operator_splits_are_disjoint():
    report = verify_mapping_holdout(
        StructuredTaskConfig(table_size=6),
        data_seed=17,
        train_examples=400,
        validation_examples=400,
        test_examples=400,
    )
    assert report["overlap"] == {
        "train_validation": 0,
        "train_test": 0,
        "validation_test": 0,
    }
    assert all(value > 0 for value in report["unique_mappings"].values())


def test_name_stable_shared_state_is_exact_and_parameter_delta_reported():
    task = StructuredTaskConfig(table_size=4)
    cfg = tiny_config(task)
    models = {
        variant: build_model(variant, task, cfg, model_seed=19)
        for variant in ("direct", "fog_full", "fog_strict")
    }
    report = common_initialization_report(models)
    assert report["exact_match"] is True
    assert "key.weight" in report["shared_tensor_names"]
    assert "value.weight" in report["shared_tensor_names"]
    assert "classifier.weight" in report["shared_tensor_names"]
    assert "backbone.blocks.0.attn.in_proj_weight" in report["shared_tensor_names"]

    counts = parameter_report(models)
    assert counts["direct"]["fog_only"] == 0
    assert counts["fog_full"]["fog_only"] > 0
    assert counts["fog_full"] == counts["fog_strict"]
    assert counts["direct"]["shared_structured_stack"] == counts["fog_full"]["shared_structured_stack"]


def test_direct_bos_matches_fog_answer_interface_without_latent_modules():
    task = StructuredTaskConfig(table_size=4)
    cfg = tiny_config(task)
    models = {
        variant: build_model(variant, task, cfg, model_seed=20)
        for variant in ("direct_bos", "fog_full", "fog_strict")
    }
    direct_bos = models["direct_bos"]
    assert isinstance(direct_bos, DirectBOSStructuredLookup)
    names = dict(direct_bos.named_parameters())
    assert "answer_bos" in names
    assert not any(name.startswith(("planner.", "memory.", "neutral")) for name in names)

    batch = make_batch(task, data_seed=23, split="train", start_index=0, batch_size=8)
    prompt = direct_bos.encode_prompt(batch)
    joined = torch.cat(
        [prompt, direct_bos.answer_bos.expand(batch.targets.size(0), 1, -1)],
        dim=1,
    )
    kinds = torch.zeros(joined.shape[:2], dtype=torch.long)
    expected = direct_bos.classifier(
        direct_bos.backbone.forward_embeds(joined, kinds)[:, -1]
    )
    assert joined.size(1) == task.prompt_length + 1
    assert torch.equal(direct_bos.logits(batch), expected)

    pairing = common_initialization_report(models)
    assert pairing["exact_match"] is True
    assert "answer_bos" in pairing["shared_tensor_names"]
    assert torch.equal(
        direct_bos.answer_bos,
        models["fog_full"].answer_bos,
    )
    counts = parameter_report(models)
    assert counts["direct_bos"]["fog_only"] == 0
    assert counts["direct_bos"]["answer_bos"] == cfg.d_model
    assert counts["direct_bos"]["shared_structured_stack"] == counts["fog_full"]["shared_structured_stack"]


def test_direct_hidden_memory_is_lossless_two_pass_strict_control():
    task = StructuredTaskConfig(table_size=4)
    cfg = tiny_config(task)
    models = {
        variant: build_model(variant, task, cfg, model_seed=22)
        for variant in ("direct_hidden_memory", "fog_full", "fog_strict")
    }
    bridge = models["direct_hidden_memory"]
    assert isinstance(bridge, DirectHiddenMemoryStructuredLookup)
    names = dict(bridge.named_parameters())
    assert "neutral" in names
    assert "answer_bos" in names
    assert not any(name.startswith(("planner.", "memory.")) for name in names)

    pairing = common_initialization_report(models)
    assert pairing["exact_match"] is True
    assert "neutral" in pairing["shared_tensor_names"]
    assert "answer_bos" in pairing["shared_tensor_names"]
    assert torch.equal(bridge.neutral, models["fog_strict"].neutral)
    assert torch.equal(bridge.answer_bos, models["fog_strict"].answer_bos)

    batch = make_batch(task, data_seed=23, split="train", start_index=0, batch_size=8)
    prompt = bridge.encode_prompt(batch)
    hidden_memory = bridge.encode_hidden_memory(batch)
    assert hidden_memory.shape == prompt.shape
    assert bridge.decode_hidden_memory(hidden_memory).shape == (8, task.table_size)
    assert torch.equal(bridge.logits(batch), bridge.decode_hidden_memory(hidden_memory))

    loss = torch.nn.functional.cross_entropy(bridge.logits(batch), batch.targets)
    loss.backward()
    assert bridge.value.weight.grad is not None
    assert bridge.backbone.blocks[0].attn.in_proj_weight.grad is not None
    counts = parameter_report(models)
    assert counts["direct_hidden_memory"]["fog_only"] == 0
    assert counts["direct_hidden_memory"]["total"] < counts["fog_strict"]["total"]


def test_fixed_orthogonal_key_calibration_is_paired_and_frozen():
    task = StructuredTaskConfig(table_size=4)
    cfg = tiny_config(task)
    cfg = StructuredModelConfig(**{**cfg.__dict__, "fixed_orthogonal_keys": True})
    models = {
        variant: build_model(variant, task, cfg, model_seed=21)
        for variant in ("direct", "fog_full", "fog_strict")
    }
    expected = torch.zeros(task.table_size, cfg.d_model)
    expected[:, : task.table_size] = torch.eye(task.table_size)
    for model in models.values():
        assert torch.equal(model.key.weight, expected)
        assert model.key.weight.requires_grad is False
    assert common_initialization_report(models)["exact_match"] is True


def test_target_shuffle_is_a_true_different_target_permutation():
    targets = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])
    donors = target_deranged_indices(targets)
    assert sorted(donors.tolist()) == list(range(targets.numel()))
    assert torch.all(targets.index_select(0, donors).ne(targets))
    assert torch.equal(donors, target_deranged_indices(targets))


def test_all_arms_forward_backward_and_strict_decoder_geometry():
    task = StructuredTaskConfig(table_size=4)
    cfg = tiny_config(task)
    batch = make_batch(task, data_seed=23, split="train", start_index=0, batch_size=8)
    for variant in (
        "direct",
        "direct_bos",
        "direct_hidden_memory",
        "fog_full",
        "fog_strict",
    ):
        model = build_model(variant, task, cfg, model_seed=29)
        logits = model_logits(model, variant, batch)
        assert logits.shape == (8, task.table_size)
        loss = torch.nn.functional.cross_entropy(logits, batch.targets)
        loss.backward()
        assert torch.isfinite(loss)
        assert model.key.weight.grad is not None
        assert model.classifier.weight.grad is not None
        if variant in ("fog_full", "fog_strict"):
            assert isinstance(model, FOGStructuredLookup)
            assert any(
                parameter.grad is not None
                for parameter in model.planner.parameters()
            )
            for intervention in ("zero", "target_deranged_shuffle"):
                changed = model_logits(model, variant, batch, intervention=intervention)
                assert changed.shape == logits.shape

    strict = build_model("fog_strict", task, cfg, model_seed=31)
    assert isinstance(strict, FOGStructuredLookup)
    prompt = strict.encode_prompt(batch)
    memory = strict.reason_embeds(prompt)
    # Strict decode length is neutral + memory + BOS; it cannot directly see
    # any of the table or query vectors.
    assert 1 + memory.size(1) + 1 < prompt.size(1) + memory.size(1) + 1
    assert strict.decode_embeds("fog_strict", prompt, memory).shape == (8, 4)


def test_interventions_reuse_the_exact_validation_stream():
    task = StructuredTaskConfig(table_size=4)
    cfg = tiny_config(task)
    model = build_model("fog_strict", task, cfg, model_seed=37)
    kwargs = dict(
        data_seed=41,
        eval_examples=16,
        eval_batch_size=8,
        device=torch.device("cpu"),
        split="validation",
    )
    results = [
        evaluate_model(model, "fog_strict", task, intervention=intervention, **kwargs)
        for intervention in ("normal", "zero", "target_deranged_shuffle")
    ]
    assert len({row["stream_sha256"] for row in results}) == 1


def test_checkpoint_only_evaluation_loads_state_and_never_trains(tmp_path):
    task = StructuredTaskConfig(table_size=4)
    cfg = tiny_config(task)
    model = build_model("fog_strict", task, cfg, model_seed=43)
    checkpoint = tmp_path / "fog_strict.pt"
    atomic_torch_save(
        {
            "format_version": 1,
            "experiment": "matched_structured_permutation_lookup_v1",
            "variant": "fog_strict",
            "task_config": task.__dict__,
            "model_config": cfg.__dict__,
            "model_state_dict": {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            },
            "metrics": {
                "model_seed": 43,
                "training_stream_sha256": "unit-test-stream",
            },
        },
        checkpoint,
    )
    output = tmp_path / "validation.json"
    result = evaluate_checkpoint(
        checkpoint,
        data_seed=41,
        split="validation",
        eval_examples=16,
        eval_batch_size=8,
        device=torch.device("cpu"),
        output_path=output,
    )
    assert result["mode"] == "checkpoint_only_evaluation"
    assert result["split"] == "validation"
    assert set(result["eval"]) == {
        "normal",
        "zero",
        "target_deranged_shuffle",
    }
    assert len({row["stream_sha256"] for row in result["eval"].values()}) == 1
    assert result["checkpoint"]["training_stream_sha256"] == "unit-test-stream"
    assert output.exists()
