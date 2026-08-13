import argparse

import torch

from matched_lookup_experiment import (
    FOG_VARIANTS,
    KEY_BASE,
    MAPS_TO,
    QUERY,
    ROW,
    ROW_END,
    TABLE_END,
    TASK,
    LookupTaskConfig,
    build_model,
    common_initialization_report,
    evaluate_model,
    make_batch,
    make_example,
    model_config,
    model_logits,
    shared_initialization_digest,
    verify_mapping_holdout,
)


def test_serialization_target_and_determinism():
    task = LookupTaskConfig(table_size=5)
    first = make_example(task, data_seed=11, split="train", sample_index=7)
    again = make_example(task, data_seed=11, split="train", sample_index=7)
    assert first == again
    assert len(first.prompt_ids) == task.prompt_length
    assert first.prompt_ids[0] == TASK
    assert first.prompt_ids[-3:-1] == (TABLE_END, QUERY)
    assert first.prompt_ids[-1] == task.key_token(first.query_key)
    assert first.target_id == task.value_token(first.mapping[first.query_key])
    for row_index in range(task.table_size):
        row = first.prompt_ids[1 + 5 * row_index : 1 + 5 * (row_index + 1)]
        assert row[0] == ROW
        assert KEY_BASE <= row[1] < task.key_end
        assert row[2] == MAPS_TO
        assert KEY_BASE <= row[3] < task.key_end
        assert row[4] == ROW_END
    # Standard associative recall uses one token identity for a state whether
    # that state appears on the source or destination side of a row.
    assert task.value_base == KEY_BASE
    assert task.key_token(3) == task.value_token(3)
    assert task.vocab_size == KEY_BASE + task.table_size


def test_legacy_separate_key_value_token_ablation():
    task = LookupTaskConfig(table_size=5, separate_key_value_tokens=True)
    assert task.value_base == KEY_BASE + task.table_size
    assert task.key_token(3) != task.value_token(3)
    assert task.vocab_size == KEY_BASE + 2 * task.table_size
    example = make_example(task, data_seed=13, split="train", sample_index=2)
    for row_index in range(task.table_size):
        row = example.prompt_ids[1 + 5 * row_index : 1 + 5 * (row_index + 1)]
        assert KEY_BASE <= row[1] < task.key_end
        assert task.value_base <= row[3] < task.vocab_size


def test_three_way_split_is_mapping_disjoint():
    task = LookupTaskConfig(table_size=6)
    checked = verify_mapping_holdout(
        task,
        data_seed=19,
        train_examples=500,
        validation_examples=500,
        test_examples=500,
    )
    assert checked["overlap"] == {
        "train_validation": 0,
        "train_test": 0,
        "validation_test": 0,
    }
    assert all(count > 0 for count in checked["unique_mappings"].values())


def test_shared_batches_and_name_stable_shared_initialization():
    task = LookupTaskConfig(table_size=4)
    cfg = model_config(
        task,
        d_model=16,
        n_heads=4,
        n_layers=1,
        d_ff=32,
        latent_slots=2,
        reasoning_steps=1,
        compare_rank=4,
        planner_ff=32,
        memory_slots=2,
    )
    batch_a = make_batch(
        task, data_seed=23, split="train", start_index=12, batch_size=8
    )
    batch_b = make_batch(
        task, data_seed=23, split="train", start_index=12, batch_size=8
    )
    assert torch.equal(batch_a.prompt_ids, batch_b.prompt_ids)
    assert torch.equal(batch_a.target_ids, batch_b.target_ids)

    models = {
        variant: build_model(variant, cfg, model_seed=31)
        for variant in ("direct", "fog_full", "fog_strict")
    }
    assert len({shared_initialization_digest(model) for model in models.values()}) == 1
    direct = models["direct"].state_dict()
    fog = models["fog_full"].state_dict()
    for name in direct:
        if name.startswith(("token.", "backbone.", "lm_head.")):
            assert torch.equal(direct[name], fog[name])


def test_per_arm_geometry_pairs_shape_compatible_tensor_intersection():
    task = LookupTaskConfig(table_size=4)
    common = dict(
        d_model=16,
        n_heads=4,
        latent_slots=2,
        reasoning_steps=1,
        compare_rank=4,
        planner_ff=32,
        memory_slots=2,
    )
    direct_cfg = model_config(task, n_layers=2, d_ff=40, **common)
    fog_cfg = model_config(task, n_layers=1, d_ff=32, **common)
    # The real runner forces the deepest shared positional geometry.  It is
    # already identical here, but state this as a contract of the test.
    assert direct_cfg.max_seq_len == fog_cfg.max_seq_len
    models = {
        "direct": build_model("direct", direct_cfg, model_seed=53),
        "fog_full": build_model("fog_full", fog_cfg, model_seed=53),
        "fog_strict": build_model("fog_strict", fog_cfg, model_seed=53),
    }
    report = common_initialization_report(models)
    assert report["exact_match"] is True
    assert "backbone.blocks.0.attn.in_proj_weight" in report["canonical_tensor_names"]
    assert "backbone.out_norm.weight" in report["canonical_tensor_names"]
    assert "backbone.blocks.0.ff.0.weight" in report["excluded_shape_mismatch_names"]
    assert "backbone.blocks.1.attn.in_proj_weight" not in report["canonical_tensor_names"]


def test_all_conditions_forward_backward_and_fog_interventions():
    task = LookupTaskConfig(table_size=4)
    cfg = model_config(
        task,
        d_model=16,
        n_heads=4,
        n_layers=1,
        d_ff=32,
        latent_slots=2,
        reasoning_steps=1,
        compare_rank=4,
        planner_ff=32,
        memory_slots=2,
    )
    batch = make_batch(
        task, data_seed=37, split="train", start_index=0, batch_size=4
    )
    for variant in ("direct", "fog_full", "fog_strict"):
        model = build_model(variant, cfg, model_seed=41)
        logits = model_logits(model, variant, batch.prompt_ids)
        assert logits.shape == (4, task.vocab_size)
        loss = torch.nn.functional.cross_entropy(logits, batch.target_ids)
        loss.backward()
        assert torch.isfinite(loss)
        assert model.token.weight.grad is not None
        if variant in FOG_VARIANTS:
            zero = model_logits(
                model, variant, batch.prompt_ids, intervention="zero"
            )
            shuffled = model_logits(
                model, variant, batch.prompt_ids, intervention="shuffle"
            )
            assert zero.shape == logits.shape == shuffled.shape


def test_unseen_validation_stream_digest_repeats():
    task = LookupTaskConfig(table_size=4)
    cfg = model_config(
        task,
        d_model=16,
        n_heads=4,
        n_layers=1,
        d_ff=32,
        latent_slots=2,
        reasoning_steps=1,
        compare_rank=4,
        planner_ff=32,
        memory_slots=2,
    )
    model = build_model("fog_strict", cfg, model_seed=43)
    kwargs = dict(
        data_seed=47,
        eval_examples=12,
        eval_batch_size=4,
        device=torch.device("cpu"),
        split="validation",
    )
    normal = evaluate_model(model, "fog_strict", task, intervention="normal", **kwargs)
    zero = evaluate_model(model, "fog_strict", task, intervention="zero", **kwargs)
    shuffle = evaluate_model(model, "fog_strict", task, intervention="shuffle", **kwargs)
    assert normal["stream_sha256"] == zero["stream_sha256"] == shuffle["stream_sha256"]
    assert normal["split"] == "validation"
