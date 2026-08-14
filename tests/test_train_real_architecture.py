from train_real import new_model_config, parser


def test_real_training_new_model_defaults_to_binding_v2():
    init_args = parser().parse_args(["init-model"])
    assert init_args.architecture == "query_bound_v2"

    pretrain_args = parser().parse_args(
        ["pretrain", "--checkpoint-dir", "unused"]
    )
    assert pretrain_args.architecture == "auto"
    config = new_model_config(
        pretrain_args.architecture,
        vocab_size=8192,
        max_seq_len=512,
        reasoning_steps=4,
        dropout=0.1,
    )
    assert config.architecture_version == "query_bound_v2"
    assert config.binding_offsets == (2,)
    assert config.effective_memory_slots() == config.latent_slots == 4


def test_real_training_can_still_request_legacy_explicitly():
    config = new_model_config(
        "legacy_v1",
        vocab_size=8192,
        max_seq_len=512,
        reasoning_steps=4,
        dropout=0.1,
    )
    assert config.architecture_version == "legacy_v1"
    assert config.effective_memory_slots() == 8


def test_real_training_accepts_model_ready_register_machine_v3():
    args = parser().parse_args(
        ["init-model", "--architecture", "register_machine_v3", "--reasoning-steps", "8"]
    )
    assert args.architecture == "register_machine_v3"
    config = new_model_config(
        args.architecture,
        vocab_size=8192,
        max_seq_len=512,
        reasoning_steps=8,
        dropout=0.1,
    )
    assert config.architecture_version == "register_machine_v3"
    assert config.machine_hard_routing is True
    assert config.machine_operator_count == 4
    assert config.effective_memory_slots() == 4
