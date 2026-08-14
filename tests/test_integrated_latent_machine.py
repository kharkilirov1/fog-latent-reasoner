from integrated_latent_machine_experiment import evaluate_arm
from latent_program_counter_experiment import PCConfig
from operator_compatible_geometry_experiment import GeometryConfig


def test_structured_integrated_machine_executes_ood_programs():
    row = evaluate_arm(
        value_cfg=GeometryConfig(),
        pc_cfg=PCConfig(),
        value_geometry="fourier",
        alu_mode="local",
        pc_mode="local",
        split_seed=101,
        value_code_seed=11,
        min_length=5,
        max_length=10,
        examples=512,
        seed=123456,
    )
    assert row["value_accuracy"] == 1.0
    assert row["halt_step_accuracy"] == 1.0
    assert row["halt_position_accuracy"] == 1.0
    assert min(row["value_hop_accuracy"]) == 1.0
