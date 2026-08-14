from trajectory_only_commutant_recovery_experiment import run_one


def test_trajectory_only_recovery_can_compile_from_noisy_hidden_state_pairs():
    row = run_one(
        multiplicity=3,
        seed=17,
        sample_factor=2.0,
        noise=0.05,
        ridge=1e-4,
        gap_threshold=5.0,
        depth=64,
        examples=256,
    )
    assert row["compiler"]["accepted"]
    assert row["compiler"]["best"]["multiplicity"] == 3
    assert row["compiled_shared_execution"]["accuracy"] > 0.99


def test_one_width_of_trace_probes_is_insufficient_for_this_pipeline():
    row = run_one(
        multiplicity=4,
        seed=19,
        sample_factor=1.0,
        noise=0.05,
        ridge=1e-4,
        gap_threshold=5.0,
        depth=32,
        examples=128,
    )
    assert not row["compiler"]["accepted"]
