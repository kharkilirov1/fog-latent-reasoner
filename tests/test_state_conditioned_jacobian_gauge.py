from state_conditioned_jacobian_gauge_experiment import make_system, run_one, global_matrix_mismatch


def test_state_conditioned_system_has_no_single_good_observed_global_matrix():
    system = make_system(contexts=7, seed=23, noise=0.0)
    mismatch = global_matrix_mismatch(system["exact_local"])
    assert mismatch["A"] > 0.5
    assert mismatch["B"] > 0.5


def test_local_jacobian_gauge_sync_and_cycle_projection_stabilize_recurrence():
    row = run_one(contexts=7, seed=24, noise=0.10, depth=64, examples=256)
    assert row["cycle_law_projection_applied"]
    assert row["synchronized_law_execution"]["fraction_cosine_gt_0_99"] > 0.99
    assert row["projected_shared_laws"]["A_cycle_residual"] < 1e-10
