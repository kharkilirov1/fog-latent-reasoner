from nonlinear_blackbox_structural_compiler_experiment import (
    NonlinearDihedralBlackBox,
    discover_orbit,
    global_linear_fit_error,
    infer_edge_graph,
    permutation_cycles,
    run_one,
)


def test_blackbox_discovers_finite_orbit_and_graph_without_semantic_labels():
    box = NonlinearDihedralBlackBox()
    states = discover_orbit(box)
    edges = infer_edge_graph(box, states)
    assert states.size(0) == 14
    assert sorted(len(c) for c in permutation_cycles(edges["op0"])) == [7, 7]
    assert sorted(len(c) for c in permutation_cycles(edges["op1"])) == [2] * 7
    mismatch = global_linear_fit_error(box, states)
    assert mismatch["op0"] > 0.10
    assert mismatch["op1"] > 0.08


def test_blackbox_local_jacobian_compiler_recovers_long_horizon_tangent_dynamics():
    row = run_one(seed=31, noise=0.05, depth=64, examples=256)
    assert all(row["law_accepted"].values())
    assert row["compiled_tangent_execution"]["fraction_cosine_gt_0_99"] > 0.99
