from highdim_jvp_relation_discovery_experiment import Config, run


def test_jvp_relation_discovery_finds_affine_group_laws_without_full_jacobian():
    result = run(Config(dim=32, probes=4, states=2, seed=7, perturbation=0.0))
    assert result["discovered"] == {
        "order_A": 31,
        "order_M": 30,
        "conjugacy_exponent": 3,
    }
    for score in result["best_scores"].values():
        assert score["state_relative_rms"] < 1e-9
        assert score["jvp_relative_rms"] < 1e-8
    check = result["full_jacobian_validation"]
    assert check["full_relative_frobenius"] < 1e-8
    assert check["jvp_relative_rms_128"] < 1e-8
