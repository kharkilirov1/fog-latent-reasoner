from multichart_operator_specificity_experiment import (
    MultiChartConfig,
    primitive_root,
    run_arm,
)


def test_primitive_root_covers_nonzero_field():
    cfg = MultiChartConfig()
    g = primitive_root(cfg.prime)
    assert len({pow(g, k, cfg.prime) for k in range(cfg.prime - 1)}) == cfg.prime - 1


def test_native_charts_make_native_group_operations_exact():
    cfg = MultiChartConfig()
    add = run_arm(cfg, chart="additive", operation="add", mode="local", split_seed=101)
    mul = run_arm(cfg, chart="multiplicative", operation="mul", mode="local", split_seed=101)
    assert add["heldout_pairs"]["accuracy"] == 1.0
    assert mul["heldout_pairs"]["accuracy"] == 1.0
