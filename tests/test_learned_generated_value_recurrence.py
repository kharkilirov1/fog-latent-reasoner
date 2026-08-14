import numpy as np

from learned_generated_value_recurrence_experiment import (
    fit_operator,
    make_heldout_chains,
    run_chain,
)
from operator_compatible_geometry_experiment import GeometryConfig, make_fourier_codebook


def test_exp005_fourier_local_reuses_generated_value_on_heldout_transitions():
    cfg = GeometryConfig()
    code = make_fourier_codebook(cfg)
    w, pairs, heldout, _ = fit_operator(cfg, code, split_seed=101, mode="local")
    start, operands, target = make_heldout_chains(
        cfg, pairs, heldout, depth=12, examples=256, seed=999
    )
    z, history = run_chain(
        cfg, code, w, mode="local", start=start, operands=operands
    )
    pred = (z @ code.T).argmax(axis=1)
    assert np.mean(pred == target) == 1.0
    assert all(row["accuracy"] == 1.0 for row in history)
    assert min(row["target_cosine_min"] for row in history) > 0.999999
