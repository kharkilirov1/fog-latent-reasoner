import numpy as np

from operator_compatible_geometry_experiment import (
    GeometryConfig,
    all_pairs,
    fit_ridge,
    local_features,
    make_fourier_codebook,
    pair_split,
)


def test_pair_split_holds_out_pairs_but_not_operand_identities():
    cfg = GeometryConfig()
    pairs, _ = all_pairs(cfg)
    train, test, _ = pair_split(cfg, pairs, split_seed=101)
    assert len(train) + len(test) == cfg.modulus ** 2
    for col in (0, 1):
        assert set(pairs[train, col]) == set(range(cfg.modulus))
        assert set(pairs[test, col]) == set(range(cfg.modulus))


def test_fourier_local_bilinear_features_recover_addition_on_all_pairs():
    cfg = GeometryConfig()
    pairs, target = all_pairs(cfg)
    train, test, _ = pair_split(cfg, pairs, split_seed=101)
    code = make_fourier_codebook(cfg)
    x = local_features(code, pairs[train])
    w = fit_ridge(x, code[target[train]], cfg.ridge)
    z = local_features(code, pairs[test]) @ w
    z /= np.linalg.norm(z, axis=1, keepdims=True)
    pred = (z @ code.T).argmax(axis=1)
    assert np.mean(pred == target[test]) == 1.0
    assert np.mean(np.sum(z * code[target[test]], axis=1)) > 0.999999
