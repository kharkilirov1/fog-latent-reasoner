import numpy as np

from latent_program_counter_experiment import (
    PCConfig,
    fit_local_successor,
    fourier_codebook,
    make_programs,
    successor_metrics,
)


def test_unique_halt_with_real_post_halt_distractors():
    cfg = PCConfig()
    length, program, _, _ = make_programs(
        cfg, min_length=5, max_length=10, examples=256, seed=123
    )
    assert np.all(np.sum(program == 4, axis=1) == 1)
    for i, l in enumerate(length):
        assert program[i, l] == 4
        if l + 1 < cfg.pc_modulus:
            assert np.all(program[i, l + 1 :] < 4)


def test_fourier_local_successor_generalizes_beyond_seen_prefix():
    cfg = PCConfig()
    code = fourier_codebook(cfg.pc_modulus, cfg.pc_harmonics)
    w = fit_local_successor(code, cfg.train_last_source, cfg.ridge)
    metrics = successor_metrics(code, w, cfg)
    assert metrics["seen_transition_accuracy"] == 1.0
    assert metrics["unseen_transition_accuracy"] == 1.0
