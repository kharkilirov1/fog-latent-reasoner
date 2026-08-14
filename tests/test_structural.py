import torch

from fog_lmw.structural import jvp_gain_stats, relation_evidence


def test_jvp_gain_stats_tracks_linear_contraction():
    state = torch.randn(64, dtype=torch.float64)
    stats = jvp_gain_stats(lambda x: 0.5 * x, state, probes=8, seed=3)
    assert abs(stats.mean - 0.5) < 1e-12
    assert abs(stats.p95 - 0.5) < 1e-12


def test_relation_evidence_detects_exact_and_wrong_laws():
    states = [torch.randn(32, dtype=torch.float64) for _ in range(2)]
    exact = relation_evidence(lambda x: x + 1, lambda x: x + 1, states, probes=4)
    wrong = relation_evidence(lambda x: x + 1, lambda x: 0.8 * x + 1, states, probes=4)
    assert exact.state_relative_rms == 0
    assert exact.jvp_relative_rms == 0
    assert exact.accepted(state_threshold=1e-9, jvp_threshold=1e-9)
    assert wrong.state_relative_rms > 0.05
    assert wrong.jvp_relative_rms > 0.1
    assert not wrong.accepted(state_threshold=0.05, jvp_threshold=0.05)
