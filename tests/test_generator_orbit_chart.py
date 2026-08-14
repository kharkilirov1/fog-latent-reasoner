import torch
from generator_orbit_chart_experiment import Config, CyclicChart


def test_orbit_chart_has_exactly_shared_phase_increment_before_training():
    model = CyclicChart(Config(), seed=0, arm='generator_orbit')
    phase = model.phases().detach()
    interior = phase[1:] - phase[:-1]
    assert torch.allclose(interior, interior[:1].expand_as(interior), atol=1e-6)
