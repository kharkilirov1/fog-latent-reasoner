import math

import torch

from generated_value_gauge_experiment import (
    FourierLatentRegister,
    GaugeConfig,
    exact_gauge_order_residual,
    theoretical_exact_roots,
)


def test_semantic_add_creates_exact_new_canonical_value_without_lookup():
    cfg = GaugeConfig(modulus=31, harmonics=15)
    model = FourierLatentRegister(cfg, initial_phi=0.0).eval()
    a = torch.tensor([1, 7, 19, 30])
    b = torch.tensor([9, 13, 20, 4])
    z = model.semantic_add(model.encode(a), b)
    target = (a + b) % cfg.modulus
    assert torch.allclose(z, model.encode(target), atol=2e-5, rtol=2e-5)


def test_depth_two_terminal_cannot_distinguish_pi_gauge():
    cfg = GaugeConfig(modulus=31, harmonics=15)
    model = FourierLatentRegister(cfg, initial_phi=math.pi).eval()
    start = torch.tensor([2, 5, 11, 17])
    operands = torch.tensor([[3, 4], [7, 1], [8, 12], [6, 9]])
    final, history = model.run(start, operands, return_history=True)
    target = (start + operands.sum(1)) % cfg.modulus
    assert torch.equal(model.logits(final).argmax(-1), target)
    one_target = (start + operands[:, 0]) % cfg.modulus
    assert not torch.equal(model.logits(history[0]).argmax(-1), one_target)


def test_coprime_terminal_depths_have_only_identity_exact_scalar_gauge():
    theory = theoretical_exact_roots((2, 3))
    assert theory["gcd"] == 1
    assert theory["noncanonical_exact_roots_exist"] is False
    assert theory["exact_gauge_roots_mod_2pi"] == [0.0]


def test_order_residual_detects_expected_roots():
    assert exact_gauge_order_residual(math.pi, 2) < 1e-6
    assert exact_gauge_order_residual(math.pi, 1) > 1.9
    assert exact_gauge_order_residual(2 * math.pi / 3, 3) < 1e-6
