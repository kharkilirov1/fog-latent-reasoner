import torch
from canonicalized_joint_algebra_experiment import Config, CanonicalizedLatentAlgebra


def test_canonicalized_operator_is_continuous_and_normalized():
    cfg = Config()
    model = CanonicalizedLatentAlgebra(cfg, seed=0)
    code = model.codebook()
    raw = model.raw_op(code[:5], code[5:10])
    out, weight = model.canonicalize(raw, return_weights=True)
    assert out.shape == (5, cfg.d_model)
    assert weight.shape == (5, cfg.modulus)
    assert torch.allclose(out.norm(dim=-1), torch.ones(5), atol=1e-5)
    assert torch.allclose(weight.sum(dim=-1), torch.ones(5), atol=1e-6)
    assert torch.all(weight > 0)
