import torch
from normed_operator_parameterization_experiment import Config, StructuredNormedAlgebra


def test_structured_operator_preserves_plane_manifold_and_commutes():
    cfg = Config()
    model = StructuredNormedAlgebra(cfg, seed=0)
    code = model.codebook()
    a, b = code[:8], code[8:16]
    ab = model.op(a, b)
    ba = model.op(b, a)
    assert torch.allclose(ab.norm(dim=-1), torch.ones(8), atol=1e-5)
    assert torch.allclose(ab, ba, atol=1e-5)
