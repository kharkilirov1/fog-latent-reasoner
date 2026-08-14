import torch
from learned_affine_representation_experiment import Config, LearnedAffineRep


def test_affine_rep_shapes():
    m=LearnedAffineRep(8,0,Config()); c=m.codebook(); assert c.shape==(31,8)
    assert m.act(c,'A').shape==c.shape and m.act(c,'M').shape==c.shape
    assert torch.allclose(c.norm(dim=-1),torch.ones(31),atol=1e-5)
