from joint_chart_operator_experiment import JointConfig, JointLatentAlgebra


def test_joint_operator_shapes_and_normalization():
    import torch
    cfg = JointConfig()
    model = JointLatentAlgebra(cfg, seed=0)
    code = model.codebook()
    out = model.op(code[:5], code[5:10])
    assert out.shape == (5, cfg.d_model)
    assert torch.allclose(out.norm(dim=-1), torch.ones(5), atol=1e-5)
