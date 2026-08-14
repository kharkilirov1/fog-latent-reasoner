import torch
from single_chart_affine_machine_experiment import Config, codebook, add_op, mul_op, decode


def test_single_chart_exact_add_and_mul_on_small_batch():
    cfg=Config(); code=codebook(cfg); a=torch.tensor([2,5,7,9]); b=torch.tensor([3,4,0,11])
    assert torch.equal(decode(add_op(code[a],code[b],cfg),code),(a+b)%31)
    assert torch.equal(decode(mul_op(code[a],code[b],cfg,code),code),(a*b)%31)
