import torch
from automatic_operator_law_discovery_experiment import discover_order, discover_conjugacy


def test_discovers_finite_rotation_order():
    n=7; theta=2*torch.pi*torch.tensor(1.0)/n
    A=torch.tensor([[torch.cos(theta),-torch.sin(theta)],[torch.sin(theta),torch.cos(theta)]],dtype=torch.float64)
    best,_=discover_order(A,14)
    assert best['order']==7
    assert best['residual']<1e-6


def test_discovers_conjugacy_power_on_diagonal_complex_representation():
    # Real 4D is unnecessary here: discovery only needs matrix algebra.
    A=torch.diag(torch.tensor([1.0,2.0],dtype=torch.float64))
    # Identity conjugacy is A^1.
    M=torch.eye(2,dtype=torch.float64)
    best,_=discover_conjugacy(A,M,5)
    assert best['exponent']==1
    assert best['residual']<1e-12
