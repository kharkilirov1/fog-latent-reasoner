import math
import torch
from motif_projection_denoising_experiment import closure_project


def test_closure_projection_preserves_exact_finite_rotation():
    p=31; theta=2*math.pi/p
    A=torch.tensor([[math.cos(theta),-math.sin(theta)],[math.sin(theta),math.cos(theta)]],dtype=torch.float32)
    M=torch.eye(2)
    Ap,Mp=closure_project(A,M)
    assert torch.allclose(Ap,A,atol=1e-5,rtol=1e-5)
    assert torch.allclose(Mp,M,atol=1e-5,rtol=1e-5)
