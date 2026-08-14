import torch


def test_orthogonal_completion_degrees_of_freedom():
    # Fixing an orthogonal map on a k-dimensional subspace leaves an arbitrary
    # O(d-k) action on its orthogonal complement.  Demonstrate the d-1 sign
    # ambiguity explicitly.
    d=5
    Q=torch.eye(d)
    T1=Q.clone(); T2=Q.clone(); T2[-1,-1]=-1
    x=torch.eye(d)[:d-1]
    assert torch.allclose(x@T1.T,x@T2.T)
    assert torch.linalg.det(T1).item()==1
    assert torch.linalg.det(T2).item()==-1
