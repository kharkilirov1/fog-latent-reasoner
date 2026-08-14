import torch


def test_simplex_spanning_boundary():
    # d+1 centered simplex vertices in R^d: any d vertices span R^d, while
    # any d-1 vertices cannot have rank d.
    d=6
    eye=torch.eye(d+1,dtype=torch.float64); centered=eye-eye.mean(0,keepdim=True)
    q,_=torch.linalg.qr(centered.T[:,:d],mode='reduced')
    codes=centered@q
    assert torch.linalg.matrix_rank(codes[:d]).item()==d
    assert torch.linalg.matrix_rank(codes[:d-1]).item()==d-1
