import torch
from degenerate_joint_anchor_synthesis_experiment import distinct_root_count


def test_distinct_root_count_detects_degeneracy():
    # diag(1,-1,1,-1) has only two distinct 2nd-root eigenvalues.
    W=torch.diag(torch.tensor([1.,-1.,1.,-1.]))
    assert distinct_root_count(W,2)==2
