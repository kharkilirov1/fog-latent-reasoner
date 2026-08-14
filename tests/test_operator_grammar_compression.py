import torch
from operator_grammar_compression_experiment import search_power


def test_search_power_recovers_known_exponent():
    theta=torch.tensor(0.37,dtype=torch.float64)
    c=torch.cos(theta); s=torch.sin(theta)
    base=torch.stack([torch.stack([c,-s]),torch.stack([s,c])])
    target=torch.linalg.matrix_power(base,5)
    best,_=search_power(base,target,11)
    assert best['exponent']==5
    assert best['residual']<1e-12
