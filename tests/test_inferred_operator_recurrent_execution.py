from inferred_operator_recurrent_execution_experiment import sample_nonambiguous_demos
import torch


def test_demo_sampler_rejects_semantic_ambiguity():
    g=torch.Generator().manual_seed(0)
    for name in ('add','mul'):
        demos=sample_nonambiguous_demos(name,2,g)
        assert not all(((a+b)%31==c) and ((a*b)%31==c) for a,b,c in demos)
