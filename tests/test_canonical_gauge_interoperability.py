import torch
from canonical_gauge_interoperability_experiment import bridge_state


def test_bridge_composes_known_gauges():
    d=4; q,_=torch.linalg.qr(torch.randn(d,d,dtype=torch.float64)); r,_=torch.linalg.qr(torch.randn(d,d,dtype=torch.float64))
    source={'V':q.to(torch.complex128),'Vi':q.T.to(torch.complex128)}; target={'V':r.to(torch.complex128),'Vi':r.T.to(torch.complex128)}
    z=torch.randn(7,d,dtype=torch.float64); out,imag=bridge_state(z,source,target); expected=torch.nn.functional.normalize((r@q.T@z.T).T,dim=-1)
    assert imag<1e-12
    assert torch.allclose(out,expected,atol=1e-10,rtol=1e-10)
