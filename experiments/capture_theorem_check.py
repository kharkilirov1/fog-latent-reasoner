"""Numerical check of NOTE-016 (Theorem A' / Prop E).
Affine grammar over F_31 (A: x->x+1, M: x->3x). Transition F_c(h) = m_{P_c(i(h))}
+ L*(h - m_i) + eps*u  (expansive local gain L, closure defect eps, nonlinear because
it is defined piecewise by nearest code). Loop with/without Hopfield capture."""
import numpy as np
P=31; rng=np.random.default_rng(0)
def codes(d):
    if d==2:
        th=2*np.pi*np.arange(P)/P; return np.stack([np.cos(th),np.sin(th)],1)
    Q,_=np.linalg.qr(rng.normal(size=(d,d))); return Q[:P]
def run(C,L,eps,beta,depth=256,trials=200):
    R=np.linalg.norm(C[0]); G=C@C.T; Delta=R*R-(G-np.eye(P)*1e9).max()
    ok=0
    for _ in range(trials):
        i=rng.integers(P); h=C[i].copy(); good=True
        for t in range(depth):
            op=rng.integers(2); j=(i+1)%P if op==0 else (3*i)%P
            u=rng.normal(size=C.shape[1]); u/=np.linalg.norm(u)
            k=np.argmin(np.linalg.norm(C-h,axis=1))         # local chart of F
            h=C[(k+1)%P if op==0 else (3*k)%P]+L*(h-C[k])+eps*u
            if beta is not None:
                z=beta*(C@h); z-=z.max(); p=np.exp(z); p/=p.sum(); h=p@C
            i=j
            if np.argmin(np.linalg.norm(C-h,axis=1))!=i: good=False;break
        ok+=good
    return ok/trials, Delta
for d in (2,32):
    C=codes(d); R=1.0
    for L in (0.9,1.5,4.0):
        _,Delta=run(C,L,0.0,None,depth=1,trials=1)
        eps=0.3*Delta/(2*R)
        s=np.linspace(eps,Delta/(2*R),2002)[1:-1]
        bstar=np.min(np.log(2*R*max(L,1)*(P-1)/(s-eps))/(Delta-2*R*s))
        row=[f"d={d} L={L} eps={eps:.4f} Delta={Delta:.4f} beta*={bstar:.0f}"]
        row.append(f"no-capture={run(C,L,eps,None)[0]:.2f}")
        for b in (bstar/8,bstar/2,bstar):
            row.append(f"beta={b:.0f}:{run(C,L,eps,b)[0]:.2f}")
        print(" | ".join(row),flush=True)
