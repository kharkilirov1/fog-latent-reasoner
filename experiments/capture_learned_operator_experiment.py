"""EXP-040: does attractor capture rescue a *learned* flexible one-step operator?
Affine grammar over F_31 (A:x->x+1, M:x->3x). Codes: random unit vectors in R^d.
Flexible MLP F(h, op) fitted on all 62 one-step pairs (100% one-step), as in the
flexible arm of EXP-014. Recurrence over random programs, depth up to 256:
  raw   : h <- F(h, op)                    (Pi = id)
  cap_b : h <- Pi_beta(F(h, op))           (Hopfield capture, NOTE-016)
  snap  : h <- m_argmax                    (discrete processor reference)
Also reports closure defect eps and the local gain L (max Jacobian spectral norm at codes)."""
import json, sys, torch, torch.nn as nn
torch.set_num_threads(1)
P=31
def run(seed,d,noise):
    g=torch.Generator().manual_seed(seed)
    C=torch.nn.functional.normalize(torch.randn(P,d,generator=g),dim=-1)
    nxt=torch.stack([(torch.arange(P)+1)%P,(3*torch.arange(P))%P])
    torch.manual_seed(seed)
    net=nn.Sequential(nn.Linear(d+2,256),nn.GELU(),nn.Linear(256,256),nn.GELU(),nn.Linear(256,d))
    F=lambda h,op: net(torch.cat([h,torch.nn.functional.one_hot(op,2).float()],-1))
    opt=torch.optim.Adam(net.parameters(),1e-3)
    X=C.repeat(2,1); OP=torch.arange(2).repeat_interleave(P); Y=C[nxt.reshape(-1)]
    for it in range(3000):
        h=X+noise*torch.randn_like(X)
        loss=((F(h,OP)-Y)**2).sum(-1).mean(); opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        out=F(X,OP); eps=(out-Y).norm(dim=-1).max().item()
        one=((out@C.T).argmax(-1)==nxt.reshape(-1)).float().mean().item()
    Ls=[]
    for k in range(2*P):
        J=torch.autograd.functional.jacobian(lambda h:F(h[None],OP[k:k+1])[0],X[k])
        Ls.append(torch.linalg.matrix_norm(J,2).item())
    G=C@C.T; Delta=1-(G-2*torch.eye(P)).max().item()
    res={"seed":seed,"d":d,"noise":noise,"one_step":one,"eps":eps,"L_max":max(Ls),"L_mean":sum(Ls)/len(Ls),"Delta":Delta,"eps_bound":Delta/2}
    gen=torch.Generator().manual_seed(1000+seed); B=512
    ops=torch.randint(0,2,(256,B),generator=gen); i0=torch.randint(0,P,(B,),generator=gen)
    for mode in ["raw","cap4","cap16","cap64","snap"]:
        h=C[i0]; i=i0.clone(); alive=torch.ones(B,dtype=torch.bool); acc={}
        with torch.no_grad():
            for t in range(256):
                h=F(h,ops[t]); i=nxt[ops[t],i]
                if mode.startswith("cap"):
                    h=torch.softmax(float(mode[3:])*h@C.T,-1)@C
                elif mode=="snap":
                    h=C[(h@C.T).argmax(-1)]
                alive&=((h@C.T).argmax(-1)==i)
                if t+1 in (1,8,16,32,64,128,256): acc[t+1]=alive.float().mean().item()
        res[mode]=acc
    return res
out=[]
for d in (32,):
    for noise in (0.0,0.05):
        for seed in (0,1,2):
            r=run(seed,d,noise); out.append(r)
            print(json.dumps({k:(round(v,3) if isinstance(v,float) else v) for k,v in r.items() if k not in("raw","cap4","cap16","cap64","snap")}),flush=True)
            for m in ("raw","cap4","cap16","cap64","snap"): print("  ",m,{k:round(v,3) for k,v in r[m].items()},flush=True)
import pathlib; pathlib.Path("artifacts/capture-learned-operator").mkdir(parents=True,exist_ok=True)
json.dump(out,open("artifacts/capture-learned-operator/results.json","w"),indent=1)
