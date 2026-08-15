import os, re, math, json, random
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

SEED=20260816
MODEL_ID="HuggingFaceTB/SmolLM2-135M-Instruct"
DEVICE="cpu"
torch.set_num_threads(max(1,min(4,os.cpu_count() or 1)))
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

def prime_label_map(tok):
    for p in [97,89,83,79,73,71,67,61,59,53,47,43,41,37,31]:
        ids=[]
        for i in range(p):
            cand=tok.encode(str(i),add_special_tokens=False)
            if len(cand)!=1: cand=tok.encode(" "+str(i),add_special_tokens=False)
            if len(cand)!=1: break
            ids.append(cand[0])
        if len(ids)==p: return p,ids
    raise RuntimeError("no contiguous prime single-token numeric vocabulary")

def mk(rng,d,p):
    x=rng.randrange(p); start=x; ops=[]
    for _ in range(d):
        kind=rng.choice(('add','sub','mul'))
        if kind=='add': k=rng.randint(1,9); x=(x+k)%p; ops.append(f"Add {k}.")
        elif kind=='sub': k=rng.randint(1,9); x=(x-k)%p; ops.append(f"Subtract {k}.")
        else: k=rng.randint(2,4); x=(x*k)%p; ops.append(f"Multiply by {k}.")
    return {'start':start,'ops':ops,'answer':x,'depth':d}

def segment_tokens(tok,e,p):
    seg=[f"Work modulo {p}. Start with {e['start']}. "]+[f"Step {i+1}: {o} " for i,o in enumerate(e['ops'])]+["Final value? Answer with only the integer.\nAnswer:"]
    ids=[]; marks=[]
    for j,s in enumerate(seg):
        ids.extend(tok.encode(s,add_special_tokens=False))
        if j<=len(e['ops']): marks.append(len(ids)-1)
    return ids,marks

@torch.no_grad()
def chain_features(base,tok,examples,p,h,batch=12):
    slots_n=max(e['depth'] for e in examples)+1; pad=tok.pad_token_id or tok.eos_token_id
    S=[];M=[];L=[];Y=[];D=[]
    for st in range(0,len(examples),batch):
        bb=examples[st:st+batch]; built=[segment_tokens(tok,e,p) for e in bb]; mx=max(len(x[0]) for x in built)
        ids=torch.full((len(bb),mx),pad,dtype=torch.long); am=torch.zeros_like(ids)
        for i,(a,_) in enumerate(built): ids[i,:len(a)]=torch.tensor(a); am[i,:len(a)]=1
        hs=base(input_ids=ids,attention_mask=am,use_cache=False).last_hidden_state
        for i,((a,marks),e) in enumerate(zip(built,bb)):
            s=torch.zeros(slots_n,h); m=torch.zeros(slots_n,dtype=torch.bool)
            for j,pos in enumerate(marks): s[j]=hs[i,pos];m[j]=1
            S.append(s);M.append(m);L.append(hs[i,len(a)-1]);Y.append(e['answer']);D.append(e['depth'])
        if st%240==0: print(f"features chain {st+len(bb)}/{len(examples)}",flush=True)
    return TensorDataset(torch.stack(S),torch.stack(M),torch.stack(L),torch.tensor(Y),torch.tensor(D))

def gsm_ans(s):
    m=re.search(r"####\s*([-+]?\d[\d,]*)",s)
    return int(m.group(1).replace(',','')) if m else None

@torch.no_grad()
def context_features(base,tok,rows,p,h,kctx=32,batch=8):
    ps=[];ys=[]
    for r in rows:
        y=gsm_ans(r['answer'])
        if y is not None and 0<=y<p:
            ps.append("Solve the problem. Answer with only the integer.\n"+r['question']+"\nAnswer:");ys.append(y)
    C=[];M=[];L=[];Y=[]
    for st in range(0,len(ps),batch):
        enc=tok(ps[st:st+batch],padding=True,truncation=True,max_length=160,return_tensors='pt',add_special_tokens=False)
        hs=base(**enc,use_cache=False).last_hidden_state
        for i in range(len(enc['input_ids'])):
            n=int(enc['attention_mask'][i].sum()); take=min(kctx,n); c=torch.zeros(kctx,h);m=torch.zeros(kctx,dtype=torch.bool)
            c[-take:]=hs[i,n-take:n];m[-take:]=1;C.append(c);M.append(m);L.append(hs[i,n-1]);Y.append(ys[st+i])
        if st%80==0: print(f"features gsm {st+len(enc['input_ids'])}/{len(ps)}",flush=True)
    return TensorDataset(torch.stack(C),torch.stack(M),torch.stack(L),torch.tensor(Y))

class MLP(nn.Module):
    def __init__(self,h,b=128):
        super().__init__();self.net=nn.Sequential(nn.Linear(2*h,b),nn.GELU(),nn.Linear(b,h));nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
    def forward(self,s,m,last):
        mf=m.float().unsqueeze(-1);mean=(s*mf).sum(1)/mf.sum(1).clamp_min(1);return self.net(torch.cat([last,mean],-1))
class GRU(nn.Module):
    def __init__(self,h,d=96):
        super().__init__();self.i=nn.Linear(h,d);self.g=nn.GRU(d,d,batch_first=True);self.o=nn.Linear(d,h);nn.init.zeros_(self.o.weight);nn.init.zeros_(self.o.bias)
    def forward(self,s,m,last):
        z,_=self.g(self.i(s));idx=m.sum(1).clamp_min(1)-1;return self.o(z[torch.arange(z.size(0)),idx])
class BOp(nn.Module):
    def __init__(self,d,r=48):
        super().__init__();self.a=nn.Linear(d,r,bias=False);self.b=nn.Linear(d,r,bias=False);self.o=nn.Linear(r,d,bias=False);self.n=nn.LayerNorm(d)
    def forward(self,z,c):return self.n(z+self.o(self.a(z)*self.b(c)))
def bprod(z,c):
    q=z.reshape(z.size(0),-1,2);w=c.reshape(c.size(0),-1,2);q=F.normalize(q,dim=-1);w=F.normalize(w,dim=-1)
    return torch.stack((q[...,0]*w[...,0]-q[...,1]*w[...,1],q[...,0]*w[...,1]+q[...,1]*w[...,0]),-1).reshape_as(z)
class FOGSeq(nn.Module):
    def __init__(self,h,d=96):
        super().__init__();self.a=nn.Linear(h,d);self.c=nn.Linear(h,d);self.ops=nn.ModuleList([BOp(d) for _ in range(3)]);self.r=nn.Linear(2*d,6);self.n=nn.LayerNorm(d);self.o=nn.Linear(d,h);nn.init.zeros_(self.o.weight);nn.init.zeros_(self.o.bias);self.routes=None
    def one(self,z,c):
        cand=[z,c,self.n(bprod(z,c))]+[op(z,c) for op in self.ops];p=F.softmax(self.r(torch.cat([z,c],-1)),-1);hard=F.one_hot(p.argmax(-1),6).float();w=hard+p-p.detach();return self.n(sum(w[:,i:i+1]*cand[i] for i in range(6))),p
    def forward(self,s,m,last,force=None,intervention=None):
        z=self.n(self.a(s[:,0]));routes=[];mx=s.size(1)-1 if force is None else min(force,s.size(1)-1)
        for t in range(mx):
            active=m[:,t+1]
            if not active.any():break
            zn,p=self.one(z,self.n(self.c(s[:,t+1])));z=torch.where(active[:,None],zn,z);routes.append(p.detach())
            if intervention=='shuffle2' and t==1:z=z[torch.randperm(z.size(0))]
            if intervention=='zero2' and t==1:z=torch.zeros_like(z)
        self.routes=routes;return self.o(z)
class FOGCtx(nn.Module):
    def __init__(self,h,d=96):
        super().__init__();self.i=nn.Linear(h,d);self.k=nn.Linear(h,d,bias=False);self.v=nn.Linear(h,d,bias=False);self.q=nn.Linear(d,d,bias=False);self.ops=nn.ModuleList([BOp(d) for _ in range(3)]);self.r=nn.Linear(2*d,6);self.n=nn.LayerNorm(d);self.o=nn.Linear(d,h);nn.init.zeros_(self.o.weight);nn.init.zeros_(self.o.bias)
    def one(self,z,ctx,m):
        sc=torch.einsum('bd,bkd->bk',F.normalize(self.q(z),dim=-1),F.normalize(self.k(ctx),dim=-1))*8;sc=sc.masked_fill(~m,-1e9);a=F.softmax(sc,-1);c=self.n(torch.einsum('bk,bkd->bd',a,self.v(ctx)));cand=[z,c,self.n(bprod(z,c))]+[op(z,c) for op in self.ops];p=F.softmax(self.r(torch.cat([z,c],-1)),-1);hard=F.one_hot(p.argmax(-1),6).float();w=hard+p-p.detach();return self.n(sum(w[:,i:i+1]*cand[i] for i in range(6))),p
    def forward(self,ctx,m,last,steps=4,intervention=None):
        z=self.n(self.i(last))
        for t in range(steps):
            z,_=self.one(z,ctx,m)
            if intervention=='shuffle2' and t==1:z=z[torch.randperm(z.size(0))]
            if intervention=='zero2' and t==1:z=torch.zeros_like(z)
        return self.o(z)

def logits(last,delta,W):return (last+delta)@W.T

def train(name,model,ds,W,epochs=6,ctx=False):
    model.train();opt=torch.optim.AdamW(model.parameters(),lr=2e-3,weight_decay=1e-4);dl=DataLoader(ds,batch_size=64,shuffle=True)
    for ep in range(epochs):
        total=0;n=0
        for b in dl:
            if ctx:c,m,l,y=b;d=model(c,m,l,steps=4)
            else:s,m,l,y,dep=b;d=model(s,m,l)
            loss=F.cross_entropy(logits(l,d,W),y);opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step();total+=loss.item()*len(y);n+=len(y)
        print(name,ep+1,total/n,flush=True)
    return model
@torch.no_grad()
def eval_chain(model,ds,W,baseline=False,force=None,inter=None):
    a=defaultdict(lambda:[0,0])
    for s,m,l,y,d in DataLoader(ds,batch_size=128):
        dd=torch.zeros_like(l) if baseline else (model(s,m,l,force=force,intervention=inter) if isinstance(model,FOGSeq) else model(s,m,l));pr=logits(l,dd,W).argmax(-1)
        for x in d.unique():q=d==x;a[int(x)][0]+=int((pr[q]==y[q]).sum());a[int(x)][1]+=int(q.sum())
    return {str(k):v[0]/v[1] for k,v in sorted(a.items())}
@torch.no_grad()
def eval_ctx(model,ds,W,baseline=False,steps=4,inter=None):
    c=n=0
    for x,m,l,y in DataLoader(ds,batch_size=64):
        d=torch.zeros_like(l) if baseline else (model(x,m,l,steps=steps,intervention=inter) if isinstance(model,FOGCtx) else model(x,m,l));pr=logits(l,d,W).argmax(-1);c+=int((pr==y).sum());n+=len(y)
    return c/n

def main():
    print('load',MODEL_ID,flush=True);tok=AutoTokenizer.from_pretrained(MODEL_ID);tok.pad_token=tok.pad_token or tok.eos_token;p,ids=prime_label_map(tok);print('p',p,flush=True)
    lm=AutoModelForCausalLM.from_pretrained(MODEL_ID);lm.eval();[x.requires_grad_(False) for x in lm.parameters()];base=lm.model;h=lm.config.hidden_size;W=lm.lm_head.weight[ids].detach();print('hidden',h,'params',sum(x.numel() for x in lm.parameters()),flush=True)
    r=random.Random(1);tr=sum(([mk(r,d,p) for _ in range(300)] for d in [1,2,3,4]),[]);q=random.Random(2);te=sum(([mk(q,d,p) for _ in range(100)] for d in [1,2,3,4,5,6,8,10,12]),[])
    trf=chain_features(base,tok,tr,p,h);tef=chain_features(base,tok,te,p,h)
    res={'baseline':eval_chain(None,tef,W,baseline=True)};mods={'mlp':MLP(h),'gru':GRU(h),'fog':FOGSeq(h)}
    for n,m in mods.items():train(n,m,trf,W,epochs=8 if n=='fog' else 6);res[n]=eval_chain(m,tef,W);print('chain',n,res[n],flush=True)
    res['fog_force1']=eval_chain(mods['fog'],tef,W,force=1);res['fog_shuffle2']=eval_chain(mods['fog'],tef,W,inter='shuffle2');res['fog_zero2']=eval_chain(mods['fog'],tef,W,inter='zero2')
    gsm=load_dataset('openai/gsm8k','main');trr=[x for x in gsm['train'] if (lambda y:y is not None and 0<=y<p)(gsm_ans(x['answer']))][:300];ter=[x for x in gsm['test'] if (lambda y:y is not None and 0<=y<p)(gsm_ans(x['answer']))][:100]
    trc=context_features(base,tok,trr,p,h);tec=context_features(base,tok,ter,p,h);gm=MLP(h);gf=FOGCtx(h);train('gsm_mlp',gm,trc,W,8,True);train('gsm_fog',gf,trc,W,10,True)
    gres={'baseline':eval_ctx(None,tec,W,True),'mlp':eval_ctx(gm,tec,W),'fog_r1':eval_ctx(gf,tec,W,steps=1),'fog_r2':eval_ctx(gf,tec,W,steps=2),'fog_r4':eval_ctx(gf,tec,W,steps=4),'fog_r8':eval_ctx(gf,tec,W,steps=8),'fog_shuffle2':eval_ctx(gf,tec,W,steps=4,inter='shuffle2'),'fog_zero2':eval_ctx(gf,tec,W,steps=4,inter='zero2')}
    out={'model':MODEL_ID,'modulus':p,'chain':res,'gsm8k_numeric_subset':gres,'n_chain_train':len(tr),'n_chain_test':len(te),'n_gsm_train':len(trc),'n_gsm_test':len(tec),'limitations':['frozen 135M backbone','numeric constrained decoding','GSM8K answers restricted to 0..p-1']};Path('retrofit_results.json').write_text(json.dumps(out,indent=2));torch.save({'fog_chain':mods['fog'].state_dict(),'fog_gsm':gf.state_dict(),'results':out},'retrofit_adapters.pt');print('FINAL',json.dumps(out,sort_keys=True),flush=True)
if __name__=='__main__':main()
