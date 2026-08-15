import json, random, re
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, Subset
from transformers import AutoTokenizer, AutoModelForCausalLM

from fog_retrofit_smollm135m import FOGSeq, MLP, GRU, Classifier, count_params

SEED=20260817
MODEL_ID="HuggingFaceTB/SmolLM2-135M-Instruct"
P=31
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.set_num_threads(4)


def make_ex(rng, depth):
    cur=rng.randrange(P); start=cur; ops=[]
    for _ in range(depth):
        kind=rng.choice(("add","sub","mul"))
        if kind=="add": k=rng.randint(1,8); cur=(cur+k)%P; code=k-1
        elif kind=="sub": k=rng.randint(1,8); cur=(cur-k)%P; code=8+k-1
        else: k=rng.randint(2,5); cur=(cur*k)%P; code=16+(k-2)
        ops.append((kind,k,code))
    return {"start":start,"ops":ops,"answer":cur,"depth":depth}


def build(tok,e):
    seg=[f"Work modulo {P}. Start with {e['start']}. "]
    for i,(kind,k,_) in enumerate(e['ops']):
        text=f"Add {k}." if kind=="add" else (f"Subtract {k}." if kind=="sub" else f"Multiply by {k}.")
        seg.append(f"Step {i+1}: {text} ")
    seg.append("What is the final value? Answer with only the integer.\nAnswer:")
    ids=[];marks=[]
    for j,s in enumerate(seg):
        ids.extend(tok.encode(s,add_special_tokens=False))
        if j<=len(e['ops']): marks.append(len(ids)-1)
    return ids,marks


@torch.no_grad()
def features(base,tok,examples,h,batch=12):
    mxslots=max(e['depth'] for e in examples)+1; pad=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    S=[];M=[];L=[];Y=[];D=[];START=[];OPS=[]
    for st in range(0,len(examples),batch):
        bb=examples[st:st+batch]; built=[build(tok,e) for e in bb]; mx=max(len(x[0]) for x in built)
        ids=torch.full((len(bb),mx),pad,dtype=torch.long);am=torch.zeros_like(ids)
        for i,(a,_) in enumerate(built):ids[i,:len(a)]=torch.tensor(a);am[i,:len(a)]=1
        hs=base(input_ids=ids,attention_mask=am,use_cache=False).last_hidden_state.float()
        for i,((a,marks),e) in enumerate(zip(built,bb)):
            s=torch.zeros(mxslots,h);m=torch.zeros(mxslots,dtype=torch.bool);op=torch.full((mxslots-1,),-1,dtype=torch.long)
            for j,pos in enumerate(marks):s[j]=hs[i,pos];m[j]=1
            for j,(_,_,code) in enumerate(e['ops']):op[j]=code
            S.append(s);M.append(m);L.append(hs[i,len(a)-1]);Y.append(e['answer']);D.append(e['depth']);START.append(e['start']);OPS.append(op)
        if st%240==0:print(f"features {st+len(bb)}/{len(examples)}",flush=True)
    return TensorDataset(torch.stack(S),torch.stack(M),torch.stack(L),torch.tensor(Y),torch.tensor(D),torch.tensor(START),torch.stack(OPS))


class Probe(nn.Module):
    def __init__(self,h,n):super().__init__();self.f=nn.Linear(h,n)
    def forward(self,x):return self.f(x)


def train_probe(model,x,y,epochs=12,lr=3e-3):
    opt=torch.optim.AdamW(model.parameters(),lr=lr)
    ds=TensorDataset(x,y)
    for _ in range(epochs):
        for a,b in DataLoader(ds,batch_size=128,shuffle=True):
            loss=F.cross_entropy(model(a),b);opt.zero_grad();loss.backward();opt.step()
    return model


@torch.no_grad()
def acc_probe(model,x,y):return float((model(x).argmax(-1)==y).float().mean())


def op_probe_tensors(ds):
    xs=[];ys=[]
    for s,m,l,y,d,start,ops in DataLoader(ds,batch_size=128):
        for t in range(ops.size(1)):
            q=ops[:,t]>=0
            if q.any():xs.append(s[q,t+1]);ys.append(ops[q,t])
    return torch.cat(xs),torch.cat(ys)


def depth_subset(ds,max_depth):
    idx=[]
    for i in range(len(ds)):
        if int(ds[i][4])<=max_depth:idx.append(i)
    return Subset(ds,idx)


def train_stage(model,ds,epochs,lr=2e-3):
    model.train();opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-4)
    for ep in range(epochs):
        total=n=0
        for s,m,l,y,d,start,ops in DataLoader(ds,batch_size=64,shuffle=True):
            loss=F.cross_entropy(model(s,m,l),y);opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step();total+=loss.item()*len(y);n+=len(y)
        if ep in {0,epochs-1}:print("stage",len(ds),ep+1,total/n,flush=True)
    return model


@torch.no_grad()
def evaluate(model,ds,force=None,inter=None):
    model.eval();out=defaultdict(lambda:[0,0])
    for s,m,l,y,d,start,ops in DataLoader(ds,batch_size=128):
        kw={"force":force,"intervention":inter} if isinstance(model.adapter,FOGSeq) else {}
        p=model(s,m,l,**kw).argmax(-1)
        for dep in d.unique():
            q=d==dep;out[int(dep)][0]+=int((p[q]==y[q]).sum());out[int(dep)][1]+=int(q.sum())
    return {str(k):v[0]/v[1] for k,v in sorted(out.items())}


def run_curriculum(name,model,train_ds):
    torch.manual_seed(SEED+sum(map(ord,name)))
    train_stage(model,depth_subset(train_ds,1),10,3e-3)
    train_stage(model,depth_subset(train_ds,2),8,2e-3)
    train_stage(model,depth_subset(train_ds,4),10,1e-3)
    return model


def main():
    tok=AutoTokenizer.from_pretrained(MODEL_ID);tok.pad_token=tok.pad_token or tok.eos_token
    lm=AutoModelForCausalLM.from_pretrained(MODEL_ID);lm.eval();[p.requires_grad_(False) for p in lm.parameters()];base=lm.model;h=lm.config.hidden_size
    print("backbone",MODEL_ID,"hidden",h,"params",sum(p.numel() for p in lm.parameters()),flush=True)
    r=random.Random(11);train=sum(([make_ex(r,d) for _ in range(400)] for d in [1,2,3,4]),[])
    q=random.Random(22);test=sum(([make_ex(q,d) for _ in range(100)] for d in [1,2,3,4,5,6,8,10,12]),[])
    tr=features(base,tok,train,h);te=features(base,tok,test,h)

    # Bridge diagnostics: can frozen LLM states expose the exact machine symbols?
    start_probe=Probe(h,P);start_probe=train_probe(start_probe,tr.tensors[0][:,0],tr.tensors[5])
    start_train=acc_probe(start_probe,tr.tensors[0][:,0],tr.tensors[5]);start_test=acc_probe(start_probe,te.tensors[0][:,0],te.tensors[5])
    ox,oy=op_probe_tensors(tr);tx,ty=op_probe_tensors(te);op_probe=Probe(h,20);op_probe=train_probe(op_probe,ox,oy)
    op_train=acc_probe(op_probe,ox,oy);op_test=acc_probe(op_probe,tx,ty)
    probes={"start_train":start_train,"start_test":start_test,"op_train":op_train,"op_test":op_test}
    print("PROBES",probes,flush=True)

    models={"linear":Classifier(h,nn.Identity(),p=P),"mlp":Classifier(h,MLP(h),p=P),"gru":Classifier(h,GRU(h),p=P),"fog":Classifier(h,FOGSeq(h),p=P)}
    # Identity has wrong signature; replace linear with a tiny zero adapter.
    class Z(nn.Module):
        def forward(self,s,m,last):return torch.zeros_like(last)
    models["linear"]=Classifier(h,Z(),p=P)
    results={};params={}
    for name,model in models.items():
        run_curriculum(name,model,tr);results[name]=evaluate(model,te);params[name]=count_params(model);print("RESULT",name,results[name],flush=True)
    results["fog_force1"]=evaluate(models["fog"],te,force=1);results["fog_shuffle2"]=evaluate(models["fog"],te,inter="shuffle2");results["fog_zero2"]=evaluate(models["fog"],te,inter="zero2")
    out={"model":MODEL_ID,"frozen_backbone":True,"modulus":P,"n_train":len(tr),"n_test":len(te),"train_depths":[1,2,3,4],"test_depths":[1,2,3,4,5,6,8,10,12],"bridge_probes":probes,"trainable_params":params,"results":results}
    Path("retrofit_v2_results.json").write_text(json.dumps(out,indent=2));torch.save({"fog":models["fog"].state_dict(),"result":out},"retrofit_v2_fog.pt");print("FINAL_V2",json.dumps(out,sort_keys=True),flush=True)

if __name__=="__main__":main()
