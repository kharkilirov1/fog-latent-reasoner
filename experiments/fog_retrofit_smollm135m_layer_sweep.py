import json, random
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

SEED=20260818
MODEL_ID="HuggingFaceTB/SmolLM2-135M-Instruct"
P=31
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.set_num_threads(4)

def mk(rng,depth):
    cur=rng.randrange(P); start=cur; ops=[]
    for _ in range(depth):
        kind=rng.choice(("add","sub","mul"))
        if kind=="add": k=rng.randint(1,8); cur=(cur+k)%P; code=k-1
        elif kind=="sub": k=rng.randint(1,8); cur=(cur-k)%P; code=8+k-1
        else: k=rng.randint(2,5); cur=(cur*k)%P; code=16+(k-2)
        ops.append((kind,k,code))
    return {"start":start,"ops":ops,"depth":depth}

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
def collect(base,tok,examples,batch=10):
    pad=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    starts=[];start_y=[];ops=[];op_y=[]
    nlayers=None
    for st in range(0,len(examples),batch):
        bb=examples[st:st+batch];built=[build(tok,e) for e in bb];mx=max(len(x[0]) for x in built)
        ids=torch.full((len(bb),mx),pad,dtype=torch.long);am=torch.zeros_like(ids)
        for i,(a,_) in enumerate(built):ids[i,:len(a)]=torch.tensor(a);am[i,:len(a)]=1
        out=base(input_ids=ids,attention_mask=am,use_cache=False,output_hidden_states=True,return_dict=True);hs=out.hidden_states
        if nlayers is None:nlayers=len(hs);print("hidden_states",nlayers,"width",hs[0].size(-1),flush=True)
        for i,((a,marks),e) in enumerate(zip(built,bb)):
            starts.append(torch.stack([x[i,marks[0]].float().cpu() for x in hs],0));start_y.append(e['start'])
            for t,(_,_,code) in enumerate(e['ops']):
                ops.append(torch.stack([x[i,marks[t+1]].float().cpu() for x in hs],0));op_y.append(code)
        if st%200==0:print("collect",st+len(bb),"/",len(examples),flush=True)
    return torch.stack(starts),torch.tensor(start_y),torch.stack(ops),torch.tensor(op_y)

def probe(train_x,train_y,val_x,val_y,test_x,test_y,ncls,epochs=12):
    h=train_x.size(-1);m=nn.Linear(h,ncls);opt=torch.optim.AdamW(m.parameters(),lr=4e-3,weight_decay=1e-4)
    for _ in range(epochs):
        perm=torch.randperm(len(train_y))
        for st in range(0,len(perm),128):
            ix=perm[st:st+128];loss=F.cross_entropy(m(train_x[ix]),train_y[ix]);opt.zero_grad();loss.backward();opt.step()
    with torch.no_grad():
        tr=float((m(train_x).argmax(-1)==train_y).float().mean());va=float((m(val_x).argmax(-1)==val_y).float().mean());te=float((m(test_x).argmax(-1)==test_y).float().mean())
    return tr,va,te

def main():
    tok=AutoTokenizer.from_pretrained(MODEL_ID);tok.pad_token=tok.pad_token or tok.eos_token
    lm=AutoModelForCausalLM.from_pretrained(MODEL_ID);lm.eval();[p.requires_grad_(False) for p in lm.parameters()];base=lm.model
    def gen(seed,n):
        r=random.Random(seed);return [mk(r,r.choice([1,2,3,4])) for _ in range(n)]
    tr=gen(100,700);va=gen(200,300);te=gen(300,400)
    ts,tsy,to,toy=collect(base,tok,tr);vs,vsy,vo,voy=collect(base,tok,va);es,esy,eo,eoy=collect(base,tok,te)
    L=ts.size(1);rows=[]
    for layer in range(L):
        st=probe(ts[:,layer],tsy,vs[:,layer],vsy,es[:,layer],esy,P,epochs=8)
        op=probe(to[:,layer],toy,vo[:,layer],voy,eo[:,layer],eoy,20,epochs=10)
        row={"layer":layer,"start_train":st[0],"start_val":st[1],"start_test":st[2],"op_train":op[0],"op_val":op[1],"op_test":op[2]};rows.append(row);print("LAYER",json.dumps(row),flush=True)
    best=max(rows,key=lambda x:x['op_val']);out={"model":MODEL_ID,"n_train":len(tr),"n_val":len(va),"n_test":len(te),"layers":rows,"best_by_op_val":best}
    Path('retrofit_layer_sweep.json').write_text(json.dumps(out,indent=2));print("FINAL_LAYER_SWEEP",json.dumps(out,sort_keys=True),flush=True)
if __name__=='__main__':main()
