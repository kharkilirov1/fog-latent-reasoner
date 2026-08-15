import json, random
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM

SEED=20260820
MODEL_ID="HuggingFaceTB/SmolLM2-135M-Instruct"
P=31
VALUE_LAYER=30   # selected by validation in prior sweep
OP_LAYER=18      # selected by validation in prior sweep
random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.set_num_threads(4)

def make_ex(rng,depth,style):
    cur=rng.randrange(P);start=cur;ops=[]
    for _ in range(depth):
        kind=rng.choice(("add","sub","mul"))
        if kind=="add":k=rng.randint(1,8);cur=(cur+k)%P;code=k-1
        elif kind=="sub":k=rng.randint(1,8);cur=(cur-k)%P;code=8+k-1
        else:k=rng.randint(2,5);cur=(cur*k)%P;code=16+(k-2)
        ops.append((kind,k,code))
    return {"start":start,"ops":ops,"answer":cur,"depth":depth,"style":style}

def text(kind,k,style):
    if style=="base":return f"Add {k}." if kind=="add" else (f"Subtract {k}." if kind=="sub" else f"Multiply by {k}.")
    if style=="para":return f"Increase the current value by {k}." if kind=="add" else (f"Decrease the current value by {k}." if kind=="sub" else f"Scale the current value by a factor of {k}.")
    return f"Now add {k} more." if kind=="add" else (f"Now take away {k}." if kind=="sub" else f"Now take {k} times the current value.")

def build(tok,e):
    seg=[f"Work modulo {P}. Start with {e['start']}. "]+[f"Step {i+1}: {text(kind,k,e['style'])} " for i,(kind,k,_) in enumerate(e['ops'])]+["What is the final value? Answer with only the integer.\nAnswer:"]
    ids=[];marks=[]
    for j,s in enumerate(seg):
        ids.extend(tok.encode(s,add_special_tokens=False))
        if j<=len(e['ops']):marks.append(len(ids)-1)
    return ids,marks

@torch.no_grad()
def extract(base,tok,examples,h,batch=10):
    maxops=max(e['depth'] for e in examples);pad=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    VF=[];OF=[];MASK=[];VY=[];OY=[];ANS=[];DEP=[]
    for st in range(0,len(examples),batch):
        bb=examples[st:st+batch];built=[build(tok,e) for e in bb];mx=max(len(x[0]) for x in built)
        ids=torch.full((len(bb),mx),pad,dtype=torch.long);am=torch.zeros_like(ids)
        for i,(a,_) in enumerate(built):ids[i,:len(a)]=torch.tensor(a);am[i,:len(a)]=1
        hs=base(input_ids=ids,attention_mask=am,use_cache=False,output_hidden_states=True,return_dict=True).hidden_states
        for i,((a,marks),e) in enumerate(zip(built,bb)):
            of=torch.zeros(maxops,h);mask=torch.zeros(maxops,dtype=torch.bool);oy=torch.full((maxops,),-1,dtype=torch.long)
            for t,(_,_,code) in enumerate(e['ops']):of[t]=hs[OP_LAYER][i,marks[t+1]].float();mask[t]=1;oy[t]=code
            VF.append(hs[VALUE_LAYER][i,marks[0]].float());OF.append(of);MASK.append(mask);VY.append(e['start']);OY.append(oy);ANS.append(e['answer']);DEP.append(e['depth'])
        if st%200==0:print("features",st+len(bb),"/",len(examples),flush=True)
    return TensorDataset(torch.stack(VF),torch.stack(OF),torch.stack(MASK),torch.tensor(VY),torch.stack(OY),torch.tensor(ANS),torch.tensor(DEP))

class LinearParser(nn.Module):
    def __init__(self,h,n):super().__init__();self.f=nn.Linear(h,n)
    def forward(self,x):return self.f(x)

def train_parser(m,x,y,epochs=10,lr=4e-3):
    opt=torch.optim.AdamW(m.parameters(),lr=lr,weight_decay=1e-4);ds=TensorDataset(x,y)
    for _ in range(epochs):
        for a,b in DataLoader(ds,batch_size=128,shuffle=True):
            loss=F.cross_entropy(m(a),b);opt.zero_grad();loss.backward();opt.step()
    return m

def flat_ops(ds):
    X=[];Y=[]
    for vf,of,mask,vy,oy,ans,dep in DataLoader(ds,batch_size=128):
        q=mask.reshape(-1);X.append(of.reshape(-1,of.size(-1))[q]);Y.append(oy.reshape(-1)[q])
    return torch.cat(X),torch.cat(Y)

def apply(x,code):
    if code<8:return (x+code+1)%P
    if code<16:return (x-(code-8+1))%P
    return (x*(code-16+2))%P

@torch.no_grad()
def evaluate(vp,op,ds,mode="pred",shuffle_ops=False):
    by=defaultdict(lambda:[0,0]);vc=vn=oc=on=0
    for vf,of,mask,vy,oy,ans,dep in DataLoader(ds,batch_size=128):
        pv=vp(vf).argmax(-1);po=op(of.reshape(-1,of.size(-1))).argmax(-1).reshape_as(oy)
        vc+=int((pv==vy).sum());vn+=len(vy);q=mask;oc+=int((po[q]==oy[q]).sum());on+=int(q.sum())
        if shuffle_ops:po=po[torch.randperm(len(po))]
        for i in range(len(vy)):
            cur=int(vy[i]) if mode in ("oracle","oracle_value_pred_ops") else int(pv[i])
            for t in range(int(dep[i])):
                code=int(oy[i,t]) if mode in ("oracle","pred_value_oracle_ops") else int(po[i,t])
                cur=apply(cur,code)
            by[int(dep[i])][0]+=int(cur==int(ans[i]));by[int(dep[i])][1]+=1
    return {str(k):v[0]/v[1] for k,v in sorted(by.items())},{"value_acc":vc/vn,"op_acc":oc/on}

def main():
    tok=AutoTokenizer.from_pretrained(MODEL_ID);tok.pad_token=tok.pad_token or tok.eos_token
    lm=AutoModelForCausalLM.from_pretrained(MODEL_ID);lm.eval();[p.requires_grad_(False) for p in lm.parameters()];base=lm.model;h=lm.config.hidden_size
    print("backbone",MODEL_ID,"layers",len(base.layers) if hasattr(base,'layers') else '?',"value_layer",VALUE_LAYER,"op_layer",OP_LAYER,flush=True)
    def gen(seed,counts,style):
        r=random.Random(seed);return sum(([make_ex(r,d,style) for _ in range(n)] for d,n in counts),[])
    tr=extract(base,tok,gen(501,[(1,400),(2,400),(3,400),(4,400)],"base"),h)
    va=extract(base,tok,gen(502,[(1,100),(2,100),(3,100),(4,100)],"base"),h)
    te=extract(base,tok,gen(503,[(d,120) for d in [1,2,3,4,5,6,8,10,12]],"base"),h)
    pa=extract(base,tok,gen(504,[(d,120) for d in [1,2,3,4,5,6,8,10,12]],"para"),h)
    vp=LinearParser(h,P);op=LinearParser(h,20);vp=train_parser(vp,tr.tensors[0],tr.tensors[3],12);ox,oy=flat_ops(tr);op=train_parser(op,ox,oy,12)
    vox,voy=flat_ops(va);tox,toy=flat_ops(te);pox,poy=flat_ops(pa)
    def ac(m,x,y):return float((m(x).argmax(-1)==y).float().mean())
    probes={"value_train":ac(vp,tr.tensors[0],tr.tensors[3]),"value_val":ac(vp,va.tensors[0],va.tensors[3]),"value_test":ac(vp,te.tensors[0],te.tensors[3]),"op_train":ac(op,ox,oy),"op_val":ac(op,vox,voy),"op_test":ac(op,tox,toy),"op_paraphrase":ac(op,pox,poy)}
    results={}
    for n,m in [("oracle","oracle"),("predicted","pred"),("pred_value_oracle_ops","pred_value_oracle_ops"),("oracle_value_pred_ops","oracle_value_pred_ops")]:results[n]=evaluate(vp,op,te,m)[0]
    results["predicted_paraphrase"]=evaluate(vp,op,pa,"pred")[0];results["shuffle_ops"]=evaluate(vp,op,te,"pred",True)[0]
    print("PROBES",probes,flush=True);print("RESULTS",json.dumps(results,sort_keys=True),flush=True)
    out={"model":MODEL_ID,"frozen_backbone":True,"selection":{"value_layer":VALUE_LAYER,"op_layer":OP_LAYER,"criterion":"chosen from prior validation-only linear layer sweep"},"modulus":P,"train_depths":[1,2,3,4],"test_depths":[1,2,3,4,5,6,8,10,12],"bridge":"two linear probes","engine":"hard FOG modular operator grammar","probes":probes,"results":results,"limitations":["Synthetic explicit-operation benchmark","Exact modular operator primitives are structured, not learned from final-answer loss","Layer selection was made in a prior validation sweep","Paraphrase split is a limited language-shift test"]}
    Path('retrofit_v4_dual_tap_results.json').write_text(json.dumps(out,indent=2));torch.save({"value_parser":vp.state_dict(),"op_parser":op.state_dict(),"result":out},'retrofit_v4_dual_tap.pt');print("FINAL_V4",json.dumps(out,sort_keys=True),flush=True)
if __name__=='__main__':main()
