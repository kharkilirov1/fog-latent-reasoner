import json, random
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM

SEED=20260819
MODEL_ID="HuggingFaceTB/SmolLM2-135M-Instruct"
P=31
random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.set_num_threads(4)

def make_ex(rng,depth,style="base"):
    cur=rng.randrange(P);start=cur;ops=[]
    for _ in range(depth):
        kind=rng.choice(("add","sub","mul"))
        if kind=="add":k=rng.randint(1,8);cur=(cur+k)%P;code=k-1
        elif kind=="sub":k=rng.randint(1,8);cur=(cur-k)%P;code=8+k-1
        else:k=rng.randint(2,5);cur=(cur*k)%P;code=16+(k-2)
        ops.append((kind,k,code))
    return {"start":start,"ops":ops,"answer":cur,"depth":depth,"style":style}

def op_text(kind,k,style):
    if style=="base":
        return f"Add {k}." if kind=="add" else (f"Subtract {k}." if kind=="sub" else f"Multiply by {k}.")
    if style=="para":
        return f"Increase the current value by {k}." if kind=="add" else (f"Decrease the current value by {k}." if kind=="sub" else f"Scale the current value by a factor of {k}.")
    return f"Now plus {k}." if kind=="add" else (f"Now take away {k}." if kind=="sub" else f"Now take {k} times the current value.")

def build(tok,e):
    seg=[f"Work modulo {P}. Start with {e['start']}. "]
    for i,(kind,k,_) in enumerate(e['ops']):seg.append(f"Step {i+1}: {op_text(kind,k,e['style'])} ")
    seg.append("What is the final value? Answer with only the integer.\nAnswer:")
    ids=[];marks=[]
    for j,s in enumerate(seg):
        ids.extend(tok.encode(s,add_special_tokens=False))
        if j<=len(e['ops']):marks.append(len(ids)-1)
    return ids,marks

@torch.no_grad()
def feats(base,tok,examples,h,batch=12):
    mxops=max(e['depth'] for e in examples);pad=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    START=[];OPS=[];MASK=[];SY=[];OY=[];ANS=[];DEP=[]
    for st in range(0,len(examples),batch):
        bb=examples[st:st+batch];built=[build(tok,e) for e in bb];mx=max(len(x[0]) for x in built)
        ids=torch.full((len(bb),mx),pad,dtype=torch.long);am=torch.zeros_like(ids)
        for i,(a,_) in enumerate(built):ids[i,:len(a)]=torch.tensor(a);am[i,:len(a)]=1
        hs=base(input_ids=ids,attention_mask=am,use_cache=False).last_hidden_state.float()
        for i,((a,marks),e) in enumerate(zip(built,bb)):
            opf=torch.zeros(mxops,h);mask=torch.zeros(mxops,dtype=torch.bool);opy=torch.full((mxops,),-1,dtype=torch.long)
            for t,(_,_,code) in enumerate(e['ops']):opf[t]=hs[i,marks[t+1]];mask[t]=1;opy[t]=code
            START.append(hs[i,marks[0]]);OPS.append(opf);MASK.append(mask);SY.append(e['start']);OY.append(opy);ANS.append(e['answer']);DEP.append(e['depth'])
        if st%240==0:print("features",st+len(bb),"/",len(examples),flush=True)
    return TensorDataset(torch.stack(START),torch.stack(OPS),torch.stack(MASK),torch.tensor(SY),torch.stack(OY),torch.tensor(ANS),torch.tensor(DEP))

class Parser(nn.Module):
    def __init__(self,h,n,b=256):super().__init__();self.net=nn.Sequential(nn.LayerNorm(h),nn.Linear(h,b),nn.GELU(),nn.Linear(b,n))
    def forward(self,x):return self.net(x)

def train_parser(model,x,y,epochs=20,lr=2e-3):
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-4);ds=TensorDataset(x,y)
    for ep in range(epochs):
        for a,b in DataLoader(ds,batch_size=128,shuffle=True):
            loss=F.cross_entropy(model(a),b);opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
    return model

def flatten_ops(ds):
    X=[];Y=[]
    for start,ops,mask,sy,oy,ans,dep in DataLoader(ds,batch_size=128):
        q=mask.reshape(-1);X.append(ops.reshape(-1,ops.size(-1))[q]);Y.append(oy.reshape(-1)[q])
    return torch.cat(X),torch.cat(Y)

@torch.no_grad()
def parser_acc(model,x,y):return float((model(x).argmax(-1)==y).float().mean())

def apply_opcode(x,code):
    if code<8:return (x+(code+1))%P
    if code<16:return (x-(code-8+1))%P
    return (x*(code-16+2))%P

@torch.no_grad()
def evaluate(start_parser,op_parser,ds,mode="pred"):
    by=defaultdict(lambda:[0,0]);op_total=op_correct=0;start_total=start_correct=0
    for sf,of,mask,sy,oy,ans,dep in DataLoader(ds,batch_size=128):
        ps=start_parser(sf).argmax(-1);pop=op_parser(of.reshape(-1,of.size(-1))).argmax(-1).reshape_as(oy)
        start_correct+=int((ps==sy).sum());start_total+=len(sy);q=mask;op_correct+=int((pop[q]==oy[q]).sum());op_total+=int(q.sum())
        for i in range(len(sy)):
            if mode=="oracle":cur=int(sy[i])
            elif mode=="pred_start_oracle_ops":cur=int(ps[i])
            elif mode=="oracle_start_pred_ops":cur=int(sy[i])
            else:cur=int(ps[i])
            for t in range(int(dep[i])):
                code=int(oy[i,t]) if mode in ("oracle","pred_start_oracle_ops") else int(pop[i,t])
                cur=apply_opcode(cur,code)
            by[int(dep[i])][0]+=int(cur==int(ans[i]));by[int(dep[i])][1]+=1
    return {str(k):v[0]/v[1] for k,v in sorted(by.items())},{"start_acc":start_correct/start_total,"op_acc":op_correct/op_total}

def main():
    tok=AutoTokenizer.from_pretrained(MODEL_ID);tok.pad_token=tok.pad_token or tok.eos_token
    lm=AutoModelForCausalLM.from_pretrained(MODEL_ID);lm.eval();[p.requires_grad_(False) for p in lm.parameters()];base=lm.model;h=lm.config.hidden_size
    print("backbone",MODEL_ID,"hidden",h,"params",sum(p.numel() for p in lm.parameters()),flush=True)
    r=random.Random(10);train=sum(([make_ex(r,d,"base") for _ in range(500)] for d in [1,2,3,4]),[])
    v=random.Random(20);val=sum(([make_ex(v,d,"base") for _ in range(100)] for d in [1,2,3,4]),[])
    q=random.Random(30);test=sum(([make_ex(q,d,"base") for _ in range(100)] for d in [1,2,3,4,5,6,8,10,12]),[])
    z=random.Random(40);para=sum(([make_ex(z,d,"para") for _ in range(100)] for d in [1,2,3,4,5,6,8,10,12]),[])
    tr=feats(base,tok,train,h);va=feats(base,tok,val,h);te=feats(base,tok,test,h);pa=feats(base,tok,para,h)
    start=Parser(h,P,256);op=Parser(h,20,256)
    start=train_parser(start,tr.tensors[0],tr.tensors[3],20,2e-3);ox,oy=flatten_ops(tr);op=train_parser(op,ox,oy,25,2e-3)
    vox,voy=flatten_ops(va);tox,toy=flatten_ops(te);pox,poy=flatten_ops(pa)
    probes={"start_train":parser_acc(start,tr.tensors[0],tr.tensors[3]),"start_val":parser_acc(start,va.tensors[0],va.tensors[3]),"start_test":parser_acc(start,te.tensors[0],te.tensors[3]),"op_train":parser_acc(op,ox,oy),"op_val":parser_acc(op,vox,voy),"op_test":parser_acc(op,tox,toy),"op_paraphrase":parser_acc(op,pox,poy)}
    print("PARSERS",probes,flush=True)
    results={}
    for name,mode in [("oracle","oracle"),("predicted","pred"),("pred_start_oracle_ops","pred_start_oracle_ops"),("oracle_start_pred_ops","oracle_start_pred_ops")]:results[name]=evaluate(start,op,te,mode)[0]
    results["predicted_paraphrase"]=evaluate(start,op,pa,"pred")[0]
    print("RESULTS",json.dumps(results,sort_keys=True),flush=True)
    out={"model":MODEL_ID,"frozen_backbone":True,"modulus":P,"train_depths":[1,2,3,4],"test_depths":[1,2,3,4,5,6,8,10,12],"bridge_supervision":"explicit value/opcode auxiliary labels","engine":"hard finite operator grammar with exact modular ADD/SUB/MUL primitives","n_train":len(tr),"n_test":len(te),"probes":probes,"results":results,"limitations":["This is a staged semantic-compiler upper bound, not final-answer-only training","Operator grammar uses exact modular primitives","Synthetic explicit operation language; paraphrase split tests limited language shift"]}
    Path('retrofit_v3_semantic_results.json').write_text(json.dumps(out,indent=2));torch.save({"start_parser":start.state_dict(),"op_parser":op.state_dict(),"result":out},'retrofit_v3_semantic_parsers.pt');print("FINAL_V3",json.dumps(out,sort_keys=True),flush=True)
if __name__=='__main__':main()
