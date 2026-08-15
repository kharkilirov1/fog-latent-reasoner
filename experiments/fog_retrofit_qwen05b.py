# /// script
# dependencies = [
#   "torch>=2.3",
#   "transformers>=4.45,<5",
#   "datasets>=2.20",
#   "huggingface-hub>=0.25",
#   "accelerate>=0.33",
#   "numpy>=1.26"
# ]
# ///

import os, re, math, json, time, random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

SEED = 1234
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if DEVICE == "cuda": torch.cuda.manual_seed_all(SEED)

def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if DEVICE == "cuda": torch.cuda.manual_seed_all(s)

def single_token_map(tok):
    for p in [97,89,83,79,73,71,67,61,59,53,47,43,41,37,31]:
        ids=[]; ok=True
        for i in range(p):
            cand = tok.encode(str(i), add_special_tokens=False)
            if len(cand) != 1:
                cand = tok.encode(" "+str(i), add_special_tokens=False)
            if len(cand) != 1:
                ok=False; break
            ids.append(cand[0])
        if ok: return p, ids
    raise RuntimeError("No contiguous prime-sized single-token integer label set found")

def make_chain_example(rng, depth, p):
    x0 = rng.randrange(p); cur = x0; ops=[]
    for _ in range(depth):
        typ = rng.choice(["add","sub","mul"])
        if typ == "add":
            k=rng.randint(1,12); cur=(cur+k)%p; text=f"Add {k}."
        elif typ == "sub":
            k=rng.randint(1,12); cur=(cur-k)%p; text=f"Subtract {k}."
        else:
            k=rng.randint(2,5); cur=(cur*k)%p; text=f"Multiply by {k}."
        ops.append(text)
    return {"start":x0,"ops":ops,"answer":cur,"depth":depth}

def build_segment_ids(tok, ex, p):
    segs=[f"Work modulo {p}. Start with {ex['start']}. "]
    segs += [f"Step {i+1}: {op} " for i,op in enumerate(ex['ops'])]
    segs += ["What is the final value? Answer with only the integer.\nAnswer:"]
    ids=[]; markers=[]
    for j,s in enumerate(segs):
        part=tok.encode(s, add_special_tokens=False); ids.extend(part)
        if j <= len(ex['ops']): markers.append(len(ids)-1)
    return ids, markers

@torch.no_grad()
def extract_segment_features(base, tok, examples, p, hidden_size, batch_size=20):
    max_slots=max(x['depth'] for x in examples)+1
    all_slots=[]; all_mask=[]; all_last=[]; labels=[]; depths=[]
    pad=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    for st in range(0,len(examples),batch_size):
        batch=examples[st:st+batch_size]; built=[build_segment_ids(tok,e,p) for e in batch]
        mx=max(len(x[0]) for x in built)
        input_ids=torch.full((len(batch),mx),pad,dtype=torch.long,device=DEVICE); attn=torch.zeros((len(batch),mx),dtype=torch.long,device=DEVICE)
        for i,(ids,_) in enumerate(built):
            input_ids[i,:len(ids)]=torch.tensor(ids,device=DEVICE); attn[i,:len(ids)]=1
        h=base(input_ids=input_ids,attention_mask=attn,use_cache=False).last_hidden_state
        for i,((ids,marks),e) in enumerate(zip(built,batch)):
            slots=torch.zeros((max_slots,hidden_size),dtype=torch.float16); mask=torch.zeros(max_slots,dtype=torch.bool)
            for j,m in enumerate(marks): slots[j]=h[i,m].detach().to("cpu",torch.float16); mask[j]=True
            all_slots.append(slots); all_mask.append(mask); all_last.append(h[i,len(ids)-1].detach().to("cpu",torch.float16)); labels.append(e['answer']); depths.append(e['depth'])
        if (st//batch_size)%25==0: print(f"feature chain {st+len(batch)}/{len(examples)}",flush=True)
    return TensorDataset(torch.stack(all_slots),torch.stack(all_mask),torch.stack(all_last),torch.tensor(labels),torch.tensor(depths))

def parse_gsm_answer(s):
    m=re.search(r"####\s*([-+]?\d[\d,]*)",s)
    if not m: return None
    try: return int(m.group(1).replace(",",""))
    except: return None

@torch.no_grad()
def extract_context_features(base,tok,rows,p,hidden_size,kctx=48,batch_size=16):
    prompts=[]; labels=[]
    for r in rows:
        a=parse_gsm_answer(r['answer'])
        if a is None or not (0<=a<p): continue
        prompts.append("Solve this math problem. Answer with only the integer.\n"+r['question']+"\nAnswer:"); labels.append(a)
    ctxs=[]; masks=[]; lasts=[]; ys=[]
    for st in range(0,len(prompts),batch_size):
        ps=prompts[st:st+batch_size]
        enc=tok(ps,padding=True,truncation=True,max_length=224,return_tensors='pt',add_special_tokens=False); enc={k:v.to(DEVICE) for k,v in enc.items()}
        h=base(**enc,use_cache=False).last_hidden_state
        for i in range(len(ps)):
            n=int(enc['attention_mask'][i].sum().item()); hv=h[i,:n]; take=min(kctx,n)
            c=torch.zeros((kctx,hidden_size),dtype=torch.float16); m=torch.zeros(kctx,dtype=torch.bool)
            c[-take:]=hv[-take:].detach().to('cpu',torch.float16); m[-take:]=True
            ctxs.append(c); masks.append(m); lasts.append(hv[-1].detach().to('cpu',torch.float16)); ys.append(labels[st+i])
        if (st//batch_size)%20==0: print(f"feature gsm {st+len(ps)}/{len(prompts)}",flush=True)
    if not ys: raise RuntimeError("No GSM examples in numeric label range")
    return TensorDataset(torch.stack(ctxs),torch.stack(masks),torch.stack(lasts),torch.tensor(ys))

class LinearResidual(nn.Module):
    def __init__(self,h): super().__init__(); self.proj=nn.Linear(h,h,bias=False); nn.init.zeros_(self.proj.weight)
    def forward(self,slots,mask,last): return self.proj(last.float())
class MLPResidual(nn.Module):
    def __init__(self,h,b=256):
        super().__init__(); self.net=nn.Sequential(nn.Linear(2*h,b),nn.GELU(),nn.Linear(b,h)); nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)
    def forward(self,slots,mask,last):
        mf=mask.float().unsqueeze(-1); mean=(slots.float()*mf).sum(1)/mf.sum(1).clamp_min(1); return self.net(torch.cat([last.float(),mean],-1))
class GRUResidual(nn.Module):
    def __init__(self,h,d=128):
        super().__init__(); self.inp=nn.Linear(h,d); self.gru=nn.GRU(d,d,batch_first=True); self.out=nn.Linear(d,h); nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)
    def forward(self,slots,mask,last):
        x=self.inp(slots.float()); out,_=self.gru(x); lens=mask.sum(1).clamp_min(1)-1; v=out[torch.arange(out.size(0),device=out.device),lens]; return self.out(v)
class BilinearOp(nn.Module):
    def __init__(self,d,r=64):
        super().__init__(); self.a=nn.Linear(d,r,bias=False); self.b=nn.Linear(d,r,bias=False); self.o=nn.Linear(r,d,bias=False); self.norm=nn.LayerNorm(d)
    def forward(self,z,c): return self.norm(z+self.o(self.a(z)*self.b(c)))
def block_product(z,c):
    d=z.size(-1); d2=d//2*2; za=z[...,:d2].reshape(*z.shape[:-1],-1,2); ca=c[...,:d2].reshape(*c.shape[:-1],-1,2)
    za=F.normalize(za.float(),dim=-1); ca=F.normalize(ca.float(),dim=-1)
    re=za[...,0]*ca[...,0]-za[...,1]*ca[...,1]; im=za[...,0]*ca[...,1]+za[...,1]*ca[...,0]
    out=torch.stack([re,im],-1).reshape(*z.shape[:-1],d2)
    if d2<d: out=torch.cat([out,z[...,d2:].float()],-1)
    return out
class FOGSeqResidual(nn.Module):
    def __init__(self,h,d=128,nlearn=4):
        super().__init__(); self.start=nn.Linear(h,d); self.opproj=nn.Linear(h,d); self.ops=nn.ModuleList([BilinearOp(d) for _ in range(nlearn)]); self.router=nn.Linear(2*d,3+nlearn); self.norm=nn.LayerNorm(d); self.out=nn.Linear(d,h); nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias); self.last_routes=None
    def step(self,z,c):
        cand=[z,c,self.norm(block_product(z,c))]+[op(z,c) for op in self.ops]; probs=F.softmax(self.router(torch.cat([z,c],-1)),-1); oh=F.one_hot(probs.argmax(-1),probs.size(-1)).float(); w=oh+probs-probs.detach(); out=sum(w[:,i:i+1]*cand[i] for i in range(len(cand))); return self.norm(out),probs
    def forward(self,slots,mask,last,intervention=None,force_steps=None):
        z=self.norm(self.start(slots[:,0].float())); routes=[]; maxops=slots.size(1)-1
        if force_steps is not None: maxops=min(maxops,force_steps)
        for t in range(maxops):
            active=mask[:,t+1]
            if not active.any(): break
            c=self.norm(self.opproj(slots[:,t+1].float())); zn,p=self.step(z,c); z=torch.where(active[:,None],zn,z); routes.append(p.detach())
            if intervention=='shuffle2' and t==1: z=z[torch.randperm(z.size(0),device=z.device)]
            if intervention=='zero2' and t==1: z=torch.zeros_like(z)
        self.last_routes=routes; return self.out(z)
class ContextFOGResidual(nn.Module):
    def __init__(self,h,d=128,nlearn=4):
        super().__init__(); self.init=nn.Linear(h,d); self.k=nn.Linear(h,d,bias=False); self.v=nn.Linear(h,d,bias=False); self.q=nn.Linear(d,d,bias=False); self.ops=nn.ModuleList([BilinearOp(d) for _ in range(nlearn)]); self.router=nn.Linear(2*d,3+nlearn); self.norm=nn.LayerNorm(d); self.out=nn.Linear(d,h); nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias); self.last_routes=None
    def step(self,z,ctx,mask):
        q=F.normalize(self.q(z),dim=-1); k=F.normalize(self.k(ctx.float()),dim=-1); sc=torch.einsum('bd,bkd->bk',q,k)*10.0; sc=sc.masked_fill(~mask,-1e9); a=F.softmax(sc,-1); c=self.norm(torch.einsum('bk,bkd->bd',a,self.v(ctx.float())))
        cand=[z,c,self.norm(block_product(z,c))]+[op(z,c) for op in self.ops]; probs=F.softmax(self.router(torch.cat([z,c],-1)),-1); oh=F.one_hot(probs.argmax(-1),probs.size(-1)).float(); w=oh+probs-probs.detach(); return self.norm(sum(w[:,i:i+1]*cand[i] for i in range(len(cand)))),probs
    def forward(self,ctx,mask,last,steps=4,intervention=None):
        z=self.norm(self.init(last.float())); routes=[]
        for t in range(steps):
            z,p=self.step(z,ctx,mask); routes.append(p.detach())
            if intervention=='shuffle2' and t==1: z=z[torch.randperm(z.size(0),device=z.device)]
            if intervention=='zero2' and t==1: z=torch.zeros_like(z)
        self.last_routes=routes; return self.out(z)
class ContextMLPResidual(MLPResidual): pass

def count_params(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)
def numeric_logits(last,delta,W): return (last.float()+delta.float()) @ W.T

def train_adapter(name,model,ds,W,epochs=8,lr=2e-3,batch=128,context_mode=False,steps=4):
    seed_all(SEED+abs(hash(name))%1000); model.to(DEVICE); model.train(); opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-4); loader=DataLoader(ds,batch_size=batch,shuffle=True)
    for ep in range(epochs):
        tot=0; n=0
        for b in loader:
            b=[x.to(DEVICE) for x in b]
            if context_mode: ctx,mask,last,y=b; delta=model(ctx,mask,last,steps=steps)
            else: slots,mask,last,y,depth=b; delta=model(slots,mask,last)
            loss=F.cross_entropy(numeric_logits(last,delta,W),y); opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); tot+=float(loss)*len(y); n+=len(y)
        if ep in {0,epochs-1} or (ep+1)%4==0: print(f"train {name} epoch {ep+1}/{epochs} loss={tot/n:.5f}",flush=True)
    return model
@torch.no_grad()
def eval_chain(model,ds,W,batch=256,baseline=False,intervention=None,force_steps=None):
    if model is not None: model.eval()
    acc=defaultdict(lambda:[0,0]); rsum=None; rcount=0
    for b in DataLoader(ds,batch_size=batch,shuffle=False):
        slots,mask,last,y,depth=[x.to(DEVICE) for x in b]
        if baseline: delta=torch.zeros_like(last.float())
        elif isinstance(model,FOGSeqResidual): delta=model(slots,mask,last,intervention=intervention,force_steps=force_steps)
        else: delta=model(slots,mask,last)
        pred=numeric_logits(last,delta,W).argmax(-1)
        for d in depth.unique().tolist():
            sel=depth==d; acc[int(d)][0]+=int((pred[sel]==y[sel]).sum()); acc[int(d)][1]+=int(sel.sum())
        if isinstance(model,FOGSeqResidual) and model.last_routes:
            rr=torch.stack([x.mean(0) for x in model.last_routes]).mean(0); rsum=rr if rsum is None else rsum+rr; rcount+=1
    return {str(d):c/n for d,(c,n) in sorted(acc.items())},((rsum/rcount).cpu().tolist() if rsum is not None else None)
@torch.no_grad()
def eval_context(model,ds,W,batch=128,baseline=False,steps=4,intervention=None):
    if model is not None:model.eval()
    cor=tot=0; rsum=None;rc=0
    for b in DataLoader(ds,batch_size=batch):
        ctx,mask,last,y=[x.to(DEVICE) for x in b]
        if baseline: delta=torch.zeros_like(last.float())
        elif isinstance(model,ContextFOGResidual): delta=model(ctx,mask,last,steps=steps,intervention=intervention)
        else: delta=model(ctx,mask,last)
        pred=numeric_logits(last,delta,W).argmax(-1); cor+=int((pred==y).sum());tot+=len(y)
        if isinstance(model,ContextFOGResidual) and model.last_routes:
            rr=torch.stack([x.mean(0) for x in model.last_routes]).mean(0); rsum=rr if rsum is None else rsum+rr;rc+=1
    return cor/tot,((rsum/rc).cpu().tolist() if rsum is not None else None)

def main():
    print("DEVICE",DEVICE,torch.cuda.get_device_name(0) if DEVICE=='cuda' else 'cpu',flush=True)
    tok=AutoTokenizer.from_pretrained(MODEL_ID,use_fast=True)
    if tok.pad_token_id is None: tok.pad_token=tok.eos_token
    p,num_ids=single_token_map(tok); print("numeric modulus",p,"labels",len(num_ids),flush=True)
    model=AutoModelForCausalLM.from_pretrained(MODEL_ID,torch_dtype=DTYPE,low_cpu_mem_usage=True).to(DEVICE).eval()
    for q in model.parameters(): q.requires_grad_(False)
    base=model.model; H=model.config.hidden_size; W=model.lm_head.weight[num_ids].detach().float().to(DEVICE)
    print("hidden",H,"backbone_params",sum(x.numel() for x in model.parameters()),flush=True)
    rng=random.Random(100); train=[]
    for d in [1,2,3,4]: train += [make_chain_example(rng,d,p) for _ in range(900)]
    rng2=random.Random(200); test=[]
    for d in [1,2,3,4,5,6,8,10,12]: test += [make_chain_example(rng2,d,p) for _ in range(260)]
    print("extract lane A",len(train),len(test),flush=True); trA=extract_segment_features(base,tok,train,p,H); teA=extract_segment_features(base,tok,test,p,H); del model,base
    if DEVICE=='cuda': torch.cuda.empty_cache()
    adapters={'linear':LinearResidual(H),'mlp':MLPResidual(H,256),'gru':GRUResidual(H,128),'fog':FOGSeqResidual(H,128,4)}
    chain_res={'baseline':eval_chain(None,teA,W,baseline=True)[0]}; trained={}
    for n,a in adapters.items():
        trained[n]=train_adapter(n,a,trA,W,epochs=(10 if n=='fog' else 8),lr=2e-3,batch=128); chain_res[n],routes=eval_chain(trained[n],teA,W)
        if routes is not None: chain_res[n+'_routes']=routes
        print("CHAIN",n,chain_res[n],flush=True)
    chain_res['fog_force_r1']=eval_chain(trained['fog'],teA,W,force_steps=1)[0]; chain_res['fog_shuffle2']=eval_chain(trained['fog'],teA,W,intervention='shuffle2')[0]; chain_res['fog_zero2']=eval_chain(trained['fog'],teA,W,intervention='zero2')[0]
    params={n:count_params(a) for n,a in trained.items()}; saved={n:{k:v.detach().cpu() for k,v in a.state_dict().items()} for n,a in trained.items()}; print("PARAMS",params,flush=True)
    del trained,adapters,trA,teA
    if DEVICE=='cuda': torch.cuda.empty_cache()
    model=AutoModelForCausalLM.from_pretrained(MODEL_ID,torch_dtype=DTYPE,low_cpu_mem_usage=True).to(DEVICE).eval(); [q.requires_grad_(False) for q in model.parameters()]; base=model.model
    gsm=load_dataset("openai/gsm8k","main")
    trrows=[r for r in gsm['train'] if (lambda a:a is not None and 0<=a<p)(parse_gsm_answer(r['answer']))]; terows=[r for r in gsm['test'] if (lambda a:a is not None and 0<=a<p)(parse_gsm_answer(r['answer']))]
    random.Random(300).shuffle(trrows); random.Random(301).shuffle(terows); trrows=trrows[:1200]; terows=terows[:320]; print("GSM filtered",len(trrows),len(terows),flush=True)
    trB=extract_context_features(base,tok,trrows,p,H,kctx=48); teB=extract_context_features(base,tok,terows,p,H,kctx=48); del model,base
    if DEVICE=='cuda':torch.cuda.empty_cache()
    mlpB=ContextMLPResidual(H,256); fogB=ContextFOGResidual(H,128,4); gsm_res={'baseline':eval_context(None,teB,W,baseline=True)[0]}
    mlpB=train_adapter('gsm_mlp',mlpB,trB,W,epochs=12,lr=1.5e-3,batch=96,context_mode=True,steps=4); fogB=train_adapter('gsm_fog',fogB,trB,W,epochs=16,lr=1.5e-3,batch=96,context_mode=True,steps=4)
    gsm_res['mlp']=eval_context(mlpB,teB,W)[0]
    for r in [1,2,4,8]: gsm_res[f'fog_r{r}']=eval_context(fogB,teB,W,steps=r)[0]
    gsm_res['fog_shuffle2']=eval_context(fogB,teB,W,steps=4,intervention='shuffle2')[0]; gsm_res['fog_zero2']=eval_context(fogB,teB,W,steps=4,intervention='zero2')[0]; gsm_res['fog_routes']=eval_context(fogB,teB,W,steps=4)[1]
    print("GSM",gsm_res,flush=True)
    result={'model':MODEL_ID,'device':DEVICE,'modulus':p,'seed':SEED,'lane_a':{'train_depths':[1,2,3,4],'test_depths':[1,2,3,4,5,6,8,10,12],'n_train':len(train),'n_test':len(test),'params':params,'results':chain_res},'lane_b':{'benchmark':'openai/gsm8k official split, final answer constrained to numeric label range','n_train':len(trB),'n_test':len(teB),'results':gsm_res},'limitations':['numeric constrained decoding','GSM8K subset with answers within numeric label range','frozen backbone','retrofit modules trained only']}
    Path('/tmp/results.json').write_text(json.dumps(result,indent=2)); torch.save({'gsm_mlp':mlpB.state_dict(),'gsm_fog':fogB.state_dict(),'chain_adapters':saved,'result':result},'/tmp/adapters.pt'); print("FINAL_RESULTS_JSON",json.dumps(result,sort_keys=True),flush=True)
    token=os.environ.get('HF_TOKEN')
    if token:
        try:
            from huggingface_hub import HfApi,create_repo,whoami
            user=whoami(token=token)['name']; repo=f"{user}/fog-retrofit-qwen05b-exp1"; create_repo(repo,repo_type='model',private=True,exist_ok=True,token=token); api=HfApi(token=token); api.upload_file(path_or_fileobj='/tmp/results.json',path_in_repo='results.json',repo_id=repo,repo_type='model'); api.upload_file(path_or_fileobj='/tmp/adapters.pt',path_in_repo='adapters.pt',repo_id=repo,repo_type='model'); print("PERSISTED_REPO",repo,flush=True)
        except Exception as e: print("PERSIST_FAILED",repr(e),flush=True)
if __name__=='__main__': main()
