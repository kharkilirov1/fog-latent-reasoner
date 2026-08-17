from __future__ import annotations
import argparse, json, math, os, random, subprocess, sys, time, hashlib, platform, copy
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

SEED=20260817
ENTITIES=("Alice","Bob","Carla","Daniel","Erin","Farid","Grace","Hector","Iris","Kai","Lena","Mira")
RELATIONS=("manager","parent","owner","north","calls","imports")
OPCODES=("BIND","LOAD","FOLLOW","COMPARE","SELECT","HALT")
E=len(ENTITIES); R=len(RELATIONS); O=len(OPCODES)
EID={x:i for i,x in enumerate(ENTITIES)}; RID={x:i for i,x in enumerate(RELATIONS)}; OID={x:i for i,x in enumerate(OPCODES)}
HELD_FOLLOW_REL="calls"; HELD_COMPARE_ENTITY="Kai"; HELD_SELECT_PAIR=("Iris","Lena")

TRAIN_TEMPLATES={
"BIND":(
    "Record that {a}'s {r} is {b}.", "Set the {r} relation from {a} to {b}.",
    "Store a {r} edge from {a} toward {b}.", "Remember: by {r}, {a} reaches {b}.",
),
"LOAD":("Focus on {a}.","Set the current entity to {a}.","Load {a} into the entity register.","Begin from {a}."),
"FOLLOW":(
    "Follow the {r} relation from the current entity.", "Move along {r} from the current entity.",
    "Replace the current entity by its {r} target.", "Traverse the active entity's {r} edge.",
),
"COMPARE":(
    "Check whether the current entity is {a}.", "Compare the current entity with {a}.",
    "Test current-entity equality against {a}.", "Ask if the active entity equals {a}.",
),
"SELECT":(
    "If the comparison is true choose {a}; otherwise choose {b}.", "Select {a} on true and {b} on false.",
    "Use {a} if the predicate holds, else use {b}.", "Truth selects {a}; falsehood selects {b}.",
),
"HALT":("Stop execution and return the current entity.","Halt now with the current entity.","Finish the program without changing the current entity.","End here and keep the active entity."),
}
SCAN_TEMPLATES={
"BIND":("Memorize a {r} edge leading from {a} to {b}.","Associate {a} with {b} through relation {r}."),
"LOAD":("Make {a} the active entity.","Begin tracking {a}."),
"FOLLOW":("Traverse the current entity's {r} link.","Jump through relation {r}."),
"COMPARE":("Ask whether we are currently at {a}.","Evaluate current equals {a}."),
"SELECT":("Choose {a} when true; choose {b} when false.","True means {a}, false means {b}."),
"HALT":("End computation here.","Freeze the current entity and stop."),
}
TEST_TEMPLATES={
"BIND":(
    "Let {b} be what you reach from {a} by the {r} connection.", "In this world, {a} points by {r} to {b}.",
    "The {r} edge leaving {a} lands on {b}.", "From {a}, relation {r} identifies {b}.",
),
"LOAD":("Turn attention to {a}.","Take {a} as the entity we are standing on.","Our current subject is now {a}.","Stand conceptually on {a}."),
"FOLLOW":("Travel through the {r} link of whoever is current.","Go to the node reached via {r}.","Step from here to its {r} destination.","Resolve the current entity through {r}."),
"COMPARE":("Decide whether the entity in hand is {a}.","Is the current node actually {a}?","Determine if our present entity matches {a}.","Judge whether the active subject is {a}."),
"SELECT":("If that check succeeded, switch to {a}; if it failed, switch to {b}.","On yes take {a}; on no take {b}.","Let truth pick {a} and falsehood pick {b}.","The true branch yields {a}, the false branch {b}."),
"HALT":("We are done; keep this entity as the answer.","Do nothing else and return the entity currently held.","Terminate with the current subject.","Stop processing and preserve the active entity."),
}
REL_BIND_TRAIN={
"manager":("{a} reports to {b}.","{b} manages {a}."),"parent":("{b} is the parent of {a}.","{a}'s parent is {b}."),
"owner":("{b} owns {a}.","{a} belongs to {b}."),"north":("{b} is north of {a}.","From {a}, moving north reaches {b}."),
"calls":("Function {a} calls function {b}.","A call from {a} enters {b}."),"imports":("Module {a} imports module {b}.","{a} depends on {b} through an import."),
}
REL_FOLLOW_TRAIN={
"manager":("Go to whoever manages the current person.",),"parent":("Move to the current person's parent.",),
"owner":("Move to whoever owns the current item.",),"north":("Move one linked location north.",),
"calls":("Jump to the function called by the current function.",),"imports":("Follow the current module's import dependency.",),
}
REL_BIND_TEST={
"manager":("{a} answers professionally to {b}.",),"parent":("The parent linked to {a} is {b}.",),
"owner":("Possession of {a} rests with {b}.",),"north":("The place immediately north-linked from {a} is {b}.",),
"calls":("Execution in {a} transfers by a call to {b}.",),"imports":("The import dependency leaving {a} resolves to {b}.",),
}
REL_FOLLOW_TEST={
"manager":("Climb to the manager of the current person.",),"parent":("Follow the parental link from the current person.",),
"owner":("Resolve the owner of the current object.",),"north":("Take the northward relation from this place.",),
"calls":("Enter the callee reached from the current function.",),"imports":("Traverse the import edge from this module.",),
}
ENTITY_ANCHORS=[(x,f"entity {x}",f"person or object named {x}") for x in ENTITIES]
REL_ANCHORS={
"manager":("manager","supervisor","reports to"),"parent":("parent","mother or father","parental link"),
"owner":("owner","owns","belongs to"),"north":("north","north of","northward"),
"calls":("calls","callee","function call"),"imports":("imports","import dependency","module import"),
}
OP_ANCHORS={
"BIND":("record relation","store fact","bind edge"),"LOAD":("load entity","focus entity","set current"),
"FOLLOW":("follow relation","traverse link","move along edge"),"COMPARE":("compare entity","check equality","test whether"),
"SELECT":("select on condition","choose if true otherwise","conditional select"),"HALT":("stop","halt","finish execution"),
}

@dataclass(frozen=True)
class Instr:
    op:int; e1:int=-1; rel:int=-1; e2:int=-1
@dataclass(frozen=True)
class LogicProgram:
    instructions: tuple[Instr,...]; answer:int; chain_len:int; contains_heldout:bool

def set_seed(seed:int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def execute_oracle(program:LogicProgram)->int:
    mem=torch.full((R,E),-1,dtype=torch.long); current=0; pred=False; halted=False
    for ins in program.instructions:
        if halted: continue
        if ins.op==OID['BIND']: mem[ins.rel,ins.e1]=ins.e2
        elif ins.op==OID['LOAD']: current=ins.e1
        elif ins.op==OID['FOLLOW']:
            nxt=int(mem[ins.rel,current]);
            if nxt<0: raise RuntimeError('undefined follow')
            current=nxt
        elif ins.op==OID['COMPARE']: pred=(current==ins.e1)
        elif ins.op==OID['SELECT']: current=ins.e1 if pred else ins.e2
        elif ins.op==OID['HALT']: halted=True
    return current

def generate_programs(seed:int, chain_lens:Sequence[int], per_len:int, *, train:bool)->list[LogicProgram]:
    rng=random.Random(seed); out=[]; seen=set(); allowed_follow=list(range(R))
    if train: allowed_follow.remove(RID[HELD_FOLLOW_REL])
    attempts=0
    while len(out)<per_len*len(chain_lens):
        attempts+=1
        if attempts>500000: raise RuntimeError('sampling exhausted')
        cl=chain_lens[(len(out)//per_len)%len(chain_lens)]
        force_held=(not train and (len(out)%2==0))
        rels=[rng.choice(allowed_follow if not force_held else list(range(R))) for _ in range(cl)]
        if force_held: rels[rng.randrange(cl)]=RID[HELD_FOLLOW_REL]
        chain=[rng.randrange(E)]
        for _ in range(cl):
            nxt=rng.randrange(E)
            while nxt==chain[-1]: nxt=rng.randrange(E)
            chain.append(nxt)
        if len({(rels[i],chain[i]) for i in range(cl)}) != cl: continue
        facts=[Instr(OID['BIND'],chain[i],rels[i],chain[i+1]) for i in range(cl)]
        protected={(rels[i],chain[i]) for i in range(cl)}
        for _ in range(rng.randint(1,2)):
            for _attempt in range(100):
                a,b=rng.randrange(E),rng.randrange(E); rel=rng.randrange(R)
                if a==b: b=(b+1)%E
                if (rel,a) not in protected:
                    facts.append(Instr(OID['BIND'],a,rel,b)); protected.add((rel,a)); break
        rng.shuffle(facts)
        compare_true=rng.random()<0.5
        compare_ent=chain[-1] if compare_true else rng.choice([x for x in range(E) if x!=chain[-1]])
        if train and compare_ent==EID[HELD_COMPARE_ENTITY]: compare_ent=(compare_ent+1)%E
        if force_held and rng.random()<0.5:
            compare_ent=EID[HELD_COMPARE_ENTITY]; compare_true=(chain[-1]==compare_ent)
        while True:
            t,f=rng.sample(range(E),2)
            if not train and force_held and rng.random()<0.5: t,f=EID[HELD_SELECT_PAIR[0]],EID[HELD_SELECT_PAIR[1]]
            if train and (t,f)==(EID[HELD_SELECT_PAIR[0]],EID[HELD_SELECT_PAIR[1]]): continue
            break
        ans=t if compare_true else f
        instr=tuple(facts+[Instr(OID['LOAD'],chain[0])]+[Instr(OID['FOLLOW'],rel=r) for r in rels]+[Instr(OID['COMPARE'],compare_ent),Instr(OID['SELECT'],t,e2=f),Instr(OID['HALT'])])
        instr=instr+(Instr(OID['LOAD'],rng.randrange(E)),Instr(OID['FOLLOW'],rel=rng.randrange(R)))
        key=tuple((x.op,x.e1,x.rel,x.e2) for x in instr)
        if key in seen: continue
        seen.add(key)
        held=(RID[HELD_FOLLOW_REL] in rels or compare_ent==EID[HELD_COMPARE_ENTITY] or (t,f)==(EID[HELD_SELECT_PAIR[0]],EID[HELD_SELECT_PAIR[1]]))
        p=LogicProgram(instr,ans,cl,held)
        if execute_oracle(p)!=ans: continue
        out.append(p)
    return out

def instr_text(ins:Instr,family:str,variant:int=0)->str:
    T={'train':TRAIN_TEMPLATES,'scan':SCAN_TEMPLATES,'test':TEST_TEMPLATES}[family]
    op=OPCODES[ins.op]; a=ENTITIES[ins.e1] if ins.e1>=0 else None; b=ENTITIES[ins.e2] if ins.e2>=0 else None; r=RELATIONS[ins.rel] if ins.rel>=0 else None
    if op=='BIND' and variant%2==1:
        pool=(REL_BIND_TRAIN if family!='test' else REL_BIND_TEST)[r]; return pool[variant%len(pool)].format(a=a,b=b)
    if op=='FOLLOW' and variant%2==1:
        pool=(REL_FOLLOW_TRAIN if family!='test' else REL_FOLLOW_TEST)[r]; return pool[variant%len(pool)]
    return T[op][variant%len(T[op])].format(a=a,b=b,r=r)

class PhraseSpec:
    def __init__(self):
        self.texts=[]; self.idx={}; self.train_examples=[]; self.scan_examples=[]; self.test_examples=[]
        self.entity_anchor_ids=[]; self.rel_anchor_ids=[]; self.op_anchor_ids=[]; self._build()
    def add(self,s):
        if s not in self.idx: self.idx[s]=len(self.texts); self.texts.append(s)
        return self.idx[s]
    def add_example(self,family,ins,variant):
        pid=self.add(instr_text(ins,family,variant)); row=(pid,ins.op,ins.e1,ins.rel,ins.e2)
        {'train':self.train_examples,'scan':self.scan_examples,'test':self.test_examples}[family].append(row)
    def _build(self):
        rng=random.Random(SEED)
        for family,nvar in [('train',4),('scan',2),('test',4)]:
            for rel in range(R):
                for _ in range(32 if family=='train' else 14):
                    a,b=rng.sample(range(E),2)
                    for v in range(nvar): self.add_example(family,Instr(OID['BIND'],a,rel,b),v)
            for a in range(E):
                for v in range(nvar):
                    self.add_example(family,Instr(OID['LOAD'],a),v)
                    if not (family!='test' and a==EID[HELD_COMPARE_ENTITY]): self.add_example(family,Instr(OID['COMPARE'],a),v)
            for rel in range(R):
                if family!='test' and rel==RID[HELD_FOLLOW_REL]: continue
                for v in range(nvar*2): self.add_example(family,Instr(OID['FOLLOW'],rel=rel),v)
            pairs=[]
            for _ in range(96 if family=='train' else 40):
                a,b=rng.sample(range(E),2)
                if family!='test' and (a,b)==(EID[HELD_SELECT_PAIR[0]],EID[HELD_SELECT_PAIR[1]]): continue
                pairs.append((a,b))
            if family=='test': pairs.append((EID[HELD_SELECT_PAIR[0]],EID[HELD_SELECT_PAIR[1]]))
            for a,b in pairs:
                for v in range(nvar): self.add_example(family,Instr(OID['SELECT'],a,e2=b),v)
            for v in range(nvar*5): self.add_example(family,Instr(OID['HALT']),v)
        for e in range(E): self.entity_anchor_ids.append([self.add(x) for x in ENTITY_ANCHORS[e]])
        for r in range(R): self.rel_anchor_ids.append([self.add(x) for x in REL_ANCHORS[RELATIONS[r]]])
        for o in range(O): self.op_anchor_ids.append([self.add(x) for x in OP_ANCHORS[OPCODES[o]]])
    def manifest(self):
        return {'n_texts':len(self.texts),'sha256':hashlib.sha256('\n'.join(self.texts).encode()).hexdigest(),'held_follow_relation':HELD_FOLLOW_REL,'held_compare_entity':HELD_COMPARE_ENTITY,'held_select_pair':HELD_SELECT_PAIR}

def ensure_deps():
    try: import transformers # noqa
    except Exception: subprocess.check_call([sys.executable,'-m','pip','install','-q','transformers>=4.45,<6','accelerate>=0.34','safetensors>=0.4'])

def load_qwen(model_id,device):
    from transformers import AutoModelForCausalLM,AutoTokenizer
    tok=AutoTokenizer.from_pretrained(model_id,use_fast=True); tok.pad_token=tok.pad_token or tok.eos_token; tok.padding_side='right'
    dtype=torch.float16 if device.type=='cuda' else torch.float32
    try: model=AutoModelForCausalLM.from_pretrained(model_id,dtype=dtype,low_cpu_mem_usage=True)
    except TypeError: model=AutoModelForCausalLM.from_pretrained(model_id,torch_dtype=dtype,low_cpu_mem_usage=True)
    model=model.to(device).eval(); [p.requires_grad_(False) for p in model.parameters()]; return model,tok

@torch.inference_mode()
def pooled_all_layers(model,tok,texts,ids,device,batch=32,max_tokens=48):
    d=model.config.hidden_size; nl=model.config.num_hidden_layers+1; out=torch.zeros(len(texts),nl,2*d,dtype=torch.float16)
    for st in range(0,len(ids),batch):
        b_ids=ids[st:st+batch]; rows=[texts[i] for i in b_ids]
        enc=tok(rows,padding=True,truncation=True,max_length=max_tokens,return_tensors='pt',add_special_tokens=False); enc={k:v.to(device) for k,v in enc.items()}
        res=model.model(**enc,use_cache=False,return_dict=True,output_hidden_states=True); am=enc['attention_mask'].bool(); pos=torch.arange(am.size(1),device=device)[None]; li=pos.masked_fill(~am,-1).max(1).values; rr=torch.arange(am.size(0),device=device); den=am.sum(1,keepdim=True).clamp_min(1)
        for l,h in enumerate(res.hidden_states): out[b_ids,l]=torch.cat([(h.float()*am[...,None]).sum(1)/den,h[rr,li].float()],-1).cpu().half()
    return out

def centroid_acc(tx,ty,ex,ey,nc):
    tx=F.normalize(tx.float(),dim=-1); ex=F.normalize(ex.float(),dim=-1); cs=[]
    for c in range(nc):
        rows=tx[ty==c]; cs.append(F.normalize(rows.mean(0),dim=-1) if len(rows) else torch.zeros(tx.size(1)))
    return float((ex@torch.stack(cs).T).argmax(-1).eq(ey).float().mean())

def scan_layers(spec,pooled):
    tr=spec.train_examples; ev=spec.scan_examples
    def metric(col,nc):
        a=[r for r in tr if r[col]>=0]; b=[r for r in ev if r[col]>=0]
        return torch.tensor([r[0] for r in a]),torch.tensor([r[col] for r in a]),torch.tensor([r[0] for r in b]),torch.tensor([r[col] for r in b]),nc
    tasks=[metric(1,O),metric(2,E),metric(3,R),metric(4,E)]; rows=[]
    for l in range(pooled.size(1)):
        vals=[]
        for tid,ty,eid,ey,nc in tasks: vals.append(centroid_acc(pooled[tid,l],ty,pooled[eid,l],ey,nc) if len(eid) else 1.0)
        score=.35*vals[0]+.25*vals[1]+.25*vals[2]+.15*vals[3]
        rows.append({'layer':l,'score':score,'opcode':vals[0],'e1':vals[1],'relation':vals[2],'e2':vals[3]})
    rows=sorted(rows,key=lambda x:(-x['score'],x['layer'])); return {'top3':[x['layer'] for x in rows[:3]],'ranked':rows}

@torch.inference_mode()
def token_features(model,tok,texts,layers,device,batch=24,max_tokens=64):
    d=model.config.hidden_size; feat=torch.zeros(len(texts),len(layers),max_tokens,d,dtype=torch.float16); mask=torch.zeros(len(texts),max_tokens,dtype=torch.bool)
    for st in range(0,len(texts),batch):
        rows=texts[st:st+batch]; enc=tok(rows,padding=True,truncation=True,max_length=max_tokens,return_tensors='pt',add_special_tokens=False); enc={k:v.to(device) for k,v in enc.items()}; res=model.model(**enc,use_cache=False,return_dict=True,output_hidden_states=True); am=enc['attention_mask'].bool()
        for i in range(len(rows)):
            n=int(am[i].sum()); mask[st+i,:n]=True
            for j,l in enumerate(layers): feat[st+i,j,:n]=res.hidden_states[l][i,:n].detach().cpu().half()
    return feat,mask

class LayerMixer(nn.Module):
    def __init__(self,k):
        super().__init__(); x=torch.zeros(k); x[0]=1.5; self.logits=nn.Parameter(x); self.register_buffer('prior',torch.softmax(x,0))
    def w(self): return torch.softmax(self.logits.float(),0)
    def ctx(self,x): return torch.einsum('k,bktd->btd',self.w().to(x),x)
    def proto(self,x): return torch.einsum('k,...kd->...d',self.w().to(x),x)

class PrototypeBinder(nn.Module):
    def __init__(self,d,proto,rank=128):
        super().__init__(); self.register_buffer('proto_layers',proto.float()); self.q=nn.Linear(d,rank,bias=False); self.k=nn.Linear(d,rank,bias=False); self.scale=nn.Parameter(torch.tensor(math.log(10.0))); self.k.weight.data.copy_(self.q.weight.data)
    def token_class_score(self,ctx,mix):
        p=mix.proto(self.proto_layers); q=F.normalize(self.q(ctx).float(),dim=-1); k=F.normalize(self.k(p).float(),dim=-1); return self.scale.exp().clamp(max=100)*torch.einsum('btr,car->btca',q,k)
    def forward(self,ctx,mask,mix):
        s=self.token_class_score(ctx,mix); s=s.masked_fill(~mask[:,:,None,None],float('-inf')); return torch.logsumexp(s,dim=(1,3))

class SharedEntityBinder(nn.Module):
    """Shared entity identity geometry; separate role gates only choose which mention is e1/e2."""
    def __init__(self,d,proto,rank=128):
        super().__init__(); self.base=PrototypeBinder(d,proto,rank); self.role1=nn.Linear(d,1,bias=False); self.role2=nn.Linear(d,1,bias=False)
    def logits(self,ctx,mask,mix,role:int):
        s=self.base.token_class_score(ctx,mix)  # B,T,E,A
        gate=(self.role1(ctx) if role==1 else self.role2(ctx)).squeeze(-1)
        gate=gate.masked_fill(~mask,float('-inf')); gate=torch.log_softmax(gate.float(),dim=1)
        s=s+gate[:,:,None,None]; s=s.masked_fill(~mask[:,:,None,None],float('-inf'))
        return torch.logsumexp(s,dim=(1,3))


def anchor_proto(features,mask,nested):
    rows=[]
    for ids in nested:
        ar=[]
        for pid in ids:
            f=features[pid].float(); m=mask[pid].float()[None,:,None]; ar.append((f*m).sum(1)/m.sum(1).clamp_min(1))
        rows.append(torch.stack(ar))
    return torch.stack(rows)

class LogicFOGV5(nn.Module):
    def __init__(self,d,k,op_proto,e_proto,r_proto):
        super().__init__(); self.mix=LayerMixer(k); self.op=PrototypeBinder(d,op_proto,128); self.ent=SharedEntityBinder(d,e_proto,128); self.rel=PrototypeBinder(d,r_proto,96)
    def logits(self,f,m):
        ctx=self.mix.ctx(f.float()); return self.op(ctx,m,self.mix),self.ent.logits(ctx,m,self.mix,1),self.rel(ctx,m,self.mix),self.ent.logits(ctx,m,self.mix,2)

# ----------------------------- v6 logical compiler -----------------------------
# The v5 prototype heads are not rational-ReLU networks: cosine normalization,
# exp(scale) and logsumexp lie outside the exact class characterized by
# Heiman--Kuusisto--Turunen (2026).  v6 therefore does not pretend to translate
# the v5 head directly.  Instead it learns an evidence-gated rational-ReLU
# semantic compiler over frozen Qwen latent features.  Its forward weights are
# on a fixed dyadic rational grid, so the compiled head itself is an exact
# rational-weight ReLU network.

def rational_ste(x:torch.Tensor, denominator:int)->torch.Tensor:
    q=torch.round(x*denominator)/float(denominator)
    return x+(q-x).detach()

class RationalLinear(nn.Linear):
    def __init__(self,in_features,out_features,bias=True,denominator=1024):
        super().__init__(in_features,out_features,bias=bias); self.denominator=int(denominator)
    def forward(self,x):
        w=rational_ste(self.weight,self.denominator)
        b=rational_ste(self.bias,self.denominator) if self.bias is not None else None
        return F.linear(x,w,b)


def compiler_pool(f:torch.Tensor,m:torch.Tensor)->torch.Tensor:
    """Fixed-size latent input: per-selected-layer mean + last token, then flatten.

    The compiler theorem applies to the ReLU head as a function of this pooled
    latent vector.  Pooling is kept outside the claimed logical translation.
    """
    x=f.float(); mm=m.float()
    den=mm.sum(1).clamp_min(1.0)
    mean=(x*mm[:,None,:,None]).sum(2)/den[:,None,None]
    li=m.long().sum(1).clamp_min(1)-1
    rr=torch.arange(x.size(0),device=x.device)
    last=x[rr,:,li,:]
    return torch.cat([mean,last],-1).flatten(1)

class RationalReLUCompiler(nn.Module):
    """One-hidden-layer rational-weight ReLU semantic interface.

    Each output logit has the exact ReLU-term form
      b_c + sum_j v_cj ReLU(b_j + sum_i w_ji x_i)
    with all trainable coefficients in (1/denominator) * Z at forward time.
    """
    def __init__(self,d,k,width=128,denominator=1024):
        super().__init__(); self.d=int(d); self.k=int(k); self.width=int(width); self.denominator=int(denominator)
        inp=2*self.k*self.d
        self.hidden=RationalLinear(inp,self.width,denominator=self.denominator)
        self.op=RationalLinear(self.width,O,denominator=self.denominator)
        self.e1=RationalLinear(self.width,E,denominator=self.denominator)
        self.rel=RationalLinear(self.width,R,denominator=self.denominator)
        self.e2=RationalLinear(self.width,E,denominator=self.denominator)
    def logits_from_x(self,x):
        h=F.relu(self.hidden(x.float()))
        return self.op(h),self.e1(h),self.rel(h),self.e2(h)
    def logits(self,f,m): return self.logits_from_x(compiler_pool(f,m))

@torch.no_grad()
def snap_rational_(model:nn.Module,denominator:int):
    for p in model.parameters(): p.copy_(torch.round(p*denominator)/float(denominator))

def rational_grid_certificate(model:nn.Module,denominator:int):
    mx=0.0; n=0
    for p in model.parameters():
        z=p.detach().float()*denominator; mx=max(mx,float((z-z.round()).abs().max())); n+=p.numel()
    return {'denominator':int(denominator),'max_grid_residual':mx,'parameter_count':n,'exact_float_grid':bool(mx==0.0)}

def export_logical_ir(model:RationalReLUCompiler,path:str):
    """Export exact dyadic numerators for the rational-ReLU compiler."""
    den=model.denominator; arrays={}
    for name,p in model.state_dict().items():
        arrays[name.replace('.','__')]=torch.round(p.detach().cpu().float()*den).to(torch.int32).numpy()
    meta={
        'format':'FOG_LOGICAL_IR_V1','input':'concat(mean,last) over selected Qwen layers',
        'activation':'ReLU','denominator':den,'hidden_width':model.width,
        'term_class':'Q[ReLU,x] rational ReLU term / proto-neuron class, affine-ReLU-affine (degree <= 1 in variables)',
        'output_heads':{'opcode':O,'entity_1':E,'relation':R,'entity_2':E},
        'formula_schema':'logit_c(x)=b_c+sum_j v_cj*ReLU(b_j+sum_i w_ji*x_i)',
    }
    arrays['__metadata_utf8__']=np.frombuffer(json.dumps(meta,sort_keys=True).encode('utf-8'),dtype=np.uint8)
    np.savez_compressed(path,**arrays)
    h=hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return meta|{'sha256':h,'path':str(path),'bytes':Path(path).stat().st_size}


def program_rows(programs,per):
    rows=[]
    for pi,p in enumerate(programs):
        for si,ins in enumerate(p.instructions): rows.append((per[pi][si],ins.op,ins.e1,ins.rel,ins.e2))
    return rows

@torch.inference_mode()
def build_compiler_dataset(teacher,features,mask,rows,device,batch=256):
    xs=[]; ys=[]; tls=[[],[],[],[]]
    for st in range(0,len(rows),batch):
        rs=rows[st:st+batch]; ids=torch.tensor([r[0] for r in rs],device=features.device)
        f=features[ids].to(device); m=mask[ids].to(device)
        xs.append(compiler_pool(f,m).cpu())
        ys.append(torch.tensor([[r[1],r[2],r[3],r[4]] for r in rs],dtype=torch.long))
        lgs=teacher.logits(f,m)
        for j,z in enumerate(lgs): tls[j].append(z.float().cpu())
    return torch.cat(xs),torch.cat(ys),tuple(torch.cat(z) for z in tls)

def compiler_supervised_loss(lgs,y):
    loss=F.cross_entropy(lgs[0],y[:,0])
    for j in (1,2,3):
        ids=(y[:,j]>=0).nonzero(as_tuple=False).squeeze(-1)
        if ids.numel(): loss=loss+F.cross_entropy(lgs[j][ids],y[ids,j])
    return loss

def compiler_distill_loss(lgs,teacher_lgs,y,temp=1.5):
    loss=lgs[0].sum()*0.0
    T=float(temp)
    for j in range(4):
        ids=torch.arange(y.size(0),device=y.device) if j==0 else (y[:,j]>=0).nonzero(as_tuple=False).squeeze(-1)
        if ids.numel():
            q=F.softmax(teacher_lgs[j][ids]/T,-1)
            loss=loss+F.kl_div(F.log_softmax(lgs[j][ids]/T,-1),q,reduction='batchmean')*(T*T)
    return loss

def train_logical_compiler(compiler,x,y,teacher_lgs,device,steps=1200,batch_size=192,seed=0,distill_weight=0.15):
    rng=random.Random(seed); compiler.train(); x=x.to(device); y=y.to(device); teacher_lgs=tuple(z.to(device) for z in teacher_lgs)
    groups={o:(y[:,0]==o).nonzero(as_tuple=False).squeeze(-1).tolist() for o in range(O)}
    opt=torch.optim.AdamW(compiler.parameters(),lr=1.5e-3,weight_decay=1e-4); hist=[]
    for step in range(1,steps+1):
        idx=[]
        for _ in range(batch_size):
            o=rng.randrange(O); pool=groups[o] or list(range(len(y))); idx.append(pool[rng.randrange(len(pool))])
        ids=torch.tensor(idx,device=device); lgs=compiler.logits_from_x(x[ids])
        sup=compiler_supervised_loss(lgs,y[ids]); td=compiler_distill_loss(lgs,tuple(z[ids] for z in teacher_lgs),y[ids])
        loss=sup+distill_weight*td
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(compiler.parameters(),1.0); opt.step()
        if step==1 or step%200==0:
            with torch.no_grad():
                pred=lgs[0].argmax(-1); oa=float(pred.eq(y[ids,0]).float().mean())
            rec={'step':step,'loss':float(loss.detach()),'supervised':float(sup.detach()),'distill':float(td.detach()),'opcode_batch_acc':oa}
            hist.append(rec); print('COMPILER',rec,flush=True)
    snap_rational_(compiler,compiler.denominator); return hist

@torch.inference_mode()
def model_decision_agreement(a,b,features,mask,rows,device,batch=256):
    same=total=0; joint_same=joint_total=0
    for st in range(0,len(rows),batch):
        rs=rows[st:st+batch]; ids=torch.tensor([r[0] for r in rs],device=features.device); f=features[ids].to(device); m=mask[ids].to(device)
        aa=[x.argmax(-1) for x in a.logits(f,m)]; bb=[x.argmax(-1) for x in b.logits(f,m)]
        for i,r in enumerate(rs):
            ok=True
            for j,val in enumerate((r[1],r[2],r[3],r[4])):
                if val>=0: same+=int(aa[j][i]==bb[j][i]); total+=1; ok=ok and bool(aa[j][i]==bb[j][i])
            joint_same+=int(ok); joint_total+=1
    return {'field_agreement':same/max(total,1),'instruction_joint_agreement':joint_same/max(joint_total,1)}

@torch.inference_mode()
def compiler_margin_metrics(model,features,mask,rows,device,batch=256):
    vals=[[] for _ in range(4)]
    for st in range(0,len(rows),batch):
        rs=rows[st:st+batch]; ids=torch.tensor([r[0] for r in rs],device=features.device); f=features[ids].to(device); m=mask[ids].to(device); lgs=model.logits(f,m)
        for j,z in enumerate(lgs):
            zz=z.float(); top=torch.topk(zz,2,dim=-1).values; mar=(top[:,0]-top[:,1]).cpu().tolist()
            for i,r in enumerate(rs):
                if j==0 or (r[1+j]>=0): vals[j].append(mar[i])
    out={}
    for name,v in zip(('opcode','e1','relation','e2'),vals):
        if v:
            a=np.asarray(v,float); out[name]={'mean':float(a.mean()),'p10':float(np.quantile(a,.10)),'min':float(a.min())}
        else: out[name]=None
    return out

@dataclass
class FeatureCache:
    def __init__(self, features, mask, device):
        self.features = features.cpu()
        self.mask = mask.cpu()
        self.device = device
    def get(self, ids):
        ids_cpu = torch.as_tensor(ids, device='cpu')
        return self.features[ids_cpu].to(self.device), self.mask[ids_cpu].to(self.device)

def build_program_text_cache(programs,family):
    texts=[]; idx={}; per=[]
    def add(s):
        if s not in idx: idx[s]=len(texts); texts.append(s)
        return idx[s]
    for pi,p in enumerate(programs):
        ids=[]
        for si,ins in enumerate(p.instructions): ids.append(add(instr_text(ins,family,(pi*11+si*5)%8)))
        per.append(ids)
    return texts,per

def instruction_aux_loss(lgs, rows, weight=1.0):
    if weight<=0: return lgs[0].sum()*0
    loss=F.cross_entropy(lgs[0],torch.tensor([r.op for r in rows],device=lgs[0].device))
    for lg,attr in ((lgs[1],'e1'),(lgs[2],'rel'),(lgs[3],'e2')):
        ids=[i for i,r in enumerate(rows) if getattr(r,attr)>=0]
        if ids: loss=loss+F.cross_entropy(lg[ids],torch.tensor([getattr(rows[i],attr) for i in ids],device=lg.device))
    return weight*loss


def st_onehot(logits):
    """Hard forward / soft backward categorical choice."""
    p=torch.softmax(logits.float(),-1)
    h=F.one_hot(p.argmax(-1),p.size(-1)).float()
    return h + p - p.detach()

def oracle_trajectory(program:LogicProgram, device):
    mem=torch.zeros(R,E,E,device=device); cur=F.one_hot(torch.tensor(0,device=device),E).float()
    pred=torch.tensor(0.0,device=device); halted=torch.tensor(0.0,device=device); states=[]
    for ins in program.instructions:
        if halted.item()<0.5:
            if ins.op==OID['BIND']:
                mem=mem.clone(); mem[ins.rel,ins.e1].zero_(); mem[ins.rel,ins.e1,ins.e2]=1.0
            elif ins.op==OID['LOAD']:
                cur=F.one_hot(torch.tensor(ins.e1,device=device),E).float()
            elif ins.op==OID['FOLLOW']:
                row=mem[ins.rel,int(cur.argmax())]
                if row.sum()>0: cur=F.one_hot(row.argmax(),E).float()
            elif ins.op==OID['COMPARE']:
                pred=(cur.argmax()==ins.e1).float()
            elif ins.op==OID['SELECT']:
                idx=ins.e1 if pred.item()>0.5 else ins.e2
                cur=F.one_hot(torch.tensor(idx,device=device),E).float()
            elif ins.op==OID['HALT']:
                halted=torch.tensor(1.0,device=device)
        states.append((cur.clone(),mem.clone(),pred.clone(),halted.clone()))
    return states

def hard_st_execute_one(op_l,e1_l,rel_l,e2_l,program:LogicProgram,trajectory_weight=0.0):
    """Forward trajectory exactly follows argmax instructions; gradients use ST categorical surrogates."""
    device=op_l.device
    cur=F.one_hot(torch.tensor(0,device=device),E).float()
    mem=torch.zeros(R,E,E,device=device)
    pred=torch.tensor(0.0,device=device)
    halted=torch.tensor(0.0,device=device)
    targets=oracle_trajectory(program,device) if trajectory_weight>0 else None
    traj_losses=[]
    for t in range(op_l.size(0)):
        op=st_onehot(op_l[t]); e1=st_onehot(e1_l[t]); rel=st_onehot(rel_l[t]); e2=st_onehot(e2_l[t])
        active=1.0-halted
        oldcur,oldmem,oldpred=cur,mem,pred

        # Follow with hard-executor-compatible fallback: undefined edge leaves current unchanged.
        raw_follow=torch.einsum('r,s,rst->t',rel,oldcur,oldmem)
        mass=raw_follow.sum()
        has_h=(mass.detach()>0.5).float()
        has=has_h + mass - mass.detach()
        follow=raw_follow + (1.0-has)*oldcur

        select=oldpred*e1+(1.0-oldpred)*e2
        pl=active*op[OID['LOAD']]
        pf=active*op[OID['FOLLOW']]
        ps=active*op[OID['SELECT']]
        cur=oldcur*(1-pl-pf-ps)+pl*e1+pf*follow+ps*select

        pc=active*op[OID['COMPARE']]
        pred=oldpred*(1-pc)+pc*(oldcur*e1).sum()

        pb=active*op[OID['BIND']]
        key=rel[:,None]*e1[None,:]
        bind=key[:,:,None]*e2[None,None,:]
        mem=oldmem*(1-pb*key[:,:,None])+pb*bind

        ph=active*op[OID['HALT']]
        halted=halted+ph

        if targets is not None:
            tc,tm,tp,th=targets[t]
            # Sum losses keep errors visible despite sparse typed registers.
            cur_l=(cur-tc).square().sum()
            pred_l=(pred-tp).square()
            halt_l=(halted-th).square()
            denom=tm.sum().detach().clamp_min(1.0)
            mem_l=(mem-tm).square().sum()/denom
            traj_losses.append(0.45*cur_l+0.15*pred_l+0.10*halt_l+0.30*mem_l)

    target=F.one_hot(torch.tensor(program.answer,device=device),E).float()
    final_loss=(cur-target).square().sum()+0.25*(halted-1.0).square()
    traj_loss=torch.stack(traj_losses).mean() if traj_losses else final_loss*0
    exact=int(cur.argmax().item()==program.answer and halted.item()>0.5)
    return final_loss+trajectory_weight*traj_loss,cur,halted,exact

def train_isolated(model,cache,rows,device,steps,seed):
    rng=random.Random(seed); opt=torch.optim.AdamW(model.parameters(),lr=2e-3,weight_decay=1e-4); hist=[]; model.train()
    for step in range(1,steps+1):
        rs=[rows[rng.randrange(len(rows))] for _ in range(96)]; ids=[r[0] for r in rs]; f,m=cache.get(ids); lgs=model.logits(f,m)
        ins=[Instr(r[1],r[2],r[3],r[4]) for r in rs]; loss=instruction_aux_loss(lgs,ins,1.0)+0.02*(model.mix.w()-model.mix.prior).square().sum()
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        if step==1 or step%200==0: hist.append({'stage':'isolated','step':step,'loss':float(loss.detach()),'mix':model.mix.w().detach().cpu().tolist()}); print(hist[-1],flush=True)
    return hist

def train_program_stage_st(model,opt,programs,features,mask,per,device,epochs,aux_start,aux_end,final_weight,traj_start,traj_end,seed,stage):
    rng=random.Random(seed); hist=[]; model.train()
    for ep in range(1,epochs+1):
        order=list(range(len(programs))); rng.shuffle(order); losses=[]; exact=[]
        q=(ep-1)/max(epochs-1,1)
        auxw=aux_start+(aux_end-aux_start)*q
        trajw=traj_start+(traj_end-traj_start)*q
        for bi in range(0,len(order),12):
            pids=order[bi:bi+12]; flat=[]; slices=[]; rows=[]
            for pi in pids:
                st=len(flat); flat.extend(per[pi]); slices.append((st,len(flat))); rows.extend(programs[pi].instructions)
            ids=torch.as_tensor(flat, device='cpu'); f=features[ids].to(device); m=mask[ids].to(device); lgs=model.logits(f,m)
            loss=instruction_aux_loss(lgs,rows,auxw); fl=[]
            for j,pi in enumerate(pids):
                st,en=slices[j]
                lf,cur,halted,ex=hard_st_execute_one(*(x[st:en] for x in lgs),programs[pi],trajectory_weight=trajw)
                fl.append(lf); exact.append(ex)
            if fl: loss=loss+final_weight*torch.stack(fl).mean()
            loss=loss+0.01*(model.mix.w()-model.mix.prior).square().sum()
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); losses.append(float(loss.detach()))
        rec={'stage':stage,'epoch':ep,'loss':sum(losses)/max(len(losses),1),'hard_st_acc':sum(exact)/max(len(exact),1),'aux_weight':auxw,'trajectory_weight':trajw,'final_weight':final_weight}
        hist.append(rec); print(rec,flush=True)
    return hist

def run_st_curriculum(model,train_prog,tr_f,tr_m,tr_per,device,seed,trajectory:bool):
    opt=torch.optim.AdamW(model.parameters(),lr=8e-4,weight_decay=1e-4); hist=[]
    tag='traj' if trajectory else 'no_traj'
    for depth,epochs in ((1,3),(2,3),(4,5)):
        ids=[i for i,p in enumerate(train_prog) if p.chain_len<=depth]
        subset=[train_prog[i] for i in ids]; subper=[tr_per[i] for i in ids]
        ts=0.35 if trajectory else 0.0; te=0.15 if trajectory else 0.0
        hist+=train_program_stage_st(model,opt,subset,tr_f,tr_m,subper,device,epochs,1.0,0.35,0.8,ts,te,seed+depth,f'{tag}_R{depth}')
    hist+=train_program_stage_st(model,opt,train_prog,tr_f,tr_m,tr_per,device,6,0.30,0.0,1.2,0.15 if trajectory else 0.0,0.0,seed+99,f'{tag}_anneal')
    hist+=train_program_stage_st(model,opt,train_prog,tr_f,tr_m,tr_per,device,4,0.0,0.0,1.5,0.0,0.0,seed+199,f'{tag}_final_only')
    return hist

@torch.inference_mode()
def eval_examples(model,cache,rows,device):
    model.eval(); counts={'opcode':[0,0],'e1':[0,0],'rel':[0,0],'e2':[0,0],'joint':[0,0]}; held={'follow_calls':[0,0],'compare_kai':[0,0],'select_pair':[0,0]}
    for st in range(0,len(rows),128):
        rs=rows[st:st+128]; ids=[r[0] for r in rs]; f,m=cache.get(ids); lgs=model.logits(f,m); preds=[x.argmax(-1).cpu().tolist() for x in lgs]
        for i,r in enumerate(rs):
            ok=True
            for key,col,p in [('opcode',1,preds[0][i]),('e1',2,preds[1][i]),('rel',3,preds[2][i]),('e2',4,preds[3][i])]:
                if r[col]>=0: counts[key][1]+=1; hit=(p==r[col]); counts[key][0]+=hit; ok=ok and hit
            counts['joint'][1]+=1; counts['joint'][0]+=ok; kind=None
            if r[1]==OID['FOLLOW'] and r[3]==RID[HELD_FOLLOW_REL]: kind='follow_calls'
            if r[1]==OID['COMPARE'] and r[2]==EID[HELD_COMPARE_ENTITY]: kind='compare_kai'
            if r[1]==OID['SELECT'] and (r[2],r[4])==(EID[HELD_SELECT_PAIR[0]],EID[HELD_SELECT_PAIR[1]]): kind='select_pair'
            if kind: held[kind][1]+=1; held[kind][0]+=ok
    return {k:(v[0]/v[1] if v[1] else None) for k,v in counts.items()}|{'heldout':{k:(v[0]/v[1] if v[1] else None) for k,v in held.items()}}

def semantic_shuffle_mapping(programs):
    """Deterministic same-opcode cyclic derangement. Preserves opcode skeleton, shuffles semantic arguments."""
    flat=[(pi,si) for pi,p in enumerate(programs) for si in range(len(p.instructions))]
    pools={o:[] for o in range(O)}
    for loc in flat:
        pi,si=loc; pools[programs[pi].instructions[si].op].append(loc)
    rotated={}
    for o,locs in pools.items():
        if len(locs)>1:
            srcs=locs[1:]+locs[:1]
        else:
            srcs=locs
        rotated.update({dst:src for dst,src in zip(locs,srcs)})
    return rotated

@torch.inference_mode()
def eval_programs(model,programs,features,mask,per,device,shuffle=False,oracle=False):
    model.eval(); correct=0; held=[0,0]; by={}; instruction_correct=instruction_total=0
    flat=[(pi,si) for pi,p in enumerate(programs) for si in range(len(p.instructions))]; mapping=semantic_shuffle_mapping(programs) if shuffle else {x:x for x in flat}
    for pi,p in enumerate(programs):
        mem=torch.full((R,E),-1,dtype=torch.long); cur=0; pred=False; halted=False
        for si,true_ins in enumerate(p.instructions):
            srcpi,srcsi=mapping[(pi,si)]
            if oracle: ins=programs[srcpi].instructions[srcsi]
            else:
                fid=per[srcpi][srcsi]; f=features[fid:fid+1].to(device); m=mask[fid:fid+1].to(device); lgs=model.logits(f,m); vals=[int(x.argmax(-1)) for x in lgs]; ins=Instr(vals[0],vals[1],vals[2],vals[3])
                if not shuffle:
                    ok=ins.op==true_ins.op and all(getattr(true_ins,a)<0 or getattr(ins,a)==getattr(true_ins,a) for a in ('e1','rel','e2')); instruction_correct+=ok; instruction_total+=1
            if halted: continue
            if ins.op==OID['BIND']: mem[ins.rel,ins.e1]=ins.e2
            elif ins.op==OID['LOAD']: cur=ins.e1
            elif ins.op==OID['FOLLOW']:
                nxt=int(mem[ins.rel,cur]); cur=nxt if nxt>=0 else cur
            elif ins.op==OID['COMPARE']: pred=(cur==ins.e1)
            elif ins.op==OID['SELECT']: cur=ins.e1 if pred else ins.e2
            elif ins.op==OID['HALT']: halted=True
        hit=(cur==p.answer and halted); correct+=hit; by.setdefault(p.chain_len,[0,0]); by[p.chain_len][0]+=hit; by[p.chain_len][1]+=1
        if p.contains_heldout: held[1]+=1; held[0]+=hit
    return {'accuracy':correct/len(programs),'heldout_accuracy':held[0]/max(held[1],1),'instruction_joint':instruction_correct/max(instruction_total,1) if not oracle and not shuffle else None,'by_chain':{str(k):v[0]/v[1] for k,v in by.items()}}

def run(args):
    device=torch.device(args.device if args.device!='auto' else ('cuda' if torch.cuda.is_available() else 'cpu'))
    if device.type == 'cuda':
        try:
            major, minor = torch.cuda.get_device_capability(0)
            print(f"CUDA GPU: {torch.cuda.get_device_name(0)} (sm_{major}{minor})")
            if major < 7:
                print(f"WARNING: GPU sm_{major}{minor} is incompatible with this PyTorch version. Falling back to CPU.")
                device = torch.device('cpu')
        except Exception as e:
            print(f"WARNING: Failed to check GPU capability: {e}. Falling back to CPU.")
            device = torch.device('cpu')
    set_seed(SEED); spec=PhraseSpec(); print('phrase manifest',spec.manifest(),flush=True)
    train_prog=generate_programs(SEED+100,(1,2,3,4),args.train_programs_per_len,train=True); dev_prog=generate_programs(SEED+200,(1,2,3,4),args.dev_programs_per_len,train=True); test_prog=generate_programs(SEED+9000,(1,2,3,4,5,6,8),args.test_programs_per_len,train=False)
    if args.smoke:
        assert all(execute_oracle(p)==p.answer for p in train_prog+dev_prog+test_prog)

        # 1) Oracle shuffle must measurably change program meaning.
        flat=[(pi,si) for pi,p in enumerate(test_prog) for si in range(len(p.instructions))]
        mp=semantic_shuffle_mapping(test_prog); c=0
        for pi,p in enumerate(test_prog):
            mem=torch.full((R,E),-1,dtype=torch.long); cur=0; pred=False; halted=False
            for si,_ in enumerate(p.instructions):
                spi,ssi=mp[(pi,si)]; ins=test_prog[spi].instructions[ssi]
                if halted: continue
                if ins.op==OID['BIND']: mem[ins.rel,ins.e1]=ins.e2
                elif ins.op==OID['LOAD']: cur=ins.e1
                elif ins.op==OID['FOLLOW']:
                    nxt=int(mem[ins.rel,cur]); cur=nxt if nxt>=0 else cur
                elif ins.op==OID['COMPARE']: pred=(cur==ins.e1)
                elif ins.op==OID['SELECT']: cur=ins.e1 if pred else ins.e2
                elif ins.op==OID['HALT']: halted=True
            c+=int(cur==p.answer and halted)
        oracle_shuffle=c/len(test_prog)

        # 2) Hard-ST forward must be exactly identical to the discrete executor under oracle logits.
        p=test_prog[0]; T=len(p.instructions)
        def make_logits(nc, attr):
            x=torch.full((T,nc),-6.0)
            for t,ins in enumerate(p.instructions):
                idx=getattr(ins,attr)
                if idx<0: idx=0
                x[t,idx]=6.0
            return x
        ol=make_logits(O,'op'); e1l=make_logits(E,'e1'); rl=make_logits(R,'rel'); e2l=make_logits(E,'e2')
        _,cur,halted,exact=hard_st_execute_one(ol,e1l,rl,e2l,p,trajectory_weight=0.3)
        assert exact==1 and int(cur.argmax())==p.answer and float(halted)>0.5

        # 3) A wrong hard decision must still produce a finite, nonzero ST gradient.
        ol=ol.clone().requires_grad_(True); e1l=e1l.clone().requires_grad_(True); rl=rl.clone().requires_grad_(True); e2l=e2l.clone().requires_grad_(True)
        load_t=next(i for i,x in enumerate(p.instructions) if x.op==OID['LOAD'])
        true_e=p.instructions[load_t].e1; wrong=(true_e+1)%E
        with torch.no_grad():
            e1l[load_t].fill_(-6.0); e1l[load_t,wrong]=6.0
        loss,_,_,_=hard_st_execute_one(ol,e1l,rl,e2l,p,trajectory_weight=0.5)
        loss.backward()
        grads=[z.grad for z in (ol,e1l,rl,e2l)]
        finite=all(g is not None and torch.isfinite(g).all() for g in grads)
        grad_norm=sum(float(g.abs().sum()) for g in grads if g is not None)
        assert finite and grad_norm>0
        # v6 compiler smoke: forward is already rational-grid ReLU; snapping must not change it.
        cc=RationalReLUCompiler(d=8,k=2,width=7,denominator=32)
        sf=torch.randn(5,2,6,8); sm=torch.ones(5,6,dtype=torch.bool); sm[0,-2:]=False
        before=[z.detach().clone() for z in cc.logits(sf,sm)]
        snap_rational_(cc,32); after=cc.logits(sf,sm)
        compiler_parity=max(float((a-b).detach().abs().max()) for a,b in zip(before,after))
        cert=rational_grid_certificate(cc,32)
        assert compiler_parity==0.0 and cert['exact_float_grid']
        assert oracle_shuffle < 0.35, f'semantic shuffle too weak for locked delta criterion: {oracle_shuffle}'

        return {
            'smoke':True,'oracle_normal':1.0,'oracle_semantic_shuffle':oracle_shuffle,
            'hard_st_oracle_exact':True,'hard_st_wrong_route_loss':float(loss.detach()),
            'hard_st_grad_norm':grad_norm,'finite_gradients':finite,
            'logical_compiler_snap_parity_max_abs':compiler_parity,'logical_compiler_grid':cert,
            'train_programs':len(train_prog),'dev_programs':len(dev_prog),'test_programs':len(test_prog),
            'manifest':spec.manifest()
        }
    ensure_deps(); qwen,tok=load_qwen(args.model,device)
    scan_ids=sorted(set([r[0] for r in spec.train_examples+spec.scan_examples])); pooled=pooled_all_layers(qwen,tok,spec.texts,scan_ids,device,batch=args.qwen_batch); scan=scan_layers(spec,pooled); layers=scan['top3']; print('top3',layers,scan['ranked'][:5],flush=True)
    feats,mask=token_features(qwen,tok,spec.texts,layers,device,batch=args.qwen_batch); d=qwen.config.hidden_size
    op_proto=anchor_proto(feats,mask,spec.op_anchor_ids); e_proto=anchor_proto(feats,mask,spec.entity_anchor_ids); r_proto=anchor_proto(feats,mask,spec.rel_anchor_ids)
    model=LogicFOGV5(d,len(layers),op_proto,e_proto,r_proto).to(device); base_cache=FeatureCache(feats,mask,device)
    isolated_hist=train_isolated(model,base_cache,spec.train_examples,device,args.isolated_steps,args.seed)
    isolated_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}

    # Program text feature caches. TRAIN and DEV are materialized before model selection.
    tr_text,tr_per=build_program_text_cache(train_prog,'train'); dv_text,dv_per=build_program_text_cache(dev_prog,'scan')
    tr_f,tr_m=token_features(qwen,tok,tr_text,layers,device,batch=args.qwen_batch,max_tokens=64)
    dv_f,dv_m=token_features(qwen,tok,dv_text,layers,device,batch=args.qwen_batch,max_tokens=64)

    # A/B on DEV only: same isolated initialization, hard-ST forward for both.
    arms={}
    for arm_name,use_traj in [('hard_st_no_traj',False),('hard_st_trajectory',True)]:
        arm=LogicFOGV5(d,len(layers),op_proto,e_proto,r_proto).to(device); arm.load_state_dict(isolated_state)
        ah=run_st_curriculum(arm,train_prog,tr_f,tr_m,tr_per,device,args.seed+(1000 if use_traj else 0),use_traj)
        tr_eval=eval_programs(arm,train_prog,tr_f,tr_m,tr_per,device)
        dv_eval=eval_programs(arm,dev_prog,dv_f,dv_m,dv_per,device)
        score=float(dv_eval['accuracy'])+0.25*float(dv_eval['instruction_joint'] or 0.0)
        arms[arm_name]={'model':arm,'history':ah,'train_eval':tr_eval,'dev_eval':dv_eval,'dev_score':score}
        print('DEV ARM',arm_name,{'score':score,'train':tr_eval,'dev':dv_eval},flush=True)

    chosen_name=max(arms,key=lambda n:(arms[n]['dev_score'], n=='hard_st_trajectory'))
    teacher=arms[chosen_name]['model']; print('CHOSEN V5 ARM',chosen_name,flush=True)
    hist=isolated_hist+arms[chosen_name]['history']
    teacher_scan_metrics=eval_examples(teacher,base_cache,spec.scan_examples,device)

    # v6: compile TRAIN-only evidence into a rational-ReLU semantic head.
    base_x,base_y,base_t=build_compiler_dataset(teacher,feats,mask,spec.train_examples,device)
    tr_rows=program_rows(train_prog,tr_per)
    prog_x,prog_y,prog_t=build_compiler_dataset(teacher,tr_f,tr_m,tr_rows,device)
    cx=torch.cat([base_x,prog_x],0); cy=torch.cat([base_y,prog_y],0)
    ct=tuple(torch.cat([base_t[j],prog_t[j]],0) for j in range(4))
    compiler=RationalReLUCompiler(d,len(layers),width=args.compiler_width,denominator=args.compiler_denominator).to(device)
    compiler_hist=train_logical_compiler(compiler,cx,cy,ct,device,steps=args.compiler_steps,batch_size=args.compiler_batch,seed=args.seed+6000,distill_weight=args.compiler_distill)
    compiler_grid=rational_grid_certificate(compiler,args.compiler_denominator)
    logical_ir=export_logical_ir(compiler,args.logical_ir)

    compiler_scan_metrics=eval_examples(compiler,base_cache,spec.scan_examples,device)
    compiler_train_eval=eval_programs(compiler,train_prog,tr_f,tr_m,tr_per,device)
    compiler_dev_eval=eval_programs(compiler,dev_prog,dv_f,dv_m,dv_per,device)
    compiler_dev_score=float(compiler_dev_eval['accuracy'])+0.25*float(compiler_dev_eval['instruction_joint'] or 0.0)
    dev_rows=program_rows(dev_prog,dv_per)
    compiler_agreement=model_decision_agreement(teacher,compiler,dv_f,dv_m,dev_rows,device)
    compiler_margins=compiler_margin_metrics(compiler,dv_f,dv_m,dev_rows,device)
    teacher_score=arms[chosen_name]['dev_score']
    min_train=max(0.90,float(arms[chosen_name]['train_eval']['accuracy'])-0.02)
    compiler_accepted=bool(compiler_grid['exact_float_grid'] and compiler_train_eval['accuracy']>=min_train and compiler_dev_score>=teacher_score-0.01 and compiler_agreement['field_agreement']>=0.80)
    execution_model=compiler if compiler_accepted else teacher
    execution_name='rational_relu_compiler' if compiler_accepted else 'v5_teacher_abstain'
    print('COMPILER DECISION',{'accepted':compiler_accepted,'execution':execution_name,'teacher_dev_score':teacher_score,'compiler_dev_score':compiler_dev_score,'agreement':compiler_agreement,'grid':compiler_grid},flush=True)

    # No TEST metric participates in the teacher-arm or compiler acceptance decision.
    scan_metrics=compiler_scan_metrics if compiler_accepted else teacher_scan_metrics
    test_phrase_metrics=eval_examples(execution_model,base_cache,spec.test_examples,device)

    # Locked program TEST encoded only after all DEV-only selections are frozen.
    te_text,te_per=build_program_text_cache(test_prog,'test')
    te_f,te_m=token_features(qwen,tok,te_text,layers,device,batch=args.qwen_batch,max_tokens=64)
    normal=eval_programs(execution_model,test_prog,te_f,te_m,te_per,device)
    shuffled=eval_programs(execution_model,test_prog,te_f,te_m,te_per,device,shuffle=True)
    oracle=eval_programs(execution_model,test_prog,te_f,te_m,te_per,device,oracle=True)
    print('locked normal',normal,flush=True); print('locked shuffled',shuffled,flush=True); print('locked oracle',oracle,flush=True)

    drop=normal['accuracy']-shuffled['accuracy']
    selected_train_eval=compiler_train_eval if compiler_accepted else arms[chosen_name]['train_eval']
    if normal['heldout_accuracy']>=0.75 and normal['accuracy']>=0.8 and drop>=0.35:
        verdict='PASS_LOGIC_V6'
    elif selected_train_eval['accuracy']<0.95:
        verdict='HARD_TRAJECTORY_OR_COMPILER_TRAINING_FAILURE'
    elif normal['instruction_joint']<0.9:
        verdict='SEMANTIC_INTERFACE_LIMITED'
    else:
        verdict='PROGRAM_TRANSFER_LIMITED'
    out={
        'protocol':'FOG_LOGIC_V6_RATIONAL_RELU_LOGICAL_COMPILER',
        'model':args.model,'frozen_backbone':True,'selected_layers':layers,'layer_scan':scan,
        'phrase_manifest':spec.manifest(),
        'holdouts':{'follow_relation':HELD_FOLLOW_REL,'compare_entity':HELD_COMPARE_ENTITY,'select_pair':HELD_SELECT_PAIR},
        'train_program_depths':[1,2,3,4],'test_program_depths':[1,2,3,4,5,6,8],
        'training_forward':'hard_argmax_straight_through',
        'semantic_compiler_class':'rational_Q[ReLU,x]_degree1_head_over_frozen_Qwen_latents',
        'logical_claim_boundary':'exact rational-ReLU term IR for the compiler head; no claim that v5 cosine/logsumexp head or frozen Qwen itself is an RPL formula',
        'dev_arm_selection_only':True,'chosen_v5_arm':chosen_name,
        'dev_ablations':{n:{k:v for k,v in a.items() if k!='model'} for n,a in arms.items()},
        'compiler':{
            'accepted':compiler_accepted,'execution_path':execution_name,'history':compiler_hist,'grid_certificate':compiler_grid,'logical_ir':logical_ir,
            'teacher_scan_metrics':teacher_scan_metrics,'compiler_scan_metrics':compiler_scan_metrics,
            'teacher_train_eval':arms[chosen_name]['train_eval'],'teacher_dev_eval':arms[chosen_name]['dev_eval'],'teacher_dev_score':teacher_score,
            'compiler_train_eval':compiler_train_eval,'compiler_dev_eval':compiler_dev_eval,'compiler_dev_score':compiler_dev_score,
            'teacher_compiler_dev_agreement':compiler_agreement,'compiler_dev_margins':compiler_margins,
            'acceptance_gate':{'min_train_accuracy':min_train,'max_dev_score_drop':0.01,'min_field_agreement':0.80}
        },
        'train_history':hist,'scan_phrase_metrics':scan_metrics,'locked_phrase_metrics':test_phrase_metrics,
        'train_program_eval':selected_train_eval,'dev_program_eval':compiler_dev_eval if compiler_accepted else arms[chosen_name]['dev_eval'],
        'locked_program_eval':normal,'shuffle_semantics':shuffled,'oracle_program_eval':oracle,
        'selected_trainable_params':sum(p.numel() for p in execution_model.parameters() if p.requires_grad),
        'program_counts':{'train':len(train_prog),'dev':len(dev_prog),'test':len(test_prog)},
        'verdict':verdict,
        'runtime_env':{'torch':torch.__version__,'device':str(device),'platform':platform.platform()}
    }
    torch.save({'teacher_state_dict':teacher.state_dict(),'compiler_state_dict':compiler.state_dict(),'compiler_accepted':compiler_accepted,'selected_layers':layers,'phrase_manifest':spec.manifest(),'protocol':out['protocol']},args.checkpoint); Path(args.output).write_text(json.dumps(out,indent=2),encoding='utf-8'); return out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--model',default='Qwen/Qwen2.5-0.5B-Instruct'); ap.add_argument('--device',default='auto'); ap.add_argument('--qwen-batch',type=int,default=24); ap.add_argument('--isolated-steps',type=int,default=1200); ap.add_argument('--train-programs-per-len',type=int,default=60); ap.add_argument('--dev-programs-per-len',type=int,default=20); ap.add_argument('--test-programs-per-len',type=int,default=60); ap.add_argument('--seed',type=int,default=0); ap.add_argument('--compiler-width',type=int,default=128); ap.add_argument('--compiler-denominator',type=int,default=1024); ap.add_argument('--compiler-steps',type=int,default=1200); ap.add_argument('--compiler-batch',type=int,default=192); ap.add_argument('--compiler-distill',type=float,default=0.15); ap.add_argument('--logical-ir',default='fog_logic_v6_logical_ir.npz'); ap.add_argument('--output',default='fog_logic_v6_results.json'); ap.add_argument('--checkpoint',default='fog_logic_v6.pt'); ap.add_argument('--smoke',action='store_true'); args=ap.parse_args(); t=time.time(); r=run(args); print(json.dumps({'summary':{k:r.get(k) for k in ('smoke','dev_program_eval','locked_program_eval','shuffle_semantics','oracle_program_eval','verdict') if k in r},'runtime_seconds':time.time()-t},indent=2))
if __name__=='__main__': main()
