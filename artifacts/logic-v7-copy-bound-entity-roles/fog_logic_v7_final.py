import subprocess, sys
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
try:
    import transformers
except ImportError:
    install("transformers")
try:
    import accelerate
except ImportError:
    install("accelerate")

import argparse, json, math, os, random, subprocess, sys, time, hashlib, platform, copy
FORCE_CPU = True
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
    "Write down that {r} maps {a} to {b}.", "In memory connect {a} to {b} using {r}.",
    "Create the fact {a} --{r}--> {b}.", "Register {b} as the {r} target of {a}.",
),
"LOAD":(
    "Focus on {a}.","Set the current entity to {a}.","Load {a} into the entity register.","Begin from {a}.",
    "Make {a} our current subject.","Place the cursor on {a}.","Use {a} as the active node.","Start reasoning from {a}.",
),
"FOLLOW":(
    "Follow the {r} relation from the current entity.", "Move along {r} from the current entity.",
    "Replace the current entity by its {r} target.", "Traverse the active entity's {r} edge.",
    "Resolve the current node through {r}.", "Advance one hop using relation {r}.",
    "Go where the current entity points by {r}.", "Use {r} to reach the next entity.",
),
"COMPARE":(
    "Check whether the current entity is {a}.", "Compare the current entity with {a}.",
    "Test current-entity equality against {a}.", "Ask if the active entity equals {a}.",
    "Determine whether we are at {a}.", "Check current equals {a}.",
    "Evaluate whether the active node is {a}.", "Set the predicate by testing against {a}.",
),
"SELECT":(
    "If the comparison is true choose {a}; otherwise choose {b}.", "Select {a} on true and {b} on false.",
    "Use {a} if the predicate holds, else use {b}.", "Truth selects {a}; falsehood selects {b}.",
    "Choose between {a} and {b}: true takes {a}.", "On a true predicate load {a}; otherwise load {b}.",
    "Branch-select {a} for yes and {b} for no.", "Let the predicate choose {a} versus {b}.",
),
"HALT":(
    "Stop execution and return the current entity.","Halt now with the current entity.","Finish the program without changing the current entity.","End here and keep the active entity.",
    "Terminate and preserve the current node.","Return the entity now held and stop.","No more operations; keep the current entity.","Conclude execution with the active subject.",
),
}
SCAN_TEMPLATES={
"BIND":("Memorize a {r} edge leading from {a} to {b}.","Associate {a} with {b} through relation {r}.","Save {b} as {a}'s {r} destination.","Add a {r} connection from {a} to {b}."),
"LOAD":("Make {a} the active entity.","Begin tracking {a}.","Move the cursor to {a}.","Treat {a} as current."),
"FOLLOW":("Traverse the current entity's {r} link.","Jump through relation {r}.","Take one {r} hop from here.","Resolve this node via {r}."),
"COMPARE":("Ask whether we are currently at {a}.","Evaluate current equals {a}.","Test if this entity is {a}.","Compare our active node to {a}."),
"SELECT":("Choose {a} when true; choose {b} when false.","True means {a}, false means {b}.","Predicate true selects {a}, otherwise {b}.","Use {a} for yes and {b} for no."),
"HALT":("End computation here.","Freeze the current entity and stop.","Return the current node and terminate.","Finish now without another state change."),
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
"SELECT":("select branch","conditional choice","pick entity"),"HALT":("stop execution","halt program","return answer"),
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
    for cl in chain_lens:
        targets=(list(range(E))*((per_len+E-1)//E))[:per_len]
        rng.shuffle(targets)
        for local_i,target_answer in enumerate(targets):
            built=False
            for attempt in range(3000):
                force_held=(not train and local_i%2==0)
                hold_mode='none'
                if force_held:
                    if target_answer in (EID[HELD_SELECT_PAIR[0]],EID[HELD_SELECT_PAIR[1]]) and local_i%6==0:
                        hold_mode='select'
                    elif local_i%4==0:
                        hold_mode='compare'
                    else:
                        hold_mode='follow'

                rels=[rng.choice(allowed_follow) for _ in range(cl)]
                if hold_mode=='follow': rels[rng.randrange(cl)]=RID[HELD_FOLLOW_REL]
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

                desired_true=((local_i+cl)%2==0)
                if hold_mode=='compare':
                    desired_true=((local_i//2)%2==0)
                    if desired_true:
                        chain[-1]=EID[HELD_COMPARE_ENTITY]
                        facts=[]
                        protected={(rels[i],chain[i]) for i in range(cl)}
                        if len(protected)!=cl: continue
                        facts=[Instr(OID['BIND'],chain[i],rels[i],chain[i+1]) for i in range(cl)]
                    compare_ent=EID[HELD_COMPARE_ENTITY]
                    compare_true=(chain[-1]==compare_ent)
                else:
                    compare_true=desired_true
                    if compare_true:
                        compare_ent=chain[-1]
                    else:
                        choices=[x for x in range(E) if x!=chain[-1] and (not train or x!=EID[HELD_COMPARE_ENTITY])]
                        compare_ent=rng.choice(choices)
                    if train and compare_ent==EID[HELD_COMPARE_ENTITY]: continue

                if hold_mode=='select':
                    t,f=EID[HELD_SELECT_PAIR[0]],EID[HELD_SELECT_PAIR[1]]
                    compare_true=(target_answer==t)
                    if compare_true:
                        compare_ent=chain[-1]
                    else:
                        choices=[x for x in range(E) if x!=chain[-1] and x!=EID[HELD_COMPARE_ENTITY]]
                        compare_ent=rng.choice(choices)
                else:
                    if compare_true:
                        t=target_answer; f=rng.choice([x for x in range(E) if x!=target_answer])
                    else:
                        f=target_answer; t=rng.choice([x for x in range(E) if x!=target_answer])
                    if train and (t,f)==(EID[HELD_SELECT_PAIR[0]],EID[HELD_SELECT_PAIR[1]]): continue

                ans=t if compare_true else f
                if ans!=target_answer: continue
                instr=tuple(facts+[Instr(OID['LOAD'],chain[0])]+[Instr(OID['FOLLOW'],rel=r) for r in rels]+[Instr(OID['COMPARE'],compare_ent),Instr(OID['SELECT'],t,e2=f),Instr(OID['HALT'])])
                instr=instr+(Instr(OID['LOAD'],rng.randrange(E)),Instr(OID['FOLLOW'],rel=rng.randrange(R)))
                key=tuple((x.op,x.e1,x.rel,x.e2) for x in instr)
                if key in seen: continue
                seen.add(key)
                held=(RID[HELD_FOLLOW_REL] in rels or compare_ent==EID[HELD_COMPARE_ENTITY] or (t,f)==(EID[HELD_SELECT_PAIR[0]],EID[HELD_SELECT_PAIR[1]]))
                p=LogicProgram(instr,ans,cl,held)
                if execute_oracle(p)!=ans: continue
                out.append(p); built=True; break
            if not built: raise RuntimeError(f'could not build balanced program cl={cl} answer={target_answer}')
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
        for family,nvar in [('train',8),('scan',4),('test',4)]:
            for rel in range(R):
                for _ in range(32 if family=='train' else 14):
                    a,b=rng.sample(range(E),2)
                    for v in range(nvar): self.add_example(family,Instr(OID['BIND'],a,rel,b),v)
            for a in range(E):
                for v in range(nvar):
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
    tasks={'opcode':metric(1,O),'relation':metric(3,R)}; rows=[]
    for l in range(pooled.size(1)):
        vals={}
        for name,(tid,ty,eid,ey,nc) in tasks.items():
            vals[name]=centroid_acc(pooled[tid,l],ty,pooled[eid,l],ey,nc) if len(eid) else 1.0
        score=.60*vals['opcode']+.40*vals['relation']
        rows.append({'layer':l,'score':score,**vals})
    ranked=sorted(rows,key=lambda x:(-x['score'],x['layer']))
    head_top={name:[x['layer'] for x in sorted(rows,key=lambda z:(-z[name],z['layer']))[:2]] for name in tasks}
    role_support=[0,1,22,23]
    selected=sorted(set(head_top['opcode']+head_top['relation']+role_support))
    return {'selected':selected,'head_top':head_top,'role_support':role_support,'ranked':ranked}

@torch.inference_mode()
def token_features(model,tok,texts,layers,device,batch=24,max_tokens=64,max_mentions=4):
    d=model.config.hidden_size
    feat=torch.zeros(len(texts),len(layers),max_tokens,d,dtype=torch.float16)
    mask=torch.zeros(len(texts),max_tokens,dtype=torch.bool)
    mention_ids=torch.full((len(texts),max_mentions),-1,dtype=torch.long)
    mention_tokmask=torch.zeros(len(texts),max_mentions,max_tokens,dtype=torch.bool)
    for st in range(0,len(texts),batch):
        rows=texts[st:st+batch]
        enc=tok(rows,padding=True,truncation=True,max_length=max_tokens,return_tensors='pt',add_special_tokens=False,return_offsets_mapping=True)
        offsets=enc.pop('offset_mapping').clone(); enc={k:v.to(device) for k,v in enc.items()}
        res=model.model(**enc,use_cache=False,return_dict=True,output_hidden_states=True); am=enc['attention_mask'].bool()
        for i,text in enumerate(rows):
            n=int(am[i].sum()); mask[st+i,:n]=True
            for j,l in enumerate(layers): feat[st+i,j,:n]=res.hidden_states[l][i,:n].detach().cpu().half()
            found=[]
            for eid,name in enumerate(ENTITIES):
                pos=text.find(name)
                if pos<0: continue
                a_char,b_char=pos,pos+len(name); toks=[]
                for ti,(aa,bb) in enumerate(offsets[i,:n].tolist()):
                    if bb>a_char and aa<b_char: toks.append(ti)
                if toks: found.append((min(toks),max(toks)+1,eid))
            found=sorted(found,key=lambda z:z[0])[:max_mentions]
            for mi,(a,b,eid) in enumerate(found):
                mention_ids[st+i,mi]=eid; mention_tokmask[st+i,mi,a:b]=True
    return feat,mask,mention_ids,mention_tokmask

class HeadLayerMixer(nn.Module):
    def __init__(self,layer_ids,preferred_layer):
        super().__init__(); self.layer_ids=tuple(layer_ids); x=torch.zeros(len(layer_ids)); x[layer_ids.index(preferred_layer)]=1.5
        self.logits=nn.Parameter(x); self.register_buffer('prior',torch.softmax(x,0))
    def w(self): return torch.softmax(self.logits.float(),0)
    def ctx(self,x): return torch.einsum('k,bktd->btd',self.w().to(x),x)
    def proto(self,x): return torch.einsum('k,...kd->...d',self.w().to(x),x)
    def vec(self,x): return torch.einsum('k,...kd->...d',self.w().to(x),x)

class PrototypeBinder(nn.Module):
    def __init__(self,d,proto,rank=128):
        super().__init__(); self.register_buffer('proto_layers',proto.float()); self.q=nn.Linear(d,rank,bias=False); self.k=nn.Linear(d,rank,bias=False); self.scale=nn.Parameter(torch.tensor(math.log(10.0))); self.k.weight.data.copy_(self.q.weight.data)
    def token_class_score(self,ctx,mix):
        p=mix.proto(self.proto_layers); q=F.normalize(self.q(ctx).float(),dim=-1); k=F.normalize(self.k(p).float(),dim=-1); return self.scale.exp().clamp(max=100)*torch.einsum('btr,car->btca',q,k)
    def forward(self,ctx,mask,mix):
        s=self.token_class_score(ctx,mix); s=s.masked_fill(~mask[:,:,None,None],float('-inf')); return torch.logsumexp(s,dim=(1,3))

class SentencePrototypeBinder(nn.Module):
    def __init__(self,d2,proto,rank=128):
        super().__init__(); self.register_buffer('proto_layers',proto.float()); self.q=nn.Linear(d2,rank,bias=False); self.k=nn.Linear(d2,rank,bias=False); self.scale=nn.Parameter(torch.tensor(math.log(10.0))); self.k.weight.data.copy_(self.q.weight.data)
    @staticmethod
    def per_layer_repr(f,mask):
        mm=mask[:,None,:,None].float(); den=mm.sum(2).clamp_min(1.0); mean=(f.float()*mm).sum(2)/den
        pos=torch.arange(mask.size(1),device=mask.device)[None]; li=pos.masked_fill(~mask,-1).max(1).values; rr=torch.arange(mask.size(0),device=mask.device)
        last=f[rr[:,None],torch.arange(f.size(1),device=f.device)[None,:],li[:,None]].float()
        return torch.cat([mean,last],-1)
    def forward(self,f,mask,mix):
        x=mix.vec(self.per_layer_repr(f,mask)); p=mix.vec(self.proto_layers)
        q=F.normalize(self.q(x).float(),dim=-1); k=F.normalize(self.k(p).float(),dim=-1)
        return self.scale.exp().clamp(max=100)*torch.einsum('br,car->bca',q,k).logsumexp(-1)

class MentionRoleBinder(nn.Module):
    def __init__(self,d,rank=128):
        super().__init__(); self.q=nn.Linear(2*d,rank,bias=False); self.k=nn.Linear(d,rank,bias=False); self.scale=nn.Parameter(torch.tensor(math.log(10.0)))
    def forward(self,f,mask,mix,mention_ids,mention_tokmask):
        per=SentencePrototypeBinder.per_layer_repr(f,mask); sent=mix.vec(per)
        ctx=mix.ctx(f.float())
        mm=mention_tokmask.float(); den=mm.sum(-1,keepdim=True).clamp_min(1.0)
        mv=torch.einsum('bmt,btd->bmd',mm,ctx)/den
        q=F.normalize(self.q(sent).float(),dim=-1); k=F.normalize(self.k(mv).float(),dim=-1)
        sc=self.scale.exp().clamp(max=100)*torch.einsum('br,bmr->bm',q,k)
        valid=mention_ids.ge(0); sc=sc.masked_fill(~valid,-1e4)
        oh=F.one_hot(mention_ids.clamp_min(0),E).float()
        z=sc[:,:,None]+torch.where(oh.bool(),torch.zeros_like(oh,device=sc.device),torch.full_like(oh,-1e4,device=sc.device))
        glob=torch.logsumexp(z,dim=1)
        anym=valid.any(1,keepdim=True)
        return torch.where(anym,glob,torch.zeros_like(glob))

def anchor_proto(features,mask,nested):
    rows=[]
    for ids in nested:
        ar=[]
        for pid in ids:
            f=features[pid].float(); m=mask[pid].float()[None,:,None]; ar.append((f*m).sum(1)/m.sum(1).clamp_min(1))
        rows.append(torch.stack(ar))
    return torch.stack(rows)

def anchor_sentence_proto(features,mask,nested):
    rows=[]
    for ids in nested:
        ar=[]
        for pid in ids:
            f=features[pid:pid+1].float(); m=mask[pid:pid+1]
            ar.append(SentencePrototypeBinder.per_layer_repr(f,m)[0])
        rows.append(torch.stack(ar))
    return torch.stack(rows)

class LogicFOGV7(nn.Module):
    def __init__(self,d,layer_ids,head_top,op_proto,r_proto):
        super().__init__(); self.layer_ids=tuple(layer_ids)
        self.op_mix=HeadLayerMixer(layer_ids,head_top['opcode'][0]); self.rel_mix=HeadLayerMixer(layer_ids,head_top['relation'][0])
        self.e1_mix=HeadLayerMixer(layer_ids,layer_ids[0]); self.e2_mix=HeadLayerMixer(layer_ids,layer_ids[0])
        with torch.no_grad(): self.e1_mix.logits.zero_(); self.e2_mix.logits.zero_(); self.e1_mix.prior.fill_(1.0/len(layer_ids)); self.e2_mix.prior.fill_(1.0/len(layer_ids))
        self.op=SentencePrototypeBinder(2*d,op_proto,128); self.rel=PrototypeBinder(d,r_proto,96)
        self.e1_role=MentionRoleBinder(d,128); self.e2_role=MentionRoleBinder(d,128)
    def logits(self,f,m,mention_ids,mention_tokmask):
        op=self.op(f,m,self.op_mix)
        rel=self.rel(self.rel_mix.ctx(f.float()),m,self.rel_mix)
        e1=self.e1_role(f,m,self.e1_mix,mention_ids,mention_tokmask)
        e2=self.e2_role(f,m,self.e2_mix,mention_ids,mention_tokmask)
        return op,e1,rel,e2
    def mix_reg(self):
        xs=(self.op_mix,self.e1_mix,self.rel_mix,self.e2_mix)
        return sum((x.w()-x.prior).square().sum() for x in xs)
    def mix_report(self):
        return {n:getattr(self,n+'_mix').w().detach().cpu().tolist() for n in ('op','e1','rel','e2')}

@dataclass
class FeatureCache:
    features:torch.Tensor; mask:torch.Tensor; mention_ids:torch.Tensor; mention_tokmask:torch.Tensor
    def get(self,ids): return self.features[ids],self.mask[ids],self.mention_ids[ids],self.mention_tokmask[ids]

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
        raw_follow=torch.einsum('r,s,rst->t',rel,oldcur,oldmem)
        mass=raw_follow.sum()
        has_h=(mass.detach()>0.5).float()
        has=has_h + mass - mass.detach()
        follow=raw_follow + (1.0-has)*oldcur
        select=oldpred*e1+(1.0-oldpred)*e2
        pl=active*op[OID['LOAD']]; pf=active*op[OID['FOLLOW']]; ps=active*op[OID['SELECT']]
        cur=oldcur*(1-pl-pf-ps)+pl*e1+pf*follow+ps*select
        pc=active*op[OID['COMPARE']]; pred=oldpred*(1-pc)+pc*(oldcur*e1).sum()
        pb=active*op[OID['BIND']]; key=rel[:,None]*e1[None,:]; bind=key[:,:,None]*e2[None,None,:]
        mem=oldmem*(1-pb*key[:,:,None])+pb*bind
        ph=active*op[OID['HALT']]; halted=halted+ph
        if targets is not None:
            tc,tm,tp,th=targets[t]
            cur_l=(cur-tc).square().sum(); pred_l=(pred-tp).square(); halt_l=(halted-th).square()
            denom=tm.sum().detach().clamp_min(1.0); mem_l=(mem-tm).square().sum()/denom
            traj_losses.append(0.45*cur_l+0.15*pred_l+0.10*halt_l+0.30*mem_l)
    target=F.one_hot(torch.tensor(program.answer,device=device),E).float()
    final_loss=(cur-target).square().sum()+0.25*(halted-1.0).square()
    traj_loss=torch.stack(traj_losses).mean() if traj_losses else final_loss*0
    exact=int(cur.argmax().item()==program.answer and halted.item()>0.5)
    return final_loss+trajectory_weight*traj_loss,cur,halted,exact

def train_isolated(model,cache,rows,device,steps,seed):
    rng=random.Random(seed); opt=torch.optim.AdamW(model.parameters(),lr=2e-3,weight_decay=1e-4); hist=[]; model.train()
    for step in range(1,steps+1):
        rs=[rows[rng.randrange(len(rows))] for _ in range(96)]; ids=torch.tensor([r[0] for r in rs],device=device); f,m,mi,mm=cache.get(ids); lgs=model.logits(f,m,mi,mm)
        ins=[Instr(r[1],r[2],r[3],r[4]) for r in rs]; loss=instruction_aux_loss(lgs,ins,1.0)+0.02*model.mix_reg()
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        if step==1 or step%200==0: hist.append({'stage':'isolated','step':step,'loss':float(loss.detach()),'mix':model.mix_report()}); print(hist[-1],flush=True)
    return hist

def train_program_stage_st(model,opt,programs,features,mask,mention_ids,mention_tokmask,per,device,epochs,aux_start,aux_end,final_weight,traj_start,traj_end,seed,stage):
    rng=random.Random(seed); hist=[]; model.train()
    for ep in range(1,epochs+1):
        order=list(range(len(programs))); rng.shuffle(order); losses=[]; exact=[]
        q=(ep-1)/max(epochs-1,1); auxw=aux_start+(aux_end-aux_start)*q; trajw=traj_start+(traj_end-traj_start)*q
        for bi in range(0,len(order),12):
            pids=order[bi:bi+12]; flat=[]; slices=[]; rows=[]
            for pi in pids:
                st=len(flat); flat.extend(per[pi]); slices.append((st,len(flat))); rows.extend(programs[pi].instructions)
            ids=torch.tensor(flat,device=device); f=features[ids].to(device); m=mask[ids].to(device); mi=mention_ids[ids].to(device); mm=mention_tokmask[ids].to(device); lgs=model.logits(f,m,mi,mm)
            loss=instruction_aux_loss(lgs,rows,auxw); fl=[]
            for j,pi in enumerate(pids):
                st,en=slices[j]; lf,cur,halted,ex=hard_st_execute_one(*(x[st:en] for x in lgs),programs[pi],trajectory_weight=trajw)
                fl.append(lf); exact.append(ex)
            if fl: loss=loss+final_weight*torch.stack(fl).mean()
            loss=loss+0.01*model.mix_reg()
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); losses.append(float(loss.detach()))
        rec={'stage':stage,'epoch':ep,'loss':sum(losses)/max(len(losses),1),'hard_st_acc':sum(exact)/max(len(exact),1),'aux_weight':auxw,'trajectory_weight':trajw,'final_weight':final_weight}
        hist.append(rec); print(rec,flush=True)
    return hist

def run_st_curriculum(model,train_prog,tr_f,tr_m,tr_mi,tr_mm,tr_per,device,seed):
    opt=torch.optim.AdamW(model.parameters(),lr=7e-4,weight_decay=1e-4); hist=[]
    for depth,epochs in ((1,4),(2,4),(4,6)):
        ids=[i for i,p in enumerate(train_prog) if p.chain_len<=depth]
        subset=[train_prog[i] for i in ids]; subper=[tr_per[i] for i in ids]
        hist+=train_program_stage_st(model,opt,subset,tr_f,tr_m,tr_mi,tr_mm,subper,device,epochs,1.0,0.40,0.8,0.0,0.0,seed+depth,f'R{depth}')
    hist+=train_program_stage_st(model,opt,train_prog,tr_f,tr_m,tr_mi,tr_mm,tr_per,device,8,0.35,0.0,1.2,0.0,0.0,seed+99,'semantic_anneal')
    hist+=train_program_stage_st(model,opt,train_prog,tr_f,tr_m,tr_mi,tr_mm,tr_per,device,5,0.0,0.0,1.5,0.0,0.0,seed+199,'final_only')
    return hist

@torch.inference_mode()
def eval_examples(model,cache,rows,device):
    model.eval(); counts={'opcode':[0,0],'e1':[0,0],'rel':[0,0],'e2':[0,0],'joint':[0,0]}; held={k:{'joint':[0,0],'opcode':[0,0],'e1':[0,0],'rel':[0,0],'e2':[0,0]} for k in ('follow_calls','compare_kai','select_pair')}; byop={x:{'joint':[0,0],'opcode':[0,0],'e1':[0,0],'rel':[0,0],'e2':[0,0]} for x in OPCODES}
    for st in range(0,len(rows),128):
        rs=rows[st:st+128]; ids=torch.tensor([r[0] for r in rs],device=device); f,m,mi,mm=cache.get(ids); lgs=model.logits(f,m,mi,mm); preds=[x.argmax(-1).cpu().tolist() for x in lgs]
        for i,r in enumerate(rs):
            ok=True; field_hits={}
            for key,col,pv in [('opcode',1,preds[0][i]),('e1',2,preds[1][i]),('rel',3,preds[2][i]),('e2',4,preds[3][i])]:
                if r[col]>=0:
                    hit=(pv==r[col]); counts[key][1]+=1; counts[key][0]+=hit; ok=ok and hit; field_hits[key]=hit
                else: field_hits[key]=None
            counts['joint'][1]+=1; counts['joint'][0]+=ok
            opn=OPCODES[r[1]]; byop[opn]['joint'][1]+=1; byop[opn]['joint'][0]+=ok
            for key in ('opcode','e1','rel','e2'):
                if field_hits[key] is not None: byop[opn][key][1]+=1; byop[opn][key][0]+=field_hits[key]
            kind=None
            if r[1]==OID['FOLLOW'] and r[3]==RID[HELD_FOLLOW_REL]: kind='follow_calls'
            if r[1]==OID['COMPARE'] and r[2]==EID[HELD_COMPARE_ENTITY]: kind='compare_kai'
            if r[1]==OID['SELECT'] and (r[2],r[4])==(EID[HELD_SELECT_PAIR[0]],EID[HELD_SELECT_PAIR[1]]): kind='select_pair'
            if kind:
                held[kind]['joint'][1]+=1; held[kind]['joint'][0]+=ok
                for key in ('opcode','e1','rel','e2'):
                    if field_hits[key] is not None: held[kind][key][1]+=1; held[kind][key][0]+=field_hits[key]
    def cv(v): return v[0]/v[1] if v[1] else None
    return {k:cv(v) for k,v in counts.items()}|{'heldout':{k:{f:cv(v) for f,v in z.items()} for k,z in held.items()},'by_opcode':{k:{f:cv(v) for f,v in z.items()} for k,z in byop.items()}}

def _deranged_mapping(programs,mode):
    flat=[(pi,si) for pi,p in enumerate(programs) for si in range(len(p.instructions))]
    if not mode: return {x:x for x in flat}
    rng=random.Random(SEED+7007)
    if mode=='full':
        perm=flat[:]; rng.shuffle(perm)
        if any(a==b for a,b in zip(flat,perm)): perm=perm[1:]+perm[:1]
        return {flat[i]:perm[i] for i in range(len(flat))}
    if mode=='within_opcode':
        mp={}; groups={o:[] for o in range(O)}
        for x in flat: groups[programs[x[0]].instructions[x[1]].op].append(x)
        for g in groups.values():
            if len(g)<=1:
                for x in g: mp[x]=x
            else:
                sh=g[:]; rng.shuffle(sh)
                for _ in range(len(g)):
                    if all(a!=b for a,b in zip(g,sh)): break
                    sh=sh[1:]+sh[:1]
                for a,b in zip(g,sh): mp[a]=b
        return mp
    raise ValueError(mode)

@torch.inference_mode()
def eval_programs(model,programs,features,mask,mention_ids,mention_tokmask,per,device,shuffle_mode=None,oracle=False,oracle_fields=(),oracle_bind=False):
    model.eval(); correct=0; held=[0,0]; by={}; by_bind={}; instruction_correct=instruction_total=0; bind_correct=bind_total=0; mapping=_deranged_mapping(programs,shuffle_mode)
    for pi,p in enumerate(programs):
        mem=torch.full((R,E),-1,dtype=torch.long); cur=0; pred=False; halted=False
        for si,true_ins in enumerate(p.instructions):
            srcpi,srcsi=mapping[(pi,si)]
            if oracle: ins=programs[srcpi].instructions[srcsi]
            else:
                fid=per[srcpi][srcsi]; f=features[fid:fid+1].to(device); m=mask[fid:fid+1].to(device); mi=mention_ids[fid:fid+1].to(device); mm=mention_tokmask[fid:fid+1].to(device); lgs=model.logits(f,m,mi,mm); vals=[int(x.argmax(-1)) for x in lgs]
                if 'op' in oracle_fields: vals[0]=true_ins.op
                if 'e1' in oracle_fields and true_ins.e1>=0: vals[1]=true_ins.e1
                if 'rel' in oracle_fields and true_ins.rel>=0: vals[2]=true_ins.rel
                if 'e2' in oracle_fields and true_ins.e2>=0: vals[3]=true_ins.e2
                if oracle_bind and true_ins.op==OID['BIND']:
                    vals[0]=true_ins.op; vals[1]=true_ins.e1; vals[2]=true_ins.rel; vals[3]=true_ins.e2
                ins=Instr(vals[0],vals[1],vals[2],vals[3])
                if not shuffle_mode:
                    ok=ins.op==true_ins.op and all(getattr(true_ins,a)<0 or getattr(ins,a)==getattr(true_ins,a) for a in ('e1','rel','e2')); instruction_correct+=ok; instruction_total+=1
                    if true_ins.op==OID['BIND']: bind_total+=1; bind_correct+=ok
            if halted: continue
            if ins.op==OID['BIND']: mem[ins.rel,ins.e1]=ins.e2
            elif ins.op==OID['LOAD']: cur=ins.e1
            elif ins.op==OID['FOLLOW']:
                nxt=int(mem[ins.rel,cur]); cur=nxt if nxt>=0 else cur
            elif ins.op==OID['COMPARE']: pred=(cur==ins.e1)
            elif ins.op==OID['SELECT']: cur=ins.e1 if pred else ins.e2
            elif ins.op==OID['HALT']: halted=True
        hit=(cur==p.answer and halted); correct+=hit; by.setdefault(p.chain_len,[0,0]); by[p.chain_len][0]+=hit; by[p.chain_len][1]+=1
        bc=sum(1 for x in p.instructions if x.op==OID['BIND']); by_bind.setdefault(bc,[0,0]); by_bind[bc][0]+=hit; by_bind[bc][1]+=1
        if p.contains_heldout: held[1]+=1; held[0]+=hit
    return {'accuracy':correct/len(programs),'heldout_accuracy':held[0]/max(held[1],1),'instruction_joint':instruction_correct/max(instruction_total,1) if not oracle and not shuffle_mode else None,'bind_instruction_joint':bind_correct/max(bind_total,1) if not oracle and not shuffle_mode else None,'by_chain':{str(k):v[0]/v[1] for k,v in by.items()},'by_bind_count':{str(k):v[0]/v[1] for k,v in by_bind.items()}}

def benchmark_baselines(programs):
    c={i:0 for i in range(E)}
    for p in programs: c[p.answer]+=1
    majority=max(c.values())/len(programs)
    return {'uniform_chance':1.0/E,'majority_answer_accuracy':majority,'answer_counts':{ENTITIES[i]:c[i] for i in range(E)}}

def run(args):
    device = torch.device("cpu" if FORCE_CPU else ("cuda" if torch.cuda.is_available() else "cpu")); set_seed(SEED); spec=PhraseSpec(); print('phrase manifest',spec.manifest(),flush=True)
    train_prog=generate_programs(SEED+100,(1,2,3,4),args.train_programs_per_len,train=True); dev_prog=generate_programs(SEED+200,(1,2,3,4),args.dev_programs_per_len,train=True); test_prog=generate_programs(SEED+9000,(1,2,3,4,5,6,8),args.test_programs_per_len,train=False)
    train_base=benchmark_baselines(train_prog); dev_base=benchmark_baselines(dev_prog); test_base=benchmark_baselines(test_prog)
    if args.smoke:
        assert all(execute_oracle(p)==p.answer for p in train_prog+dev_prog+test_prog)
        assert test_base['majority_answer_accuracy'] <= 1.0/E + 1e-9, test_base
        def oracle_shuffle(mode):
            mp=_deranged_mapping(test_prog,mode); c=0
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
            return c/len(test_prog)
        p=test_prog[0]; T=len(p.instructions)
        def make_logits(nc, attr):
            x=torch.full((T,nc),-6.0)
            for t,ins in enumerate(p.instructions):
                idx=getattr(ins,attr); idx=0 if idx<0 else idx; x[t,idx]=6.0
            return x
        ol=make_logits(O,'op'); e1l=make_logits(E,'e1'); rl=make_logits(R,'rel'); e2l=make_logits(E,'e2')
        _,cur,halted,exact=hard_st_execute_one(ol,e1l,rl,e2l,p,trajectory_weight=0.0); assert exact==1 and int(cur.argmax())==p.answer and float(halted)>0.5
        ol=ol.clone().requires_grad_(True); e1l=e1l.clone().requires_grad_(True); rl=rl.clone().requires_grad_(True); e2l=e2l.clone().requires_grad_(True)
        load_t=next(i for i,x in enumerate(p.instructions) if x.op==OID['LOAD']); true_e=p.instructions[load_t].e1; wrong=(true_e+1)%E
        with torch.no_grad(): e1l[load_t].fill_(-6.0); e1l[load_t,wrong]=6.0
        loss,_,_,_=hard_st_execute_one(ol,e1l,rl,e2l,p,trajectory_weight=0.0); loss.backward(); grads=[z.grad for z in (ol,e1l,rl,e2l)]; grad_norm=sum(float(g.abs().sum()) for g in grads if g is not None); assert grad_norm>0 and all(torch.isfinite(g).all() for g in grads)
        return {'smoke':True,'balanced_test_baseline':test_base,'oracle_full_shuffle':oracle_shuffle('full'),'oracle_within_opcode_shuffle':oracle_shuffle('within_opcode'),'hard_st_oracle_exact':True,'hard_st_grad_norm':grad_norm,'manifest':spec.manifest(),'program_counts':{'train':len(train_prog),'dev':len(dev_prog),'test':len(test_prog)}}
    ensure_deps(); qwen,tok=load_qwen(args.model,device)
    scan_ids=sorted(set([r[0] for r in spec.train_examples+spec.scan_examples])); pooled=pooled_all_layers(qwen,tok,spec.texts,scan_ids,device,batch=args.qwen_batch); scan=scan_layers(spec,pooled); layers=scan['selected']; print('selected layers',layers,'head_top',scan['head_top'],flush=True)
    feats,mask,mids,mmask=token_features(qwen,tok,spec.texts,layers,device,batch=args.qwen_batch); d=qwen.config.hidden_size
    op_proto=anchor_sentence_proto(feats,mask,spec.op_anchor_ids); r_proto=anchor_proto(feats,mask,spec.rel_anchor_ids)
    model=LogicFOGV7(d,layers,scan['head_top'],op_proto,r_proto).to(device); base_cache=FeatureCache(feats.to(device),mask.to(device),mids.to(device),mmask.to(device))
    isolated_hist=train_isolated(model,base_cache,spec.train_examples,device,args.isolated_steps,args.seed)
    tr_text,tr_per=build_program_text_cache(train_prog,'train'); dv_text,dv_per=build_program_text_cache(dev_prog,'scan')
    tr_f,tr_m,tr_mi,tr_mm=token_features(qwen,tok,tr_text,layers,device,batch=args.qwen_batch,max_tokens=64); dv_f,dv_m,dv_mi,dv_mm=token_features(qwen,tok,dv_text,layers,device,batch=args.qwen_batch,max_tokens=64)
    hist=isolated_hist+run_st_curriculum(model,train_prog,tr_f,tr_m,tr_mi,tr_mm,tr_per,device,args.seed)
    tr_eval=eval_programs(model,train_prog,tr_f,tr_m,tr_mi,tr_mm,tr_per,device); dv_eval=eval_programs(model,dev_prog,dv_f,dv_m,dv_mi,dv_mm,dv_per,device); print('train',tr_eval,'dev',dv_eval,flush=True)
    scan_metrics=eval_examples(model,base_cache,spec.scan_examples,device); test_phrase_metrics=eval_examples(model,base_cache,spec.test_examples,device)
    te_text,te_per=build_program_text_cache(test_prog,'test'); te_f,te_m,te_mi,te_mm=token_features(qwen,tok,te_text,layers,device,batch=args.qwen_batch,max_tokens=64)
    normal=eval_programs(model,test_prog,te_f,te_m,te_mi,te_mm,te_per,device); sh_full=eval_programs(model,test_prog,te_f,te_m,te_mi,te_mm,te_per,device,shuffle_mode='full'); sh_op=eval_programs(model,test_prog,te_f,te_m,te_mi,te_mm,te_per,device,shuffle_mode='within_opcode'); oracle=eval_programs(model,test_prog,te_f,te_m,te_mi,te_mm,te_per,device,oracle=True)
    interventions={'oracle_opcode':eval_programs(model,test_prog,te_f,te_m,te_mi,te_mm,te_per,device,oracle_fields=('op',)),'oracle_e1':eval_programs(model,test_prog,te_f,te_m,te_mi,te_mm,te_per,device,oracle_fields=('e1',)),'oracle_relation':eval_programs(model,test_prog,te_f,te_m,te_mi,te_mm,te_per,device,oracle_fields=('rel',)),'oracle_e2':eval_programs(model,test_prog,te_f,te_m,te_mi,te_mm,te_per,device,oracle_fields=('e2',)),'oracle_all_args':eval_programs(model,test_prog,te_f,te_m,te_mi,te_mm,te_per,device,oracle_fields=('e1','rel','e2')),'oracle_bind':eval_programs(model,test_prog,te_f,te_m,te_mi,te_mm,te_per,device,oracle_bind=True)}
    print('locked',normal,'shuffle_full',sh_full,'shuffle_op',sh_op,'oracles',interventions,flush=True)
    drop=normal['accuracy']-sh_full['accuracy']
    if normal['heldout_accuracy']>=0.65 and normal['accuracy']>=0.75 and drop>=0.30: verdict='PASS_LOGIC_V7'
    elif tr_eval['accuracy']<0.95: verdict='HARD_TRAINING_FAILURE'
    elif test_phrase_metrics['joint']<0.85 or test_phrase_metrics['opcode']<0.90: verdict='SEMANTIC_INTERFACE_LIMITED'
    else: verdict='PROGRAM_TRANSFER_LIMITED'
    out={'protocol':'FOG_LOGIC_V7_COPY_BOUND_ENTITY_ROLES','model':args.model,'frozen_backbone':True,'selected_layers':layers,'layer_scan':scan,'phrase_manifest':spec.manifest(),'holdouts':{'follow_relation':HELD_FOLLOW_REL,'compare_entity':HELD_COMPARE_ENTITY,'select_pair':HELD_SELECT_PAIR},'train_program_depths':[1,2,3,4],'test_program_depths':[1,2,3,4,5,6,8],'training_forward':'hard_argmax_straight_through','semantic_heads':'sentence_opcode + relation_semantics + lexical_entity_copy + learned_role_addressing','train_history':hist,'scan_phrase_metrics':scan_metrics,'locked_phrase_metrics':test_phrase_metrics,'train_program_eval':tr_eval,'dev_program_eval':dv_eval,'locked_program_eval':normal,'shuffle_full':sh_full,'shuffle_within_opcode':sh_op,'field_oracle_interventions':interventions,'oracle_program_eval':oracle,'baselines':{'train':train_base,'dev':dev_base,'test':test_base},'layer_mix_weights':model.mix_report(),'trainable_params':sum(p.numel() for p in model.parameters() if p.requires_grad),'program_counts':{'train':len(train_prog),'dev':len(dev_prog),'test':len(test_prog)},'entity_binding':'protected_lexical_copy_with_learned_roles','verdict':verdict,'runtime_env':{'torch':torch.__version__,'device':str(device),'platform':platform.platform()}}
    torch.save({'state_dict':model.state_dict(),'selected_layers':layers,'head_top':scan['head_top'],'phrase_manifest':spec.manifest(),'protocol':out['protocol']},args.checkpoint); Path(args.output).write_text(json.dumps(out,indent=2),encoding='utf-8'); return out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--model',default='Qwen/Qwen2.5-0.5B-Instruct'); ap.add_argument('--device',default='auto'); ap.add_argument('--qwen-batch',type=int,default=24); ap.add_argument('--isolated-steps',type=int,default=1600); ap.add_argument('--train-programs-per-len',type=int,default=72); ap.add_argument('--dev-programs-per-len',type=int,default=24); ap.add_argument('--test-programs-per-len',type=int,default=72); ap.add_argument('--seed',type=int,default=0); ap.add_argument('--output',default='fog_logic_v7_results.json'); ap.add_argument('--checkpoint',default='fog_logic_v7.pt'); ap.add_argument('--smoke',action='store_true'); args=ap.parse_args(); t=time.time(); r=run(args); print(json.dumps({'summary':{k:r.get(k) for k in ('smoke','dev_program_eval','locked_program_eval','shuffle_full','shuffle_within_opcode','oracle_program_eval','verdict') if k in r},'runtime_seconds':time.time()-t},indent=2))
if __name__=='__main__': main()
