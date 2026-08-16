#!/usr/bin/env python3
"""Evaluation-only natural-language audit for FOG factorized v2.

Loads a saved FOG checkpoint and the original frozen Qwen2.5-0.5B backbone.
NO training or weight updates are performed.

The audit intentionally separates:
  1) supported arithmetic language (ADD/SUB/MUL modulo 31),
  2) natural paraphrase/implicit-language stress,
  3) unsupported logical operations (branch/relational/division),
  4) causal controls (zero semantics, engine bypass, shuffled instruction semantics),
  5) whole-prompt stress showing whether the current pipeline itself segments prose.
"""
from __future__ import annotations
import argparse, json, math
from dataclasses import dataclass
from typing import List, Tuple
import torch
from torch import nn
import torch.nn.functional as F

P=31
OPN=('ADD','SUB','MUL')

def complex_block_product(a,b,conj=False):
    ap=a.reshape(*a.shape[:-1],-1,2); bp=b.reshape(*b.shape[:-1],-1,2)
    ar,ai=ap.unbind(-1); br,bi=bp.unbind(-1)
    if conj: bi=-bi
    out=torch.stack([ar*br-ai*bi, ar*bi+ai*br],-1)
    return (out/math.sqrt(2.)).reshape_as(a)

class Executor(nn.Module):
    def __init__(self, codebook, mul_mats):
        super().__init__(); self.register_buffer('codebook',codebook.float()); self.register_buffer('mul_mats',mul_mats.float())
    def forward(self,value,op,operand):
        qw=F.one_hot(operand,P).float(); ow=F.one_hot(op,3).float(); operand_z=qw@self.codebook
        add=complex_block_product(value,operand_z,False); sub=complex_block_product(value,operand_z,True)
        ml=torch.einsum('bd,kde->bke',value,self.mul_mats)
        z0=self.codebook[0].view(1,1,-1).expand(value.size(0),1,-1)
        mall=torch.cat([z0,ml[:,1:]],1); mul=torch.einsum('bk,bkd->bd',qw,mall)
        cand=torch.stack([add,sub,mul],1); out=torch.einsum('bo,bod->bd',ow,cand)
        return F.normalize(out.float(),dim=-1)*math.sqrt(value.size(-1))
    def decode(self,z):
        return (F.normalize(z.float(),dim=-1)@F.normalize(self.codebook.float(),dim=-1).T).argmax(-1)

class LayerMixer(nn.Module):
    def __init__(self, logits, prior):
        super().__init__(); self.logits=nn.Parameter(logits.clone().float(), requires_grad=False); self.register_buffer('prior',prior.clone().float())
    def weights(self): return torch.softmax(self.logits.float(),-1)
    def context(self,x): return torch.einsum('k,bktd->btd',self.weights().to(x),x)
    def proto(self,x): return torch.einsum('k,...kd->...d',self.weights().to(x),x)

class Binder(nn.Module):
    def __init__(self,sd,prefix):
        super().__init__(); self.register_buffer('prototype_layers',sd[prefix+'.prototype_layers'].float())
        if prefix+'.anchor_mask' in sd: self.register_buffer('anchor_mask',sd[prefix+'.anchor_mask'].bool())
        else: self.register_buffer('anchor_mask',torch.ones(self.prototype_layers.shape[:2],dtype=torch.bool))
        d=self.prototype_layers.size(-1); rank=sd[prefix+'.q_proj.weight'].size(0)
        self.q_proj=nn.Linear(d,rank,bias=False); self.k_proj=nn.Linear(d,rank,bias=False)
        self.q_proj.weight.data.copy_(sd[prefix+'.q_proj.weight'].float()); self.k_proj.weight.data.copy_(sd[prefix+'.k_proj.weight'].float())
        self.logit_scale=nn.Parameter(sd[prefix+'.logit_scale'].clone().float(), requires_grad=False)
        for p in self.q_proj.parameters(): p.requires_grad_(False)
        for p in self.k_proj.parameters(): p.requires_grad_(False)
    def forward(self,layer_features,mask,mix):
        ctx=mix.context(layer_features.float()); proto=mix.proto(self.prototype_layers)
        q=F.normalize(self.q_proj(ctx).float(),-1); k=F.normalize(self.k_proj(proto).float(),-1)
        score=self.logit_scale.exp().clamp(max=100.)*torch.einsum('btr,car->btca',q,k)
        score=score.masked_fill(~mask[:,:,None,None],float('-inf'))
        score=score.masked_fill(~self.anchor_mask[None,None,:,:],float('-inf'))
        return torch.logsumexp(score,dim=(1,3))

class FOG(nn.Module):
    def __init__(self,sd):
        super().__init__(); self.mix=LayerMixer(sd['layer_mixer.logits'],sd['layer_mixer.prior'])
        self.qty=Binder(sd,'quantity_binder'); self.op=Binder(sd,'opcode_binder'); self.halt=Binder(sd,'halt_binder')
        self.register_buffer('codebook',sd['codebook'].float()); self.exec=Executor(sd['executor.codebook'],sd['executor.mul_mats'])
        self.eval()
    @staticmethod
    def top2(logits):
        pr=torch.softmax(logits.float(),-1); vals,idx=pr.topk(min(2,pr.size(-1)),dim=-1)
        return [(int(idx[0,j]),float(vals[0,j])) for j in range(idx.size(1))]
    def bind_start(self,f,m):
        lg=self.qty(f,m,self.mix); ix=lg.argmax(-1); return self.codebook[ix],ix,self.top2(lg)
    def bind_inst(self,f,m):
        ol=self.op(f,m,self.mix); ql=self.qty(f,m,self.mix); oi=ol.argmax(-1); qi=ql.argmax(-1)
        return oi,qi,self.top2(ol),self.top2(ql)
    def bind_halt(self,f,m):
        lg=self.halt(f,m,self.mix); return lg.argmax(-1),self.top2(lg)

@dataclass
class Case:
    name:str; start:str; start_value:int; instructions:List[Tuple[str,int,int]]; stop:str; expected:int|None; category:str='supported'; note:str=''

CASES=[
 Case('plain_unseen_chain','Suppose the register begins with seventeen.',17,[('Give the running number eleven extra units.',0,11),('Make it five times as large as it is now.',2,5),('Take twelve away from what remains.',1,12),('Add another seven.',0,7),('Triple the result.',2,3)],'That is enough arithmetic. Keep the value you have and stop.',2,'supported','Mixes held-out pairs and implicit triple.'),
 Case('story_style','Mira begins with seventeen tokens in a cyclic counter.',17,[('A friend gives her eleven more tokens.',0,11),('The counter is then expanded to five times its amount.',2,5),('She spends twelve tokens from the counter.',1,12)],'The story ends here; preserve the amount now shown.',4,'supported','Story vocabulary rather than register vocabulary.'),
 Case('colloquial','Start me off at twenty-nine.',29,[('Bump it up by four.',0,4),('Double it.',2,2),('Knock three off the result.',1,3),('Give it ten more.',0,10)],'Okay, stop there.',11,'supported','Colloquial + implicit double.'),
 Case('heldout_pairs','The cyclic register initially contains six.',6,[('Put eleven more on top of it.',0,11),('Quintuple what you have.',2,5),('Take away twelve.',1,12)],'Freeze the register now.',11,'supported','All three never-trained opcode/operand pairings.'),
 Case('negation_distractor','Begin at nine.',9,[('Do not add eleven; subtract two instead.',1,2),('Ignore the words multiply by five; just add one.',0,1)],'Stop now.',8,'stress','Tests compositional negation/distractor handling.'),
 Case('implicit_world_story','A score counter starts on fourteen.',14,[('The score gains three points.',0,3),('A penalty removes five points.',1,5),('The score is tripled.',2,3)],'Final whistle: keep this score.',5,'supported','Natural event semantics.'),
 Case('synonym_and_idiom','We open with a value of twenty-four.',24,[('Increase it by a dozen.',0,12),('Then halve-looking language is absent; instead double the current amount.',2,2),('Take away half a dozen.',1,6)],'Finish here.',10,'stress','Quantities expressed as dozen/half-dozen were not trained.'),
 Case('conditional_branch_UNSUPPORTED','Begin at eight.',8,[('If the current value is even, add three; otherwise subtract two.',-1,-1)],'Stop after the conditional.',None,'unsupported','No BRANCH/COMPARE opcode exists.'),
 Case('relational_logic_UNSUPPORTED','Alice is taller than Bob, and Bob is taller than Carla.',0,[('Who is taller, Alice or Carla?',-1,-1)],'Return the answer.',None,'unsupported','No entity/relation/FOLLOW primitive exists.'),
 Case('division_UNSUPPORTED','Begin at twenty.',20,[('Halve the current value.',-1,-1)],'Stop.',None,'unsupported','No DIV primitive exists.'),
]

class FeatureCache:
    def __init__(self,model,tok,layers,device,batch_size=24,max_tokens=96):
        self.model=model; self.tok=tok; self.layers=layers; self.device=device; self.bs=batch_size; self.max_tokens=max_tokens; self.cache={}
    @torch.inference_mode()
    def preload(self,texts):
        uniq=[]
        for t in texts:
            if t not in self.cache: uniq.append(t)
        for st in range(0,len(uniq),self.bs):
            rows=uniq[st:st+self.bs]
            enc=self.tok(rows,padding=True,truncation=True,max_length=self.max_tokens,return_tensors='pt',add_special_tokens=False)
            enc={k:v.to(self.device) for k,v in enc.items()}
            out=self.model(**enc,use_cache=False,return_dict=True,output_hidden_states=True)
            hs=out.hidden_states; am=enc['attention_mask'].bool()
            for i,t in enumerate(rows):
                n=int(am[i].sum())
                f=torch.stack([hs[li][i,:n].float().cpu() for li in self.layers],dim=0)
                self.cache[t]=(f,torch.ones(n,dtype=torch.bool))
    def get(self,text):
        f,m=self.cache[text]
        return f[None].to(self.device),m[None].to(self.device)

def full_prompt(c): return c.start+' '+' '.join(x[0] for x in c.instructions)+' '+c.stop

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--checkpoint',default='fog_v2_seed0_fp16.pt'); ap.add_argument('--model',default='Qwen/Qwen2.5-0.5B-Instruct'); ap.add_argument('--seed',default='0'); ap.add_argument('--device',default='auto'); ap.add_argument('--output',default='natural_language_audit.json'); args=ap.parse_args()
    from transformers import AutoTokenizer,AutoModel
    device=torch.device('cuda' if args.device=='auto' and torch.cuda.is_available() else ('cpu' if args.device=='auto' else args.device))
    dtype=torch.float16 if device.type=='cuda' else torch.float32
    tok=AutoTokenizer.from_pretrained(args.model,use_fast=True); tok.pad_token=tok.pad_token or tok.eos_token; tok.padding_side='right'
    try:qwen=AutoModel.from_pretrained(args.model,dtype=dtype,low_cpu_mem_usage=True).to(device).eval()
    except TypeError:qwen=AutoModel.from_pretrained(args.model,torch_dtype=dtype,low_cpu_mem_usage=True).to(device).eval()
    for p in qwen.parameters():p.requires_grad_(False)
    payload=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    sd=payload['main_fog_state_dicts'][str(args.seed)] if 'main_fog_state_dicts' in payload else payload['main_fog_state_dict']
    layers=payload['selected_hidden_state_indices']; fog=FOG(sd).to(device).eval()
    print('device',device,'layers',layers,'layer_mix',fog.mix.weights().tolist(),flush=True)
    all_text=[]
    for c in CASES: all_text += [c.start,c.stop]+[x[0] for x in c.instructions]+[full_prompt(c)]
    cache=FeatureCache(qwen,tok,layers,device); cache.preload(all_text)
    results=[]; supported_pass=supported_n=stress_pass=stress_n=0
    for c in CASES:
        trace={'name':c.name,'category':c.category,'note':c.note,'start_text':c.start,'instructions':[],'expected':c.expected}
        f,m=cache.get(c.start); z,sidx,stop2=fog.bind_start(f,m); current=int(fog.exec.decode(z)); trace['start_pred']=current; trace['start_expected']=c.start_value; trace['start_top2']=stop2
        for text,trueop,truearg in c.instructions:
            f,m=cache.get(text); oi,qi,otop,qtop=fog.bind_inst(f,m); hi,htop=fog.bind_halt(f,m)
            before=current; z=fog.exec(z,oi,qi); current=int(fog.exec.decode(z))
            trace['instructions'].append({'text':text,'pred_opcode':OPN[int(oi)],'pred_operand':int(qi),'opcode_top2':otop,'operand_top2':qtop,'halt_pred':'HALT' if int(hi) else 'EXECUTE','halt_top2':htop,'expected_opcode':OPN[trueop] if trueop>=0 else None,'expected_operand':truearg if truearg>=0 else None,'before':before,'after':current})
        sf,sm=cache.get(c.stop); sh,shtop=fog.bind_halt(sf,sm); trace['stop_pred']='HALT' if int(sh) else 'EXECUTE'; trace['stop_top2']=shtop; trace['final']=current
        if c.expected is not None:
            trace['pass']=(current==c.expected and int(sh)==1)
            if c.category=='supported': supported_pass+=int(trace['pass']); supported_n+=1
            elif c.category=='stress': stress_pass+=int(trace['pass']); stress_n+=1
        else: trace['pass']=None
        trace['engine_bypass_final']=trace['start_pred']
        zf=torch.zeros(1,len(layers),1,int(qwen.config.hidden_size),device=device); zm=torch.ones(1,1,dtype=torch.bool,device=device)
        zz,_,_=fog.bind_start(zf,zm)
        for _text,_op,_arg in c.instructions:
            oi,qi,*_=fog.bind_inst(zf,zm); zz=fog.exec(zz,oi,qi)
        trace['zero_semantics_final']=int(fog.exec.decode(zz))
        wf,wm=cache.get(full_prompt(c)); _wz,wsix,_=fog.bind_start(wf,wm); woi,wqi,wotop,wqtop=fog.bind_inst(wf,wm); wh,whtop=fog.bind_halt(wf,wm)
        trace['whole_prompt_single_step_probe']={'start_pred':int(wsix),'opcode':OPN[int(woi)],'operand':int(wqi),'halt':'HALT' if int(wh) else 'EXECUTE','opcode_top2':wotop,'operand_top2':wqtop,'halt_top2':whtop}
        results.append(trace); print(json.dumps(trace,ensure_ascii=False),flush=True)
    eval_cases=[c for c in CASES if c.expected is not None]; flat=[text for c in eval_cases for text,_,_ in c.instructions]; shuffled=flat[1:]+flat[:1]; ptr=0; shuffled_rows=[]
    for c in eval_cases:
        sf,sm=cache.get(c.start); z,_,_=fog.bind_start(sf,sm)
        for _text,_op,_arg in c.instructions:
            wrong=shuffled[ptr]; ptr+=1; f,m=cache.get(wrong); oi,qi,*_=fog.bind_inst(f,m); z=fog.exec(z,oi,qi)
        final=int(fog.exec.decode(z)); shuffled_rows.append({'case':c.name,'final':final,'expected':c.expected,'pass':final==c.expected})
    summary={'supported_exact':[supported_pass,supported_n,supported_pass/max(supported_n,1)],'stress_exact':[stress_pass,stress_n,stress_pass/max(stress_n,1)],'semantic_shuffle_exact':[sum(int(x['pass']) for x in shuffled_rows),len(shuffled_rows),sum(int(x['pass']) for x in shuffled_rows)/max(len(shuffled_rows),1)],'unsupported_cases':len([c for c in CASES if c.expected is None])}
    out={'checkpoint':args.checkpoint,'seed':str(args.seed),'qwen':args.model,'layers':layers,'layer_mix':fog.mix.weights().tolist(),'summary':summary,'cases':results,'semantic_shuffle':shuffled_rows,'notes':['NO TRAINING performed.','Supported arithmetic uses only ADD/SUB/MUL modulo 31.','Unsupported logic probes diagnose missing operator vocabulary and should not be scored as arithmetic failures.','The current pipeline is instruction-segmented by the harness. whole_prompt_single_step_probe deliberately tests this boundary; it is not a learned multi-instruction parser.','zero_semantics preserves the FOG executor but deletes all language features; semantic_shuffle preserves real Qwen features but assigns them to the wrong steps.']}
    open(args.output,'w',encoding='utf-8').write(json.dumps(out,ensure_ascii=False,indent=2)); print('SUMMARY',json.dumps(summary),flush=True); print('saved',args.output)
if __name__=='__main__': main()
