"""Post-v8.2 relation experiment; previously viewed tests are regression evidence.

Two frozen-Qwen label-likelihood readers are selected using TRAIN/DEV only.
No grammar rules, expected relations, or target answers enter inference.
"""
from __future__ import annotations
import argparse,hashlib,json,time
from pathlib import Path
import torch
from cache import load_train_dev
from model import TextBank
from predict import load_adapter
from run import make_program_refs,predict_instructions,evaluate_predictions,instruction_metrics

PROMPTS = (
    'Identify the relation expressed by the sentence. Choose exactly one label from: {labels}. Return only the label.',
    'Classify the relation in the sentence. Labels and meanings:\n{definitions}\nReturn exactly one label and nothing else.',
)
FRESH_BIND = {
'manager': ('{a} is supervised by {b}.','{b} supervises {a}.','{a} has {b} as their manager.','{b} is in charge of {a}.'),
'parent': ('{a} has {b} as a parent.','{b} is a parent of {a}.','{a} is a child of {b}.','{b} is the mother or father of {a}.'),
'owner': ('{a} is owned by {b}.','{b} is the owner of {a}.','{a} is the property of {b}.','{b} holds ownership of {a}.'),
'north': ('{a} has {b} directly to its north.','{b} lies directly north of {a}.','Going north from {a} takes you to {b}.','{b} is the northern neighbor of {a}.'),
'calls': ('Function {a} invokes function {b}.','Function {b} is invoked by function {a}.','{a} makes a function call to {b}.','{b} is the function that {a} calls.'),
'imports': ('Module {a} has an import dependency on module {b}.','Module {b} is imported by module {a}.','{a} uses an import statement to load {b}.','{b} is the module that {a} imports.'),
}
FRESH_FOLLOW = {
'manager': ('Visit the supervisor of the current person.','Move from this employee to their manager.'),
'parent': ('Visit the parent of the current person.','Move from this child to their parent.'),
'owner': ('Visit the owner of the current item.','Move from this object to its owner.'),
'north': ('Visit the northern neighbor of the current place.','Move from this location toward its north-linked neighbor.'),
'calls': ('Visit the function invoked by the current function.','Move from this caller to its callee.'),
'imports': ('Visit the module imported by the current module.','Move from this module to its import dependency.'),
}

class FrozenRelationReader:
    def __init__(self,backbone,tokenizer,reference,prompt_index):
        if prompt_index not in range(len(PROMPTS)):raise ValueError('Unsupported prompt')
        self.backbone,self.tokenizer,self.ref=backbone,tokenizer,reference
        self.device=next(backbone.parameters()).device
        self.system=PROMPTS[prompt_index].format(labels=', '.join(reference.RELATIONS),
            definitions='\n'.join(f'{r}: '+', '.join(reference.REL_ANCHORS[r]) for r in reference.RELATIONS))
        self.label_ids=[tokenizer.encode(r,add_special_tokens=False) for r in reference.RELATIONS]
        self.cache={}
    @torch.inference_mode()
    def scores(self,texts,batch=8):
        missing=list(dict.fromkeys(t for t in texts if t not in self.cache))
        for start in range(0,len(missing),batch):
            chunk=missing[start:start+batch]
            prompts=[self.tokenizer.apply_chat_template([{'role':'system','content':self.system},
                {'role':'user','content':t}],tokenize=True,add_generation_prompt=True) for t in chunk]
            if all(len(x)==1 for x in self.label_ids):
                ids,mask=self._pad(prompts)
                hidden=self.backbone.model(input_ids=ids,attention_mask=mask,use_cache=False,return_dict=True).last_hidden_state
                last=hidden[torch.arange(len(prompts),device=self.device),mask.sum(-1)-1]
                score=self.backbone.lm_head(last).float()[:,[x[0] for x in self.label_ids]]
                for text,row in zip(chunk,score.cpu()):self.cache[text]=row
            else:
                for text,prefix in zip(chunk,prompts):
                    sequences=[prefix+tail[:-1] for tail in self.label_ids]
                    ids,mask=self._pad(sequences)
                    hidden=self.backbone.model(input_ids=ids,attention_mask=mask,use_cache=False,return_dict=True).last_hidden_state
                    scores=[]
                    for i,tail in enumerate(self.label_ids):
                        h=hidden[i,len(prefix)-1:len(prefix)+len(tail)-1]
                        lp=torch.log_softmax(self.backbone.lm_head(h).float(),-1)
                        scores.append(lp[torch.arange(len(tail),device=self.device),tail].mean())
                    self.cache[text]=torch.stack(scores).cpu()
        return torch.stack([self.cache[t] for t in texts])
    def _pad(self,sequences):
        if not sequences or max(map(len,sequences))>512:raise ValueError('Unsupported relation-reader input length')
        ids=torch.full((len(sequences),max(map(len,sequences))),self.tokenizer.pad_token_id,device=self.device,dtype=torch.long)
        mask=torch.zeros_like(ids)
        for i,s in enumerate(sequences):ids[i,:len(s)]=torch.tensor(s,device=self.device);mask[i,:len(s)]=1
        return ids,mask

def relations_metric(reader,texts,labels):
    pred=reader.scores(texts).argmax(-1).tolist()
    return {'accuracy':sum(a==b for a,b in zip(pred,labels))/len(labels),'count':len(labels),
            'errors':[{'text':t,'predicted':reader.ref.RELATIONS[p],'expected':reader.ref.RELATIONS[y]}
                      for t,p,y in zip(texts,pred,labels) if p!=y]}

def repair_predictions(reader,pred,bank,refs,ref):
    ids=[i for i,p in enumerate(pred) if p.op in (ref.OID['BIND'],ref.OID['FOLLOW'])]
    output=list(pred)
    if ids:
        labels=reader.scores([bank.texts[refs[i].index] for i in ids]).argmax(-1).tolist()
        for i,relation in zip(ids,labels):
            p=output[i];output[i]=ref.Instr(p.op,p.e1,relation,p.e2)
    return output

def run(args):
    start=time.time();torch.set_num_threads(2)
    ref,data,programs,banks,refs=load_train_dev(args.features)
    model,meta,_=load_adapter(args.checkpoint,torch.device('cpu'))
    from transformers import AutoModelForCausalLM,AutoTokenizer
    tok=AutoTokenizer.from_pretrained(meta['model'],revision=meta['backbone_revision'],use_fast=True)
    tok.pad_token=tok.pad_token or tok.eos_token;tok.padding_side='right'
    backbone=AutoModelForCausalLM.from_pretrained(meta['model'],revision=meta['backbone_revision'],torch_dtype=torch.float32).eval()
    for p in backbone.parameters():p.requires_grad_(False)
    spec=ref.PhraseSpec();candidates=[];readers=[]
    for prompt in range(len(PROMPTS)):
        reader=FrozenRelationReader(backbone,tok,ref,prompt);readers.append(reader)
        results={}
        for split,rows,rr in [('train',spec.train_examples,refs['isolated']),('dev',spec.scan_examples,refs['dev_phrase'])]:
            indices=[i for i,row in enumerate(rows) if row[3]>=0]
            results[split]=relations_metric(reader,[banks[split].texts[rr[i].index] for i in indices],[rows[i][3] for i in indices])
        candidates.append(results);print(json.dumps({'prompt':prompt,'results':results}),flush=True)
    selected=max(range(len(candidates)),key=lambda i:(candidates[i]['dev']['accuracy'],candidates[i]['train']['accuracy'],-i))
    reader=readers[selected]
    config={'protocol':'FOG_RELATION_PROBE_V1','prompt_index':selected,'selection':'TRAIN_DEV_ONLY',
            'base_checkpoint_sha256':hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
            'model':meta['model'],'backbone_revision':meta['backbone_revision']}
    Path(args.config).write_text(json.dumps(config,indent=2))
    out={'protocol':config['protocol'],'candidates':candidates,'selected':selected,'model_updated':False,
         'old_test_status':'Previously viewed test; regression only, not a new blind locked score.'}
    eligible=min(candidates[selected][x]['accuracy'] for x in ('train','dev'))>=.99
    out['eligible_for_integration']=eligible
    if eligible:
        test=ref.generate_programs(ref.SEED+9000,(1,2,3,4,5,6,8),72,train=False)
        tb=TextBank(ref.ENTITIES);tr=make_program_refs(test,'test',tb,ref)
        phr=[tb.add(spec.texts[x[0]]) for x in spec.test_examples]
        tb.featurize(backbone,tok,'cpu')
        flat=[r for p in tr for r in p]
        base=predict_instructions(model,tb,flat,ref)
        changed=repair_predictions(reader,base,tb,flat,ref)
        out['old_test_regression']=evaluate_predictions(changed,test,ref)
        out['shuffle_full']=evaluate_predictions(changed,test,ref,mode='full')
        p=repair_predictions(reader,predict_instructions(model,tb,phr,ref),tb,phr,ref)
        out['old_phrase_regression']=instruction_metrics(p,[ref.Instr(*r[1:]) for r in spec.test_examples],ref)
    fb=TextBank(ref.ENTITIES);texts=[];gold=[]
    for rel in ref.RELATIONS:
        for template in FRESH_BIND[rel]:
            texts.append(template.format(a='Iris',b='Lena'));gold.append(ref.Instr(0,ref.EID['Iris'],ref.RID[rel],ref.EID['Lena']))
        for text in FRESH_FOLLOW[rel]:texts.append(text);gold.append(ref.Instr(2,rel=ref.RID[rel]))
    fr=[fb.add(t) for t in texts]
    out['fresh_relation_only']=relations_metric(reader,[fb.texts[r.index] for r in fr],[r.rel for r in gold])
    fb.featurize(backbone,tok,'cpu')
    pred=repair_predictions(reader,predict_instructions(model,fb,fr,ref),fb,fr,ref)
    out['fresh_full_instruction']=instruction_metrics(pred,gold,ref)
    out['fresh_instruction_errors']=[{'text':t,'predicted':[p.op,p.e1,p.rel,p.e2],'expected':[g.op,g.e1,g.rel,g.e2]}
        for t,p,g in zip(texts,pred,gold) if p.op!=g.op or any(getattr(g,k)>=0 and getattr(g,k)!=getattr(p,k) for k in ('e1','rel','e2'))]
    out['runtime_seconds']=time.time()-start
    Path(args.output).write_text(json.dumps(out,indent=2));print(json.dumps(out),flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--features',required=True);p.add_argument('--checkpoint',required=True)
    p.add_argument('--config',default='relation-reader.json');p.add_argument('--output',default='relation-probe-results.json')
    run(p.parse_args())

if __name__=='__main__':main()
