"""Frozen-weight entity-vocabulary expansion on known training wording.

This tests copying and register dimensions, NOT unseen language understanding.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import random
import torch
from cache import load_train_dev
from integrity import derangement
from model import TextBank,forward_refs,slot_payload,hard_arguments,execute_tensor
from predict import load_adapter,runtime_vocabulary
from run import padded_arguments


def run(checkpoint,features,output):
    torch.set_num_threads(2);rng=random.Random(202609068201)
    ref,data,_,banks,_=load_train_dev(features)
    model,meta,_=load_adapter(checkpoint,'cpu')
    if meta['backbone_revision']!=data['backbone_revision']:raise ValueError('Backbone mismatch')
    names=[f'Node-{i:03}' for i in range(28)]+['North Office','Acme Corp','Иван','Склад']
    runtime=runtime_vocabulary(ref,names)
    bank=TextBank(names,canonical_names=ref.ENTITIES)
    for k in ('texts','spans','index','features','mask','mentions','valid'):
        setattr(bank,k,getattr(banks['train'],k))
    refs=[];lengths=[];answers=[];witness=hashlib.sha256()
    for i in range(96):
        graph=[derangement(range(runtime.E),rng.randrange(2**31)) for _ in range(ref.R)]
        facts=[(a,r,graph[r][a]) for r in range(ref.R) for a in range(runtime.E)];rng.shuffle(facts)
        current=rng.randrange(runtime.E)
        instructions=[('BIND',a,r,b) for a,r,b in facts]+[('LOAD',current,-1,-1)]
        for _ in range(64):
            r=rng.randrange(ref.R);instructions.append(('FOLLOW',-1,r,-1));current=graph[r][current]
        truth=bool(i%2);target=i%runtime.E
        compare=current if truth else (current+1+rng.randrange(runtime.E-1))%runtime.E
        other=(target+1+rng.randrange(runtime.E-1))%runtime.E
        instructions.extend([('COMPARE',compare,-1,-1),('SELECT',target if truth else other,-1,other if truth else target),
                             ('HALT',-1,-1,-1),('LOAD',(target+1)%runtime.E,-1,-1),('FOLLOW',-1,4,-1)])
        texts=[]
        for op,a,r,b in instructions:
            text=ref.TRAIN_TEMPLATES[op][0].format(a=names[a] if a>=0 else None,b=names[b] if b>=0 else None,r=ref.RELATIONS[r] if r>=0 else None)
            refs.append(bank.add(text));texts.append(text)
        witness.update(json.dumps({'texts':texts,'expected':target},ensure_ascii=False).encode())
        lengths.append(len(texts));answers.append(target)
    logits=forward_refs(model,bank,refs)
    args=hard_arguments(*logits,slot_payload(refs,runtime.E,'cpu'))
    state=execute_tensor(*padded_arguments(args,lengths))
    correct=(state['current'].argmax(-1)==torch.tensor(answers))&(state['halted']>.5)
    mapping=derangement(range(len(refs)),202609068202)
    order=torch.tensor([mapping[i] for i in range(len(refs))])
    shuffle=execute_tensor(*padded_arguments(tuple(x[order] for x in args),lengths))
    shuffled=(shuffle['current'].argmax(-1)==torch.tensor(answers))&(shuffle['halted']>.5)
    out={'protocol':'FROZEN_CUSTOM_ENTITY_STRESS_V1','checkpoint_sha256':hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
         'seed':202609068201,'data_sha256':witness.hexdigest(),'entities':names,'entity_count':runtime.E,
         'facts_per_program':runtime.E*ref.R,'follow_steps':64,'programs':96,'correct':int(correct.sum()),
         'accuracy':float(correct.float().mean()),'programs_with_undefined_reads':int((state['invalid_reads']>.5).sum()),
         'shuffle_correct':int(shuffled.sum()),'shuffle_accuracy':float(shuffled.float().mean()),
         'training_updates':0,'wording':'original TRAIN template zero','interpretation':'Copy/payload dimension expansion, not new-language generalization'}
    Path(output).write_text(json.dumps(out,indent=2,ensure_ascii=False));print(json.dumps(out,indent=2,ensure_ascii=False));return out

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--features',required=True);p.add_argument('--output',default='entity-stress.json')
    a=p.parse_args();run(a.checkpoint,a.features,a.output)
