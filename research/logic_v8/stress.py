"""Post-training length/memory-capacity stress test using KNOWN TRAIN wording.

This is not a replacement for the locked language-transfer test. New graph seeds,
72 memory facts per graph, and 16/32/64/128 read steps test recurrent execution.
No training or model selection occurs in this script.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import random
import torch
from integrity import derangement
from cache import load_train_dev
from predict import load_adapter
from run import predict_instructions,evaluate_predictions,dataset_witness

STRESS_SEED = 202609067701
DEPTHS = (16,32,64,128)

def make_stress(ref, per_depth=120):
    if per_depth<=0 or per_depth%ref.E:raise ValueError('Count must be a positive multiple of twelve')
    rng=random.Random(STRESS_SEED);out=[]
    for depth in DEPTHS:
        for i in range(per_depth):
            graph=[derangement(range(ref.E),rng.randrange(2**31)) for _ in range(ref.R)]
            facts=[ref.Instr(0,a,r,graph[r][a]) for r in range(ref.R) for a in range(ref.E)]
            rng.shuffle(facts)
            current=rng.randrange(ref.E);rows=facts+[ref.Instr(1,current)]
            for _ in range(depth):
                relation=rng.randrange(ref.R);rows.append(ref.Instr(2,rel=relation));current=graph[relation][current]
            true_branch=bool(i%2)
            compare=current if true_branch else (current+1+rng.randrange(ref.E-1))%ref.E
            answer=i%ref.E;other=(answer+1+rng.randrange(ref.E-1))%ref.E
            rows.append(ref.Instr(3,compare))
            rows.append(ref.Instr(4,answer if true_branch else other,e2=other if true_branch else answer))
            rows.extend([ref.Instr(5),ref.Instr(1,(answer+1)%ref.E),ref.Instr(2,rel=4)])
            held=any(x.op==2 and x.rel==4 for x in rows[:-2])
            p=ref.LogicProgram(tuple(rows),answer,depth,held)
            if ref.execute_oracle(p)!=answer:raise AssertionError('Stress generator oracle mismatch')
            out.append(p)
    return out

def run(checkpoint, features, output, per_depth=120):
    torch.set_num_threads(2)
    ref,data,_,banks,_=load_train_dev(features)
    model,meta,_=load_adapter(checkpoint,'cpu')
    if meta['backbone_revision']!=data['backbone_revision']:raise ValueError('Feature revision mismatch')
    programs=make_stress(ref,per_depth)
    bank=banks['train']
    refs=[bank.add(ref.instr_text(ins,'train',0)) for p in programs for ins in p.instructions]
    pred=predict_instructions(model,bank,refs,ref)
    report={'protocol':'FOG_KNOWN_WORDING_LENGTH_MEMORY_STRESS_V1', 'seed':STRESS_SEED,
            'graph_hash':dataset_witness(programs),'depths':list(DEPTHS),'facts_per_program':72,
            'programs_per_depth':per_depth,'wording':'existing TRAIN template variant zero only',
            'checkpoint_sha256':hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
            'normal':evaluate_predictions(pred,programs,ref),
            'shuffle_full':evaluate_predictions(pred,programs,ref,mode='full'),
            'oracle':evaluate_predictions([i for p in programs for i in p.instructions],programs,ref),
            'interpretation':'Tests new graph composition, longer execution and larger memory; NOT unseen-language generalization.'}
    Path(output).write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True);p.add_argument('--features',required=True);p.add_argument('--output',default='stress-results.json');p.add_argument('--per-depth',type=int,default=120)
    a=p.parse_args();run(a.checkpoint,a.features,a.output,a.per_depth)
