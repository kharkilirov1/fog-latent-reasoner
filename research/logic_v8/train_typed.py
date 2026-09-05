"""Fit a typed local BIND reader, then freeze before optional locked evaluation.

The base adapter and frozen TRAIN/DEV features must come from the clean legacy-v8
experiment. No additional training text, holdout label or test score is consumed.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import random
import time
import numpy as np
import torch
from cache import load_train_dev
from integrity import REFERENCE_SHA256,strict_gates
from model import TextBank,final_loss
from window_roles import TypedSemanticReader,WindowRoleReader
from run import (program_batch,predict_instructions,evaluate_predictions,
                 instruction_metrics,make_program_refs)

BIND_CONFIG={'width':32,'radius':1,'linear':True,'ignore_prefix':False}

def state_digest(model):
    h=hashlib.sha256()
    for name,tensor in sorted(model.state_dict().items()):
        h.update(name.encode());h.update(str(tensor.dtype).encode());h.update(str(tuple(tensor.shape)).encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def fit(args):
    start=time.time();torch.set_num_threads(2)
    ref,data,programs,banks,refs=load_train_dev(args.features)
    base=torch.load(args.base_checkpoint,map_location='cpu',weights_only=True)
    if base['reference_sha256']!=REFERENCE_SHA256 or base.get('training_texts_variant','legacy')!='legacy':
        raise ValueError('Only a clean legacy-v8 base is accepted')
    if base['model']!=data['model'] or base['backbone_revision']!=data['backbone_revision']:
        raise ValueError('Base/feature backbone mismatch')
    if any(base['dataset_hashes'][k]!=v for k,v in data['dataset_hashes'].items()):
        raise ValueError('Base/feature dataset mismatch')
    model=TypedSemanticReader(ref,data['hidden_size'],base['state_dict']['op.proto_layers'],
                              base['state_dict']['rel.proto_layers'],BIND_CONFIG)
    missing,unexpected=model.load_state_dict(base['state_dict'],strict=False)
    if unexpected or any(not x.startswith('bind_roles.') for x in missing):
        raise ValueError('Unexpected base state mismatch')
    for p in model.parameters():p.requires_grad_(False)
    torch.manual_seed(0)
    model.bind_roles=WindowRoleReader(ref,data['hidden_size'],**BIND_CONFIG)
    spec=ref.PhraseSpec();targets={}
    for row,r in zip(spec.train_examples,refs['isolated']):
        if row[1]!=ref.OID['BIND']:continue
        target=r.payload.index(row[2])*4+r.payload.index(row[4])
        if r.index in targets and targets[r.index]!=target:
            raise ValueError('Contradictory role supervision after lexical copying')
        targets[r.index]=target
    ids=torch.tensor(sorted(targets));y=torch.tensor([targets[int(i)] for i in ids])
    features=banks['train'].get(ids)
    opt=torch.optim.AdamW(model.bind_roles.parameters(),lr=2e-3,weight_decay=.01)
    history=[];model.train()
    for step in range(1,501):
        logits=model.bind_roles(*features)
        loss=torch.nn.functional.cross_entropy(logits.flatten(1),y)
        opt.zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_(model.bind_roles.parameters(),1.);opt.step()
        if step==1 or step%100==0:
            record={'stage':'bind_semantics','step':step,'loss':float(loss.detach())}
            history.append(record);print(json.dumps(record),flush=True)
    def evaluate():
        result={}
        for split in ('train','dev'):
            pred=predict_instructions(model,banks[split],[r for p in refs[split] for r in p],ref)
            result[split]=evaluate_predictions(pred,programs[split],ref)
        result['dev_phrase']=instruction_metrics(predict_instructions(model,banks['dev'],refs['dev_phrase'],ref),
                        [ref.Instr(*r[1:]) for r in spec.scan_examples],ref)
        return result
    before=evaluate()
    # Five genuinely final-answer-only epochs. No role/opcode targets, oracle
    # intermediate states, mix regularizer or auxiliary loss is used here.
    opt=torch.optim.SGD(model.bind_roles.parameters(),lr=1e-3)
    rng=random.Random(0)
    for epoch in range(1,6):
        order=list(range(len(programs['train'])));rng.shuffle(order)
        losses=[];hits=[];model.train()
        for offset in range(0,len(order),12):
            chosen=order[offset:offset+12]
            _,_,_,_,state=program_batch(model,banks['train'],programs['train'],refs['train'],chosen,ref)
            target=torch.tensor([programs['train'][i].answer for i in chosen])
            loss=final_loss(state,target)
            opt.zero_grad(set_to_none=True);loss.backward();opt.step()
            losses.append(float(loss.detach()))
            hits.extend(((state['current'].argmax(-1)==target)&(state['halted']>.5)).tolist())
        record={'stage':'final_only','epoch':epoch,'loss':float(np.mean(losses)),
                'accuracy':float(np.mean(hits)),'auxiliary_weight':0.}
        history.append(record);print(json.dumps(record),flush=True)
    after=evaluate();model.eval()
    metadata=dict(base)
    metadata.update(state_dict=model.state_dict(),model_kind='typed_local_bind',bind_config=BIND_CONFIG,
        training_texts_variant='legacy',protocol='FOG_LOGIC_V8_2_TYPED_LOCAL_BIND',
        base_checkpoint_sha256=hashlib.sha256(Path(args.base_checkpoint).read_bytes()).hexdigest(),
        bind_fit_seed=0,bind_fit_steps=500,final_only_epochs=5)
    Path(args.checkpoint).parent.mkdir(parents=True,exist_ok=True)
    torch.save(metadata,args.checkpoint)
    frozen_sha=hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest()
    frozen_state=state_digest(model)
    eligible=after['train']['accuracy']>=.99 and after['dev']['accuracy']>=.90 and after['dev']['bind_instruction_joint']>=.90
    out={'protocol':metadata['protocol'],'reference_sha256':REFERENCE_SHA256,
        'backbone':data['model'],'backbone_revision':data['backbone_revision'],
        'bind_config':BIND_CONFIG,'bind_canonical_training_examples':len(targets),
        'training_texts_variant':'legacy','additional_training_texts':0,
        'dataset_hashes':base['dataset_hashes'],'phrase_manifest':spec.manifest(),
        'base_checkpoint_sha256':metadata['base_checkpoint_sha256'],'feature_cache_sha256':hashlib.sha256(Path(args.features).read_bytes()).hexdigest(),
        'checkpoint_sha256':frozen_sha,'state_dict_sha256':frozen_state,
        'adapter_total_parameters':sum(p.numel() for p in model.parameters()),
        'new_bind_parameters':sum(p.numel() for p in model.bind_roles.parameters()),
        'before_final_only':before,'after_final_only':after,'train_history':history,
        'dev_eligible':eligible,'locked_test_evaluated':False,'config':vars(args)}
    print(json.dumps({'dev_eligible':eligible,'train':after['train'],'dev':after['dev'],'state_dict_sha256':frozen_state}),flush=True)
    if args.locked and eligible:
        # Nothing from the locked split is featurized until after checkpoint freeze.
        from transformers import AutoTokenizer,AutoModelForCausalLM
        tok=AutoTokenizer.from_pretrained(data['model'],revision=data['backbone_revision'],use_fast=True)
        tok.pad_token=tok.pad_token or tok.eos_token;tok.padding_side='right'
        net=AutoModelForCausalLM.from_pretrained(data['model'],revision=data['backbone_revision'],torch_dtype=torch.float32).eval()
        for p in net.parameters():p.requires_grad_(False)
        test=ref.generate_programs(ref.SEED+9000,(1,2,3,4,5,6,8),72,train=False)
        bank=TextBank(ref.ENTITIES)
        test_refs=make_program_refs(test,'test',bank,ref)
        phrase_refs=[bank.add(spec.texts[r[0]]) for r in spec.test_examples]
        bank.featurize(net,tok,torch.device('cpu'))
        pred=predict_instructions(model,bank,[r for p in test_refs for r in p],ref)
        normal=evaluate_predictions(pred,test,ref)
        phrases=instruction_metrics(predict_instructions(model,bank,phrase_refs,ref),[ref.Instr(*r[1:]) for r in spec.test_examples],ref)
        shuffled=evaluate_predictions(pred,test,ref,mode='full')
        oracle=evaluate_predictions([r for p in test for r in p.instructions],test,ref)
        out.update(locked_test_evaluated=True,locked_program_eval=normal,locked_phrase_metrics=phrases,
                   shuffle_full=shuffled,shuffle_within_opcode=evaluate_predictions(pred,test,ref,mode='within_opcode'),
                   oracle_program_eval=oracle,oracle_bind=evaluate_predictions(pred,test,ref,oracle_bind=True),
                   oracle_all_args=evaluate_predictions(pred,test,ref,oracle_fields=('e1','rel','e2')),
                   oracle_opcode=evaluate_predictions(pred,test,ref,oracle_fields=('op',)))
        out['gates']=strict_gates(after['train'],normal,phrases,shuffled,oracle,ref.benchmark_baselines(test)['majority_answer_accuracy'])
        out['verdict']='PASS_LOCKED_DOMAIN_GATES' if out['gates']['passed'] else 'NOT_ALL_LOCKED_GATES_PASSED'
        print(json.dumps({'locked':normal,'gates':out['gates']}),flush=True)
    else:
        out['verdict']='DEV_QUALIFIED_TEST_NOT_RUN' if eligible else 'DEV_GATE_FAILED'
    if hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest()!=frozen_sha or state_digest(model)!=frozen_state:
        raise AssertionError('Checkpoint or model mutated after freeze')
    out['runtime_seconds']=time.time()-start
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    Path(args.output).write_text(json.dumps(out,indent=2))
    return out

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-checkpoint',required=True);p.add_argument('--features',required=True)
    p.add_argument('--checkpoint',default='fog_logic_v8_2.pt');p.add_argument('--output',default='typed-results.json')
    p.add_argument('--locked',action='store_true')
    fit(p.parse_args())
