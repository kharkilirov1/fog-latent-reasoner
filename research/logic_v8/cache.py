"""Verified train/dev-only frozen feature cache; never a replacement for unseen input encoding."""
from pathlib import Path
import torch
from integrity import REFERENCE_SHA256, load_reference
from model import LAYERS, TextBank, TextRef
from run import dataset_witness,make_program_refs


def load_train_dev(path, device='cpu'):
    ref=load_reference()
    data=torch.load(path,map_location=device,weights_only=True)
    if data.get('locked_test_included') is not False or data.get('reference_sha256') != REFERENCE_SHA256:
        raise ValueError('Feature provenance does not match the train/dev-only reference')
    if tuple(data['layers']) != LAYERS:
        raise ValueError('Feature layers do not match')
    programs={'train':ref.generate_programs(ref.SEED+100,(1,2,3,4),72,train=True),
              'dev':ref.generate_programs(ref.SEED+200,(1,2,3,4),24,train=True)}
    if data['dataset_hashes'] != {k:dataset_witness(v) for k,v in programs.items()}:
        raise ValueError('Dataset hash mismatch')
    banks={}
    for split in ('train','dev'):
        raw=data[split+'_bank'];bank=TextBank(ref.ENTITIES)
        bank.texts=raw['texts'];bank.spans=raw['spans'];bank.index={s:i for i,s in enumerate(bank.texts)}
        for key in ('features','mask','mentions','valid'):
            setattr(bank,key,raw[key].to(device))
        banks[split]=bank
    refs={}
    for split in ('train','dev'):
        expected=make_program_refs(programs[split],'train' if split=='train' else 'scan',banks[split],ref)
        stored=[[TextRef(x['index'],tuple(x['payload'])) for x in row] for row in data[split+'_refs']]
        if expected != stored:raise ValueError('Text reference mismatch')
        refs[split]=stored
    refs['isolated']=[TextRef(x['index'],tuple(x['payload'])) for x in data['isolated_refs']]
    refs['dev_phrase']=[TextRef(x['index'],tuple(x['payload'])) for x in data['dev_phrase_refs']]
    return ref,data,programs,banks,refs
