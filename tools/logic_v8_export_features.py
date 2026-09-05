"""Export TRAIN/DEV frozen features for offline reproducible adapter experiments.

No locked-test text is encoded and no locked-test label is exported.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'research'/'logic_v8'))
from integrity import REFERENCE_SHA256, load_reference
from model import LAYERS, TextBank
from run import make_program_refs, dataset_witness

def encode_bank(bank):
    n = int(bank.mask.sum(-1).max())
    return {'texts':bank.texts,'spans':bank.spans,'features':bank.features[:,:,:n].cpu().contiguous(),
            'mask':bank.mask[:,:n].cpu().contiguous(),'mentions':bank.mentions[:,:,:n].cpu().contiguous(),
            'valid':bank.valid.cpu().contiguous()}

def refs_json(rows):
    return [{'index':r.index,'payload':list(r.payload)} for r in rows]

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',default='logic-v8-features.pt');a=p.parse_args()
    torch.set_num_threads(2)
    v=load_reference();s=v.PhraseSpec()
    from transformers import AutoTokenizer,AutoModelForCausalLM
    name='Qwen/Qwen2.5-0.5B-Instruct'
    tok=AutoTokenizer.from_pretrained(name,use_fast=True);tok.pad_token=tok.pad_token or tok.eos_token;tok.padding_side='right'
    net=AutoModelForCausalLM.from_pretrained(name,torch_dtype=torch.float32).eval()
    for x in net.parameters():x.requires_grad_(False)
    train=v.generate_programs(v.SEED+100,(1,2,3,4),72,train=True)
    dev=v.generate_programs(v.SEED+200,(1,2,3,4),24,train=True)
    bank=TextBank(v.ENTITIES)
    isolated=[bank.add(s.texts[r[0]]) for r in s.train_examples]
    trainrefs=make_program_refs(train,'train',bank,v)
    opids=[[bank.add(s.texts[i]).index for i in row] for row in s.op_anchor_ids]
    relids=[[bank.add(s.texts[i]).index for i in row] for row in s.rel_anchor_ids]
    bank.featurize(net,tok,torch.device('cpu'))
    db=TextBank(v.ENTITIES)
    devrefs=make_program_refs(dev,'scan',db,v)
    phrase=[db.add(s.texts[r[0]]) for r in s.scan_examples]
    db.featurize(net,tok,torch.device('cpu'))
    out={'model':name,'backbone_revision':getattr(net.config,'_commit_hash',None),'hidden_size':net.config.hidden_size,
         'layers':list(LAYERS),'reference_sha256':REFERENCE_SHA256,'locked_test_included':False,
         'dataset_hashes':{'train':dataset_witness(train),'dev':dataset_witness(dev)},
         'train_bank':encode_bank(bank),'dev_bank':encode_bank(db),
         'isolated_refs':refs_json(isolated),'train_refs':[refs_json(x) for x in trainrefs],
         'dev_refs':[refs_json(x) for x in devrefs],'dev_phrase_refs':refs_json(phrase),
         'op_anchor_ids':opids,'rel_anchor_ids':relids}
    torch.save(out,a.output)
    print(json.dumps({'file':a.output,'bytes':Path(a.output).stat().st_size,'train_canonical':len(bank.texts),
                      'dev_canonical':len(db.texts),'backbone_revision':out['backbone_revision'],
                      'sha256':hashlib.sha256(Path(a.output).read_bytes()).hexdigest(),'locked_test_included':False}),flush=True)
if __name__=='__main__':main()
