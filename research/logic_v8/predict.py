"""Run an independently supplied text program with a frozen Logic-v8 checkpoint.

Input: a JSON list of instruction strings, or one instruction per nonempty line.
No opcode labels, argument labels, reference program or expected answer are used.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from typing import Sequence
from types import SimpleNamespace
import torch
from integrity import REFERENCE_SHA256, load_reference, select_device
from model import (LAYERS, SemanticReader, TextBank, execute_tensor,
                   forward_refs, hard_arguments, slot_payload)

def load_adapter(path: str | Path, device):
    ref = load_reference()
    ckpt = torch.load(path, map_location=device, weights_only=True)
    if ckpt.get('reference_sha256') != REFERENCE_SHA256:
        raise ValueError('Checkpoint has an incompatible reference protocol')
    if tuple(ckpt.get('layers', ())) != LAYERS:
        raise ValueError('Checkpoint feature layers do not match the implementation')
    state = ckpt['state_dict']
    kind = ckpt.get('model_kind', 'joint')
    if kind == 'typed_local_bind':
        from window_roles import TypedSemanticReader
        model = TypedSemanticReader(ref, ckpt['hidden_size'], state['op.proto_layers'], state['rel.proto_layers'], ckpt['bind_config']).to(device)
    elif kind == 'joint':
        model = SemanticReader(ref, ckpt['hidden_size'], state['op.proto_layers'], state['rel.proto_layers']).to(device)
    else:
        raise ValueError(f'Unsupported model kind: {kind}')
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, ckpt, ref

def parse_input(path: str | Path) -> list[str]:
    text = Path(path).read_text(encoding='utf-8')
    if text.lstrip().startswith('['):
        rows = json.loads(text)
        if not isinstance(rows, list):
            raise ValueError('Expected a JSON list of instruction strings')
    else:
        rows = [x.strip() for x in text.splitlines() if x.strip()]
    if not rows or len(rows) > 512:
        raise ValueError('Supply between 1 and 512 instructions')
    if any(not isinstance(x, str) or not x.strip() for x in rows):
        raise ValueError('Every instruction must be a nonempty string')
    return rows

@torch.inference_mode()
def infer_bank(model, bank, text_refs, input_texts, ref, *, trace=False):
    """Inference for an already encoded bank, also usable in offline tests."""
    if not text_refs or len(text_refs) != len(input_texts):
        raise ValueError('Input text and reference counts must match and be nonzero')
    logits = forward_refs(model, bank, text_refs)
    payload = slot_payload(text_refs, ref.E, logits[0].device)
    arguments = hard_arguments(*logits, payload)
    result = execute_tensor(*(x[None] for x in arguments), return_trace=trace)
    decoded = [x.argmax(-1).cpu().tolist() for x in arguments]
    halted = float(result['halted'][0]) > .5
    invalid = int(round(float(result['invalid_reads'][0])))
    failures, records, active = [], [], True
    required_entities = {ref.OID['BIND']:True, ref.OID['LOAD']:True,
                         ref.OID['COMPARE']:True, ref.OID['SELECT']:True}
    for i, (op,a,r,b) in enumerate(zip(*decoded)):
        if active and required_entities.get(op, False) and not text_refs[i].payload:
            failures.append({'step':i, 'reason':'Predicted argument-bearing operation has no recognized entity mention'})
        if trace:
            state = result['trace'][i]
            record = {'step':i, 'text':input_texts[i], 'executed':active,
                      'predicted_opcode':ref.OPCODES[op],
                      'entity_after':ref.ENTITIES[int(state[0][0].argmax())],
                      'predicate_after':bool(float(state[2][0]) > .5),
                      'halted_after':bool(float(state[3][0]) > .5)}
            if op in (0,1,3,4): record['e1'] = ref.ENTITIES[a]
            if op in (0,2): record['relation'] = ref.RELATIONS[r]
            if op in (0,4): record['e2'] = ref.ENTITIES[b]
            records.append(record)
        if op == ref.OID['HALT']: active = False
    status = 'EXECUTED_UNVERIFIED'
    if failures: status = 'INVALID_ENTITY_ARGUMENT'
    elif not halted: status = 'NO_HALT'
    elif invalid: status = 'UNDEFINED_RELATION_READ'
    candidate = ref.ENTITIES[int(result['current'][0].argmax())]
    return {'status':status,
            'answer':candidate if status == 'EXECUTED_UNVERIFIED' else None,
            'candidate_entity':candidate, 'halted':halted, 'undefined_reads':invalid,
            'structural_errors':failures, 'trace':records if trace else None,
            'scope':'Closed-domain learned instruction interpreter; execution is not a proof of semantic correctness.'}

def runtime_vocabulary(ref, entities=None):
    if entities is None: return ref
    if not 1 <= len(entities) <= 256:
        raise ValueError('Declare between 1 and 256 entity names')
    if any(not isinstance(x, str) or not x.strip() for x in entities) or len(set(entities)) != len(entities):
        raise ValueError('Entity names must be nonempty unique strings')
    values = {k:getattr(ref,k) for k in ('OPCODES','OID','RELATIONS','RID','R','O')}
    values.update(ENTITIES=tuple(entities), E=len(entities), EID={x:i for i,x in enumerate(entities)})
    return SimpleNamespace(**values)

def predict(checkpoint, input_texts: Sequence[str], device='auto', *, trace=False, threads=2, entities=None):
    torch.set_num_threads(threads)
    device = select_device(device)
    model, meta, ref = load_adapter(checkpoint, device)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    revision = meta.get('backbone_revision')
    tok = AutoTokenizer.from_pretrained(meta['model'], revision=revision, use_fast=True)
    tok.pad_token = tok.pad_token or tok.eos_token
    tok.padding_side = 'right'
    backbone = AutoModelForCausalLM.from_pretrained(meta['model'], revision=revision,
             torch_dtype=torch.float32 if device.type == 'cpu' else torch.float16).to(device).eval()
    for param in backbone.parameters(): param.requires_grad_(False)
    runtime_ref = runtime_vocabulary(ref, entities)
    bank = TextBank(runtime_ref.ENTITIES, canonical_names=ref.ENTITIES)
    refs = [bank.add(s) for s in input_texts]
    bank.featurize(backbone,tok,device)
    output = infer_bank(model,bank,refs,input_texts,runtime_ref,trace=trace)
    output.update({'checkpoint_sha256':hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
                   'backbone':meta['model'], 'backbone_revision':revision,
                   'supported_entities':list(runtime_ref.ENTITIES), 'device':str(device)})
    return output

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--input',required=True,help='JSON list or text with one instruction per line')
    p.add_argument('--output',help='Optional JSON output file')
    p.add_argument('--device',default='auto')
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--trace',action='store_true')
    p.add_argument('--entities',help='Optional JSON list of 1..256 declared entity names; no retraining needed')
    a = p.parse_args()
    entities = json.loads(Path(a.entities).read_text()) if a.entities else None
    if entities is not None and not isinstance(entities,list):
        raise ValueError('--entities must contain a JSON list')
    output = predict(a.checkpoint,parse_input(a.input),a.device,trace=a.trace,threads=a.threads,entities=entities)
    text = json.dumps(output,indent=2,ensure_ascii=False)
    if a.output: Path(a.output).write_text(text,encoding='utf-8')
    print(text)

if __name__ == '__main__': main()
