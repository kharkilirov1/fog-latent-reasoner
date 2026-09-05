"""Leak-free, copy-equivariant continuation of the frozen-Qwen Logic-v7 task.

The reference generator, phrase split, depths, seeds and thresholds are unchanged.
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
import torch.nn.functional as F
from integrity import (REFERENCE_SHA256, active_mask, audit_training, derangement,
                       load_reference, select_device, strict_gates)
from model import (LAYERS, SemanticReader, TextBank, auxiliary_loss, execute_tensor,
                   final_loss, forward_refs, hard_arguments, slot_payload)

def make_program_refs(programs, family, bank, ref):
    return [[bank.add(ref.instr_text(ins, family, (pi*11+si*5) % 8))
             for si, ins in enumerate(p.instructions)] for pi, p in enumerate(programs)]

def padded_arguments(arguments, lengths):
    maximum = max(lengths)
    output = []
    for x in arguments:
        chunks, offset = [], 0
        for length in lengths:
            chunks.append(F.pad(x[offset:offset+length], (0, 0, 0, maximum-length)))
            offset += length
        if offset != len(x):
            raise ValueError('Program lengths do not match arguments')
        output.append(torch.stack(chunks))
    return output

def program_batch(model, bank, programs, program_refs, ids, ref):
    # Crop ALL training computation, not only CE: a wrong predicted HALT must
    # not allow tail semantics to receive final-answer supervision either.
    lengths = [sum(active_mask(programs[i].instructions, ref.OID['HALT'])) for i in ids]
    refs = [r for i, n in zip(ids, lengths) for r in program_refs[i][:n]]
    rows = [r for i, n in zip(ids, lengths) for r in programs[i].instructions[:n]]
    use = [True]*len(rows)
    logits = forward_refs(model, bank, refs)
    payload = slot_payload(refs, ref.E, logits[0].device)
    arguments = hard_arguments(*logits, payload)
    state = execute_tensor(*padded_arguments(arguments, lengths))
    return logits, refs, rows, use, state

def train_isolated(model, bank, rows, text_refs, ref, steps, seed):
    rng = random.Random(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    model.train(); history = []
    for step in range(1, steps+1):
        idx = [rng.randrange(len(rows)) for _ in range(96)]
        batch_rows, batch_refs = [rows[i] for i in idx], [text_refs[i] for i in idx]
        logits = forward_refs(model, bank, batch_refs)
        loss = auxiliary_loss(logits, batch_rows, batch_refs, [True]*len(idx), ref)+.02*model.mix_reg()
        optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        if step == 1 or step % 200 == 0 or step == steps:
            rec = {'stage': 'isolated', 'step': step, 'loss': float(loss.detach())}
            history.append(rec); print(json.dumps(rec), flush=True)
    return history

def train_curriculum(model, bank, programs, text_refs, ref, seed):
    optimizer = torch.optim.AdamW(model.parameters(), lr=7e-4, weight_decay=1e-4)
    stages = [('R1', 1, 4, 1., .4, .8), ('R2', 2, 4, 1., .4, .8),
              ('R4', 4, 6, 1., .4, .8), ('semantic_anneal', 4, 8, .35, 0., 1.2),
              ('final_only', 4, 5, 0., 0., 1.5)]
    history = []
    for stage_no, (name, depth, epochs, aux_start, aux_end, fw) in enumerate(stages):
        rng = random.Random(seed+stage_no)
        selected = [i for i, p in enumerate(programs) if p.chain_len <= depth]
        for epoch in range(1, epochs+1):
            model.train(); rng.shuffle(selected); losses, hits = [], []
            aw = aux_start+(aux_end-aux_start)*(epoch-1)/max(epochs-1, 1)
            for start in range(0, len(selected), 12):
                ids = selected[start:start+12]
                logits, refs, rows, use, state = program_batch(model, bank, programs, text_refs, ids, ref)
                target = torch.tensor([programs[i].answer for i in ids], device=logits[0].device)
                loss = fw*final_loss(state, target)+.01*model.mix_reg()
                if aw:
                    loss = loss+aw*auxiliary_loss(logits, rows, refs, use, ref)
                optimizer.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step(); losses.append(float(loss.detach()))
                hit = (state['current'].argmax(-1) == target) & (state['halted'] > .5)
                hits.extend(hit.detach().cpu().tolist())
            rec = {'stage': name, 'epoch': epoch, 'loss': float(np.mean(losses)),
                   'hard_st_acc': float(np.mean(hits)), 'aux_weight': aw, 'final_weight': fw}
            history.append(rec); print(json.dumps(rec), flush=True)
    return history

@torch.inference_mode()
def predict_instructions(model, bank, text_refs, ref):
    model.eval()
    op, relation, pair = forward_refs(model, bank, text_refs)
    args = hard_arguments(op, relation, pair, slot_payload(text_refs, ref.E, op.device))
    decoded = [x.argmax(-1).cpu().tolist() for x in args]
    return [ref.Instr(*values) for values in zip(*decoded)]

def instruction_metrics(predicted, gold, ref):
    fields = {'opcode': [], 'e1': [], 'rel': [], 'e2': [], 'joint': []}
    byop = {name: [] for name in ref.OPCODES}
    held = {name: [] for name in ('follow_calls', 'compare_kai', 'select_pair')}
    for pred, ins in zip(predicted, gold):
        ok = pred.op == ins.op; fields['opcode'].append(ok)
        for key in ('e1', 'rel', 'e2'):
            if getattr(ins, key) >= 0:
                correct = getattr(ins, key) == getattr(pred, key)
                fields[key].append(correct); ok &= correct
        fields['joint'].append(ok); byop[ref.OPCODES[ins.op]].append(ok)
        if ins.op == ref.OID['FOLLOW'] and ins.rel == ref.RID['calls']:
            held['follow_calls'].append(ok)
        if ins.op == ref.OID['COMPARE'] and ins.e1 == ref.EID['Kai']:
            held['compare_kai'].append(ok)
        if ins.op == ref.OID['SELECT'] and (ins.e1, ins.e2) == (ref.EID['Iris'], ref.EID['Lena']):
            held['select_pair'].append(ok)
    def mean(x): return float(np.mean(x)) if x else None
    return {k: mean(v) for k, v in fields.items()} | {
        'heldout': {k: {'joint': mean(v), 'count': len(v)} for k, v in held.items()},
        'by_opcode': {k: {'joint': mean(v), 'count': len(v)} for k, v in byop.items()}}

@torch.inference_mode()
def evaluate_predictions(predicted, programs, ref, mode=None, oracle_fields=(), oracle_bind=False):
    gold = [ins for p in programs for ins in p.instructions]
    pred = list(predicted)
    if len(pred) != len(gold):
        raise ValueError('Wrong number of predicted instructions')
    if mode:
        if mode == 'full':
            mapping = derangement(range(len(gold)), ref.SEED+7007)
        elif mode == 'within_opcode':
            mapping = {}
            for op in range(ref.O):
                mapping.update(derangement([i for i, ins in enumerate(gold) if ins.op == op], ref.SEED+7007+op))
        else:
            raise ValueError(mode)
        pred = [pred[mapping[i]] for i in range(len(pred))]
    if oracle_fields or oracle_bind:
        for i, ins in enumerate(gold):
            names = ('op', 'e1', 'rel', 'e2')
            replace = names if oracle_bind and ins.op == ref.OID['BIND'] else oracle_fields
            vals = [getattr(ins, a) if a in replace and getattr(ins, a) >= 0 else getattr(pred[i], a) for a in names]
            pred[i] = ref.Instr(*vals)
    lengths = [len(p.instructions) for p in programs]
    args = []
    for attr, classes in (('op', ref.O), ('e1', ref.E), ('rel', ref.R), ('e2', ref.E)):
        labels = torch.tensor([max(getattr(ins, attr), 0) for ins in pred])
        args.append(F.one_hot(labels, classes).float())
    state = execute_tensor(*padded_arguments(args, lengths))
    hits = ((state['current'].argmax(-1) == torch.tensor([p.answer for p in programs])) & (state['halted'] > .5)).tolist()
    held = [hit for hit, p in zip(hits, programs) if p.contains_heldout]
    bydepth, bybind = {}, {}
    for hit, p in zip(hits, programs):
        bydepth.setdefault(str(p.chain_len), []).append(hit)
        bybind.setdefault(str(sum(x.op == ref.OID['BIND'] for x in p.instructions)), []).append(hit)
    im = instruction_metrics(pred, gold, ref)
    active = [x for p in programs for x in active_mask(p.instructions, ref.OID['HALT'])]
    aim = instruction_metrics([p for p, a in zip(pred, active) if a], [g for g, a in zip(gold, active) if a], ref)
    return {'accuracy': float(np.mean(hits)), 'heldout_accuracy': float(np.mean(held)) if held else None,
            'correct': sum(hits), 'count': len(hits), 'heldout_count': len(held),
            'instruction_joint': im['joint'], 'active_instruction_joint': aim['joint'],
            'bind_instruction_joint': im['by_opcode']['BIND']['joint'],
            'by_chain': {k: float(np.mean(v)) for k, v in bydepth.items()},
            'by_bind_count': {k: float(np.mean(v)) for k, v in bybind.items()},
            'programs_with_invalid_reads': int((state['invalid_reads'] > .5).sum()),
            'oracle_fields': list(oracle_fields), 'oracle_bind': oracle_bind, 'shuffle_mode': mode}

def dataset_witness(programs):
    data = [[[i.op, i.e1, i.rel, i.e2] for i in p.instructions] + [['answer', p.answer]] for p in programs]
    return hashlib.sha256(json.dumps(data, separators=(',', ':')).encode()).hexdigest()

def run(args):
    ref = load_reference()
    torch.set_num_threads(args.threads)
    train = ref.generate_programs(ref.SEED+100, (1, 2, 3, 4), 72, train=True)
    dev = ref.generate_programs(ref.SEED+200, (1, 2, 3, 4), 24, train=True)
    test = ref.generate_programs(ref.SEED+9000, (1, 2, 3, 4, 5, 6, 8), 72, train=False)
    audit = audit_training(train, ref)
    if audit['masked_holdout_targets']:
        raise AssertionError('Active training corpus violates holdouts')
    spec = ref.PhraseSpec()
    out = {'protocol': 'FOG_LOGIC_V8_COPY_EQUIVARIANT_JOINT_ROLES', 'base_commit': 'acc05871bfc5b135e8b5e39c86e8c43e8a56c05b',
           'reference_sha256': REFERENCE_SHA256, 'phrase_manifest': spec.manifest(), 'training_audit': audit,
           'dataset_hashes': {k: dataset_witness(v) for k, v in [('train', train), ('dev', dev), ('test', test)]},
           'program_counts': {'train': len(train), 'dev': len(dev), 'test': len(test)},
           'train_depths': [1, 2, 3, 4], 'test_depths': [1, 2, 3, 4, 5, 6, 8],
           'model_seed': args.seed, 'locked_test_evaluated': False,
           'training_view': 'prefix_through_first_gold_halt_for_all_losses', 'config': vars(args)}
    if args.audit_only:
        oracle = evaluate_predictions([x for p in train+dev+test for x in p.instructions], train+dev+test, ref)
        out['tensor_oracle'] = oracle
        Path(args.output).write_text(json.dumps(out, indent=2)); print(json.dumps(out, indent=2)); return out
    device = select_device(args.device)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if device.type == 'cuda': torch.cuda.manual_seed_all(args.seed)
    saved = torch.load(args.evaluate_checkpoint, map_location=device, weights_only=True) if args.evaluate_checkpoint else None
    if saved is not None:
        if saved['reference_sha256'] != REFERENCE_SHA256 or saved['dataset_hashes'] != out['dataset_hashes'] or saved['model'] != args.model:
            raise ValueError('Checkpoint provenance does not match this experiment')
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True, revision=saved.get('backbone_revision') if saved else None)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.padding_side = 'right'
    backbone = AutoModelForCausalLM.from_pretrained(args.model, revision=saved.get('backbone_revision') if saved else None, torch_dtype=torch.float32 if device.type == 'cpu' else torch.float16).to(device).eval()
    for p in backbone.parameters(): p.requires_grad_(False)
    bank = TextBank(ref.ENTITIES)
    isolated_refs = [bank.add(spec.texts[x[0]]) for x in spec.train_examples]
    isolated_rows = [ref.Instr(*x[1:]) for x in spec.train_examples]
    train_refs = make_program_refs(train, 'train', bank, ref)
    op_ids = [[bank.add(spec.texts[i]).index for i in ids] for ids in spec.op_anchor_ids]
    rel_ids = [[bank.add(spec.texts[i]).index for i in ids] for ids in spec.rel_anchor_ids]
    bank.featurize(backbone, tokenizer, device, args.qwen_batch)
    op_proto = ref.anchor_sentence_proto(bank.features, bank.mask, op_ids)
    rel_proto = ref.anchor_proto(bank.features, bank.mask, rel_ids)
    model = SemanticReader(ref, backbone.config.hidden_size, op_proto, rel_proto).to(device)
    out['canonical_train_texts'] = len(bank.texts)
    out['trainable_params'] = sum(p.numel() for p in model.parameters())
    out['backbone_revision'] = getattr(backbone.config, '_commit_hash', None)
    print(json.dumps({'canonical_train_texts': len(bank.texts), 'trainable_params': out['trainable_params'], 'device': str(device)}), flush=True)
    if saved is None:
        history = train_isolated(model, bank, isolated_rows, isolated_refs, ref, args.isolated_steps, args.seed)
        history += train_curriculum(model, bank, train, train_refs, ref, args.seed)
    else:
        model.load_state_dict(saved['state_dict'], strict=True)
        history = []
        out['evaluation_only'] = True
    out['train_history'] = history
    out['train_program_eval'] = evaluate_predictions(predict_instructions(model, bank, [x for p in train_refs for x in p], ref), train, ref)
    devbank = TextBank(ref.ENTITIES)
    dev_refs = make_program_refs(dev, 'scan', devbank, ref)
    dev_phrase_refs = [devbank.add(spec.texts[x[0]]) for x in spec.scan_examples]
    devbank.featurize(backbone, tokenizer, device, args.qwen_batch)
    out['dev_program_eval'] = evaluate_predictions(predict_instructions(model, devbank, [x for p in dev_refs for x in p], ref), dev, ref)
    out['dev_phrase_metrics'] = instruction_metrics(predict_instructions(model, devbank, dev_phrase_refs, ref), [ref.Instr(*x[1:]) for x in spec.scan_examples], ref)
    # Freeze and persist BEFORE any locked text is encoded or test score queried.
    model.eval()
    if saved is None:
        torch.save({'state_dict': model.state_dict(), 'hidden_size': backbone.config.hidden_size,
                    'model': args.model, 'backbone_revision': getattr(backbone.config, '_commit_hash', None),
                    'layers': LAYERS, 'reference_sha256': REFERENCE_SHA256,
                    'protocol': out['protocol'], 'seed': args.seed, 'dataset_hashes': out['dataset_hashes']}, args.checkpoint)
    out['checkpoint_sha256'] = hashlib.sha256(Path(args.evaluate_checkpoint or args.checkpoint).read_bytes()).hexdigest()
    print(json.dumps({'train': out['train_program_eval'], 'dev': out['dev_program_eval']}), flush=True)
    if args.locked:
        tb = TextBank(ref.ENTITIES)
        test_refs = make_program_refs(test, 'test', tb, ref)
        phrase_refs = [tb.add(spec.texts[x[0]]) for x in spec.test_examples]
        tb.featurize(backbone, tokenizer, device, args.qwen_batch)
        pred = predict_instructions(model, tb, [x for p in test_refs for x in p], ref)
        out['locked_program_eval'] = evaluate_predictions(pred, test, ref)
        out['locked_phrase_metrics'] = instruction_metrics(predict_instructions(model, tb, phrase_refs, ref), [ref.Instr(*x[1:]) for x in spec.test_examples], ref)
        out['shuffle_full'] = evaluate_predictions(pred, test, ref, mode='full')
        out['shuffle_within_opcode'] = evaluate_predictions(pred, test, ref, mode='within_opcode')
        out['oracle_program_eval'] = evaluate_predictions([x for p in test for x in p.instructions], test, ref)
        out['oracle_bind'] = evaluate_predictions(pred, test, ref, oracle_bind=True)
        out['oracle_all_args'] = evaluate_predictions(pred, test, ref, oracle_fields=('e1', 'rel', 'e2'))
        out['oracle_opcode'] = evaluate_predictions(pred, test, ref, oracle_fields=('op',))
        out['gates'] = strict_gates(out['train_program_eval'], out['locked_program_eval'], out['locked_phrase_metrics'], out['shuffle_full'], out['oracle_program_eval'], ref.benchmark_baselines(test)['majority_answer_accuracy'])
        out['locked_test_evaluated'] = True
        out['verdict'] = 'PASS_LOCKED_DOMAIN_GATES' if out['gates']['passed'] else 'NOT_ALL_LOCKED_GATES_PASSED'
        print(json.dumps({'locked': out['locked_program_eval'], 'gates': out['gates']}), flush=True)
    else:
        out['verdict'] = 'DEV_ONLY_NOT_A_LOCKED_RESULT'
    out['runtime_env'] = {'torch': torch.__version__, 'device': str(device), 'frozen_backbone': True}
    Path(args.output).write_text(json.dumps(out, indent=2))
    return out

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-only', action='store_true')
    parser.add_argument('--locked', action='store_true', help='Evaluate locked test once, after checkpoint freeze')
    parser.add_argument('--model', default='Qwen/Qwen2.5-0.5B-Instruct')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--isolated-steps', type=int, default=1600)
    parser.add_argument('--qwen-batch', type=int, default=12)
    parser.add_argument('--output', default='fog_logic_v8_results.json')
    parser.add_argument('--checkpoint', default='fog_logic_v8.pt')
    parser.add_argument('--evaluate-checkpoint', help='Load frozen weights, do no training, optionally evaluate --locked')
    args = parser.parse_args()
    start = time.time(); run(args)
    print(json.dumps({'runtime_seconds': time.time()-start}), flush=True)

if __name__ == '__main__':
    main()
