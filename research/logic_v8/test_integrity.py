from __future__ import annotations
import copy
from pathlib import Path
import sys
import pytest
import torch
import torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).resolve().parent))
from integrity import (active_mask, audit_training, derangement, forbidden_holdout,
                       LexicalSlots, load_reference, select_device, strict_gates)
from model import (LAYERS, SemanticReader, TextBank, TextRef, auxiliary_loss,
                   execute_tensor, final_loss, hard_arguments, slot_payload, st_onehot)
from run import evaluate_predictions, make_program_refs, program_batch

@pytest.fixture(scope='module')
def ref():
    torch.set_num_threads(2)
    return load_reference()

@pytest.fixture(scope='module')
def programs(ref):
    return (ref.generate_programs(ref.SEED+100, (1,2,3,4), 72, train=True),
            ref.generate_programs(ref.SEED+200, (1,2,3,4), 24, train=True),
            ref.generate_programs(ref.SEED+9000, (1,2,3,4,5,6,8), 72, train=False))

def test_reference_manifest_unchanged(ref):
    manifest = ref.PhraseSpec().manifest()
    assert manifest['n_texts'] == 2420
    assert manifest['sha256'] == '4e4697299e47b99f7408815509870a184cc8899a3327a01231545541bb6270aa'

def test_exact_leak_and_masks(ref, programs):
    audit = audit_training(programs[0], ref)
    assert audit['unmasked_holdout_targets'] == 51
    assert audit['masked_holdout_targets'] == 0
    assert audit['post_halt_targets_excluded'] == 576
    for p in programs[0]:
        use = active_mask(p.instructions, ref.OID['HALT'])
        assert sum(use) == len(p.instructions)-2
        assert not any(forbidden_holdout(ins, ref) for ins, used in zip(p.instructions, use) if used)

def test_first_halt_included(ref):
    ins = [ref.Instr(1), ref.Instr(5), ref.Instr(2), ref.Instr(5)]
    assert active_mask(ins, 5) == [True, True, False, False]
    assert active_mask([], 5) == []

@pytest.mark.parametrize('n', [0,1,2,3,4,5,7,12,97,1000])
def test_derangement(n):
    for seed in range(12):
        mp = derangement(range(n), seed)
        assert set(mp) == set(mp.values()) == set(range(n))
        assert n < 2 or all(i != j for i,j in mp.items())
        assert mp == derangement(range(n), seed)

def test_lexical_copy_all_occurrences(ref):
    lexical = LexicalSlots(ref.ENTITIES)
    text, payload, spans = lexical.encode('Choose between Iris and Lena: true takes Iris.')
    assert text == 'Choose between Alice and Bob: true takes Alice.'
    assert payload == (ref.EID['Iris'], ref.EID['Lena'])
    assert len(spans[0]) == 2 and len(spans[1]) == 1
    for slot, occurrences in enumerate(spans):
        for a,b in occurrences:
            assert text[a:b] == ref.ENTITIES[slot]

def test_lexical_boundaries_and_self_loop(ref):
    lexical = LexicalSlots(ref.ENTITIES)
    assert lexical.encode('Malice Bobcat AliceBob Bob_Alice')[1] == ()
    text, payload, spans = lexical.encode('Bob manages Bob.')
    assert text == 'Alice manages Alice.' and payload == (1,) and len(spans[0]) == 2
    assert lexical.encode("Iris's manager is Lena.")[0] == "Alice's manager is Bob."

def test_canonical_semantics_not_role_oracle(ref):
    lexical = LexicalSlots(ref.ENTITIES)
    assert lexical.encode('Bob manages Alice.')[0] == 'Alice manages Bob.'
    assert lexical.encode('Bob manages Alice.')[1] == (1,0)
    assert lexical.encode('Alice reports to Bob.')[1] == (0,1)

def test_copy_equivariance_and_consistent_joint_map(ref):
    op = torch.randn(2, ref.O); relation = torch.randn(2, ref.R)
    pair = torch.full((2,4,4), -100.)
    pair[:,0,1] = 4.; pair[:,1,0] = 3.9
    payload = slot_payload([TextRef(0,(0,1)), TextRef(0,(8,10))],ref.E,'cpu')
    args = hard_arguments(op,relation,pair,payload)
    assert args[1].argmax(-1).tolist() == [0,8]
    assert args[3].argmax(-1).tolist() == [1,10]

def test_st_exact_and_finite_gradients():
    x = torch.randn(10,12,requires_grad=True)
    y = st_onehot(x)
    assert set(y.detach().unique().tolist()) <= {0.,1.}
    assert torch.equal(y.sum(-1), torch.ones(10))
    (y*torch.randn_like(y)).sum().backward()
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum()>0

def test_tensor_oracle_all_888(ref, programs):
    all_programs = sum((list(x) for x in programs), [])
    result = evaluate_predictions([i for p in all_programs for i in p.instructions], all_programs, ref)
    assert result['correct'] == result['count'] == 888
    assert result['programs_with_invalid_reads'] == 0
    assert all(v == 1 for v in result['by_chain'].values())

def test_tensor_matches_reference_every_state(ref, programs):
    for p in programs[2][::43]:
        args = []
        for attr,n in [('op',ref.O),('e1',ref.E),('rel',ref.R),('e2',ref.E)]:
            args.append(F.one_hot(torch.tensor([max(getattr(i,attr),0) for i in p.instructions]),n).float()[None])
        actual = execute_tensor(*args,return_trace=True)['trace']
        expected = ref.oracle_trajectory(p,torch.device('cpu'))
        for a,e in zip(actual,expected):
            for at,et in zip(a,e):
                assert torch.equal(at[0],et)

def test_post_halt_is_immutable_and_undefined_read_visible(ref):
    p = ref.LogicProgram((ref.Instr(1,2),ref.Instr(2,rel=1),ref.Instr(5),ref.Instr(0,2,1,5),ref.Instr(1,7)),2,1,False)
    args = [F.one_hot(torch.tensor([max(getattr(x,attr),0) for x in p.instructions]),n).float()[None]
            for attr,n in [('op',6),('e1',12),('rel',6),('e2',12)]]
    out = execute_tensor(*args,return_trace=True)
    assert out['current'].argmax() == 2 and out['invalid_reads'].item() == 1
    assert out['memory'].sum() == 0 and out['halted'].item() == 1
    for a,b in zip(out['trace'][2],out['trace'][-1]): assert torch.equal(a,b)

def test_final_loss_reaches_wrong_bind_argument(ref):
    rows = [ref.Instr(0,0,0,1),ref.Instr(1,0),ref.Instr(2,rel=0),ref.Instr(5)]
    logits = []
    for attr,n in [('op',6),('e1',12),('rel',6),('e2',12)]:
        x = torch.full((1,4,n),-2.)
        for i,ins in enumerate(rows): x[0,i,max(getattr(ins,attr),0)] = 2.
        if attr == 'e2': x[0,0,1] = -2.; x[0,0,2] = 2.
        logits.append(x.requires_grad_())
    out = execute_tensor(*(st_onehot(x) for x in logits))
    final_loss(out,torch.tensor([1])).backward()
    assert logits[3].grad[0,0].abs().sum()>0
    assert all(torch.isfinite(x.grad).all() for x in logits)

def passing_gate_inputs():
    return ({'accuracy':1.}, {'accuracy':.8,'heldout_accuracy':.75,'bind_instruction_joint':.95,'instruction_joint':.95},
            {'heldout':{'compare_kai':{'joint':1.},'follow_calls':{'joint':1.},'select_pair':{'joint':1.}}},
            {'accuracy':.1}, {'accuracy':1.}, 1/12)

@pytest.mark.parametrize('key', ['accuracy','heldout_accuracy','bind_instruction_joint','instruction_joint'])
def test_no_false_pass_for_missing_locked_gates(key):
    args = list(copy.deepcopy(passing_gate_inputs()))
    args[1][key] = .65 if key == 'heldout_accuracy' else .1
    assert not strict_gates(*args)['passed']

def test_all_gates_and_null_metrics():
    args = list(copy.deepcopy(passing_gate_inputs()))
    assert strict_gates(*args)['passed']
    assert len(strict_gates(*args)['criteria']) == 11
    args[2]['heldout']['follow_calls']['joint'] = None
    assert not strict_gates(*args)['passed']
    args[2]['heldout']['follow_calls']['joint'] = float('nan')
    assert not strict_gates(*args)['passed']

def fake_bank(ref):
    bank = TextBank(ref.ENTITIES)
    refs = [bank.add('Alice manages Bob.'),bank.add('Focus on Alice.'),bank.add('Follow calls.'),bank.add('Stop.')]
    torch.manual_seed(0)
    bank.features = torch.randn(4,len(LAYERS),10,16)
    bank.mask = torch.ones(4,10,dtype=torch.bool)
    bank.mentions = torch.zeros(4,4,10,dtype=torch.bool)
    bank.valid = torch.zeros(4,4,dtype=torch.bool)
    for i,r in enumerate(refs):
        for j in range(len(r.payload)):
            bank.mentions[i,j,2+j*3] = True; bank.valid[i,j] = True
    return bank,refs

def test_reader_masks_and_train_backward(ref):
    bank,refs = fake_bank(ref)
    model = SemanticReader(ref,16,torch.randn(6,3,len(LAYERS),32),torch.randn(6,3,len(LAYERS),16))
    op,rel,pair = model(*bank.get(torch.arange(4)))
    assert all(torch.isfinite(x).all() for x in (op,rel,pair))
    assert pair[0,2:].max() < -1000 and pair[0,:,2:].max() < -1000
    rows = [ref.Instr(0,1,0,0),ref.Instr(1,0),ref.Instr(2,rel=0),ref.Instr(5)]
    loss = auxiliary_loss((op,rel,pair),rows,refs,[True,True,False,True],ref)
    loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())

def test_training_forward_excludes_tail_even_for_wrong_halt(ref):
    bank,refs = fake_bank(ref)
    model = SemanticReader(ref,16,torch.randn(6,3,len(LAYERS),32),torch.randn(6,3,len(LAYERS),16))
    p = ref.LogicProgram((ref.Instr(1,0),ref.Instr(5),ref.Instr(2,rel=4)),0,1,False)
    _,seen_refs,rows,use,_ = program_batch(model,bank,[p],[[refs[1],refs[3],refs[2]]],[0],ref)
    assert len(rows) == len(seen_refs) == 2 and all(use)
    assert not any(forbidden_holdout(x,ref) for x in rows)

def test_training_loss_refuses_holdout(ref):
    bank,refs = fake_bank(ref)
    model = SemanticReader(ref,16,torch.randn(6,3,len(LAYERS),32),torch.randn(6,3,len(LAYERS),16))
    logits = model(*bank.get(torch.arange(4)))
    rows = [ref.Instr(5),ref.Instr(5),ref.Instr(2,rel=4),ref.Instr(5)]
    with pytest.raises(AssertionError): auxiliary_loss(logits,rows,refs,[True]*4,ref)

def test_device_no_silent_fallback():
    assert select_device('cpu').type == 'cpu'
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError): select_device('cuda')

def test_bank_reduces_only_names_not_language(ref, programs):
    bank = TextBank(ref.ENTITIES)
    make_program_refs(programs[0],'train',bank,ref)
    assert len(bank.texts) < 150
    before = len(bank.texts)
    bank.add('Alice manages Bob.'); bank.add('Iris manages Lena.')
    assert len(bank.texts) <= before+1
    with pytest.raises(ValueError): bank.add('')
