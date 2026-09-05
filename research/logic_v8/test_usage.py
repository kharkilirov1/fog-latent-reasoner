from pathlib import Path
import sys
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
from integrity import load_reference, REFERENCE_SHA256, forbidden_holdout
from model import LAYERS, SemanticReader
from predict import load_adapter,parse_input,infer_bank
from test_integrity import fake_bank

def test_checkpoint_roundtrip_without_qwen():
    import tempfile
    ref=load_reference();torch.set_num_threads(2)
    bank,refs=fake_bank(ref)
    model=SemanticReader(ref,16,torch.randn(6,3,len(LAYERS),32),torch.randn(6,3,len(LAYERS),16)).eval()
    with tempfile.TemporaryDirectory() as folder:
        path=Path(folder)/'adapter.pt'
        torch.save({'state_dict':model.state_dict(),'hidden_size':16,'reference_sha256':REFERENCE_SHA256,'layers':LAYERS},path)
        loaded,meta,_=load_adapter(path,torch.device('cpu'))
        with torch.no_grad():
            for a,b in zip(model(*bank.get(torch.arange(4))),loaded(*bank.get(torch.arange(4)))):
                assert torch.equal(a,b)

def test_inference_no_gold_and_missing_halt():
    ref=load_reference();bank,refs=fake_bank(ref)
    model=SemanticReader(ref,16,torch.randn(6,3,len(LAYERS),32),torch.randn(6,3,len(LAYERS),16))
    def fixed_forward(f,mask,mentions,valid):
        n=len(f);op=torch.full((n,6),-100.);op[:,1]=100.
        rel=torch.zeros(n,6);pair=torch.full((n,4,4),-100.);pair[:,0,0]=100.
        return op,rel,pair
    model.forward=fixed_forward
    out=infer_bank(model,bank,[refs[1]],['Focus on Alice.'],ref,trace=True)
    assert out['status']=='NO_HALT' and out['answer'] is None
    out=infer_bank(model,bank,[refs[2]],['Follow calls.'],ref)
    assert out['status']=='INVALID_ENTITY_ARGUMENT'

def test_input_file_validation(tmp_path):
    path=tmp_path/'input.txt';path.write_text('Focus on Alice.\n\nStop.\n')
    assert parse_input(path)==['Focus on Alice.','Stop.']
    path.write_text('["Focus on Alice.","Stop."]');assert len(parse_input(path))==2
    path.write_text('["Focus on Alice.",123]')
    with pytest.raises(ValueError):parse_input(path)
    path.write_text('')
    with pytest.raises(ValueError):parse_input(path)

def test_declared_training_repair_no_holdout_or_test_mutation():
    from training_texts import augment_declared,declared_training_texts
    ref=load_reference();spec=ref.PhraseSpec()
    programs=ref.generate_programs(ref.SEED+100,(1,2,3,4),72,train=True)
    before=[(spec.texts[r[0]],r[1:]) for r in spec.test_examples];baseline=len(spec.train_examples)
    info=augment_declared(spec,ref,programs)
    assert len(spec.train_examples)>baseline
    assert not any(forbidden_holdout(ref.Instr(*r[1:]),ref) for r in spec.train_examples)
    assert before==[(spec.texts[r[0]],r[1:]) for r in spec.test_examples]
    texts=declared_training_texts(ref.Instr(0,0,ref.RID['parent'],1),ref)
    assert len(texts)==10
    assert 'Bob is the parent of Alice.' in texts
    assert 'Register Bob as the parent target of Alice.' in texts
    with pytest.raises(ValueError):declared_training_texts(ref.Instr(2,rel=4),ref)
    assert info['added_test_templates']==0
