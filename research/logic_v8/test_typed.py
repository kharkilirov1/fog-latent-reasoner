from pathlib import Path
import sys
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
from integrity import load_reference,REFERENCE_SHA256
from model import LAYERS,TextBank,TextRef,slot_payload,hard_arguments
from predict import load_adapter,runtime_vocabulary
from window_roles import WindowRoleReader,TypedSemanticReader
from test_integrity import fake_bank


def test_typed_checkpoint_roundtrip(tmp_path):
    ref=load_reference();torch.set_num_threads(2)
    bank,_=fake_bank(ref)
    config={'width':32,'radius':1,'linear':True,'ignore_prefix':False}
    model=TypedSemanticReader(ref,16,torch.randn(6,3,5,32),torch.randn(6,3,5,16),config).eval()
    path=tmp_path/'typed.pt'
    torch.save({'state_dict':model.state_dict(),'hidden_size':16,'reference_sha256':REFERENCE_SHA256,
                'layers':LAYERS,'model_kind':'typed_local_bind','bind_config':config},path)
    actual,_,_=load_adapter(path,'cpu')
    with torch.no_grad():
        for a,b in zip(model(*bank.get(torch.arange(4))),actual(*bank.get(torch.arange(4)))):
            assert torch.equal(a,b)


def test_binding_context_is_identity_blind_and_local():
    ref=load_reference();bank,_=fake_bank(ref)
    head=WindowRoleReader(ref,16,width=32,radius=1,linear=True).eval()
    f,mask,mentions,valid=bank.get(torch.arange(4))
    before=head(f,mask,mentions,valid)
    changed=f.clone()
    changed[mentions.any(1)[:,None,:,None].expand_as(changed)]=12345.
    changed[:,1:]=torch.randn_like(changed[:,1:])*1000
    assert torch.equal(before,head(changed,mask,mentions,valid))
    before.sum().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in head.parameters())


def test_joint_binding_masks_diagonal_but_allows_self_loop():
    ref=load_reference();bank,_=fake_bank(ref)
    head=WindowRoleReader(ref,16,width=32,radius=1,linear=True).eval()
    score=head(*bank.get(torch.arange(4)))
    assert score[0,0,0].item()==-10000 and score[0,1,1].item()==-10000
    assert score[0,0,1]>-1000 and score[0,1,0]>-1000
    assert score[1,0,0]>-1000
    assert torch.isfinite(score).all()


def test_custom_vocabulary_changes_payload_not_semantic_input():
    ref=load_reference();external=['Acme','North Office','Node-203','Иван']
    runtime=runtime_vocabulary(ref,external)
    bank=TextBank(runtime.ENTITIES,canonical_names=ref.ENTITIES)
    row=bank.add('Иван manages Acme.')
    assert bank.texts[row.index]=='Alice manages Bob.' and row.payload==(3,0)
    row=bank.add("North Office's owner is Node-203.")
    assert bank.texts[row.index]=="Alice's owner is Bob." and row.payload==(1,2)
    assert runtime.E==4 and ref.E==12
    for names in ([],['duplicate','duplicate']):
        with pytest.raises(ValueError):runtime_vocabulary(ref,names)


@pytest.mark.parametrize('n',[1,2,13,32,128,256])
def test_runtime_entity_count_does_not_require_parameters(n):
    ref=load_reference();runtime=runtime_vocabulary(ref,[f'Node-{i}' for i in range(n)])
    values=(0,min(n-1,1))
    payload=slot_payload([TextRef(0,values)],runtime.E,'cpu')
    pair=torch.full((1,4,4),-100.);pair[0,1,0]=100.
    args=hard_arguments(torch.zeros(1,6),torch.zeros(1,6),pair,payload)
    assert args[1].size(-1)==n and args[1].argmax().item()==min(n-1,1)
