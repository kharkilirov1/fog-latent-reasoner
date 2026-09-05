"""Label-free frozen-LM role probes on TRAIN/DEV only.

Probes use predicted opcode/relation, never gold roles, program answers, or test text.
"""
import argparse
import json
from pathlib import Path
import sys
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'research'/'logic_v8'))
from integrity import load_reference, REFERENCE_SHA256
from model import SemanticReader

DEFINITIONS={
'manager':'The manager relation goes from an employee to their manager.',
'parent':'The parent relation goes from a child to their parent.',
'owner':'The owner relation goes from an owned thing to its owner.',
'north':'The north relation goes from a starting place to the place north of it.',
'calls':'The calls relation goes from the calling function to the called function.',
'imports':'The imports relation goes from the importing module to the imported module.'}
QUESTIONS={'manager':'Whose manager is specified?', 'parent':'Whose parent is specified?',
'owner':'Whose owner is specified?', 'north':'Which place is the starting place?',
'calls':'Which function is the caller?', 'imports':'Which module is doing the importing?'}

def hypothesis(r,a,b):
    return {'manager':f'{a}\'s manager is {b}.','parent':f'{a}\'s parent is {b}.',
            'owner':f'{a}\'s owner is {b}.','north':f'{b} is north of {a}.',
            'calls':f'{a} calls {b}.','imports':f'{a} imports {b}.'}[r]

def prompt(text,rel,variant,swap):
    if variant == 'entailment':
        choices=[hypothesis(rel,'Alice','Bob'),hypothesis(rel,'Bob','Alice')]
        question='Which statement has the same meaning as the given statement?'
        prefix=''
    elif variant == 'source':
        choices=['Alice','Bob'];question=QUESTIONS[rel];prefix=DEFINITIONS[rel]+'\n'
    elif variant == 'edge':
        choices=[f'Alice --{rel}--> Bob',f'Bob --{rel}--> Alice']
        question='Which directed edge is being recorded?';prefix=DEFINITIONS[rel]+'\n'
    else:raise ValueError(variant)
    if swap:choices.reverse()
    return prefix+f'Statement: {text}\n{question}\nA: {choices[0]}\nB: {choices[1]}\nReply with A or B only.'

@torch.no_grad()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--features',required=True);ap.add_argument('--checkpoint',required=True);ap.add_argument('--output',default='role-probes.json');a=ap.parse_args()
    torch.set_num_threads(2);v=load_reference()
    data=torch.load(a.features,weights_only=True);ck=torch.load(a.checkpoint,weights_only=True)
    assert data['locked_test_included'] is False
    assert data['reference_sha256']==ck['reference_sha256']==REFERENCE_SHA256
    assert data['backbone_revision']==ck['backbone_revision']
    state=ck['state_dict'];reader=SemanticReader(v,ck['hidden_size'],state['op.proto_layers'],state['rel.proto_layers']).eval();reader.load_state_dict(state)
    rows=[]
    for split in ('train','dev'):
        bank=data[split+'_bank']
        op,rel,pair=reader(bank['features'],bank['mask'],bank['mentions'],bank['valid'])
        for i,text in enumerate(bank['texts']):
            if int(bank['valid'][i].sum()) == 2 and int(op[i].argmax()) == v.OID['BIND']:
                rows.append({'split':split,'index':i,'text':text,'predicted_opcode':'BIND','predicted_relation':v.RELATIONS[int(rel[i].argmax())],'scores':{}})
    from transformers import AutoTokenizer,AutoModelForCausalLM
    tok=AutoTokenizer.from_pretrained(data['model'],revision=data['backbone_revision'],use_fast=True);tok.pad_token=tok.pad_token or tok.eos_token;tok.padding_side='right'
    net=AutoModelForCausalLM.from_pretrained(data['model'],revision=data['backbone_revision'],torch_dtype=torch.float32).eval()
    ids=[tok.encode(s,add_special_tokens=False) for s in ('A','B')]
    if not all(len(x)==1 for x in ids):raise ValueError('Choice letters must be single tokens')
    choices=torch.tensor([x[0] for x in ids])
    texts=[];keys=[]
    for i,row in enumerate(rows):
        for variant in ('entailment','source','edge'):
            for swap in (False,True):
                messages=[{'role':'system','content':'You interpret the meaning of a single statement. Answer with A or B only.'},
                          {'role':'user','content':prompt(row['text'],row['predicted_relation'],variant,swap)}]
                texts.append(tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True));keys.append((i,variant,swap))
    for st in range(0,len(texts),8):
        enc=tok(texts[st:st+8],padding=True,return_tensors='pt',add_special_tokens=False)
        h=net.model(**enc,use_cache=False,return_dict=True).last_hidden_state
        last=enc['attention_mask'].sum(-1)-1
        logits=net.lm_head(h[torch.arange(h.size(0)),last])[:,choices].float()
        margin=logits[:,0]-logits[:,1]
        for j,value in enumerate(margin.tolist()):
            i,variant,swap=keys[st+j]
            rows[i]['scores'].setdefault(variant,[]).append({'swap':swap,'margin_forward':-value if swap else value})
        print(json.dumps({'processed':min(st+8,len(texts)),'total':len(texts)}),flush=True)
    for row in rows:
        row['mean_margins']={k:sum(x['margin_forward'] for x in z)/len(z) for k,z in row['scores'].items()}
    out={'model':data['model'],'backbone_revision':data['backbone_revision'],'locked_test_included':False,
         'uses_gold_labels':False,'probe_roles':'two lexical-role assignments; choice order symmetrized','rows':rows}
    Path(a.output).write_text(json.dumps(out,indent=2))
    print(json.dumps({'rows':len(rows),'output':a.output,'locked_test_included':False}),flush=True)
if __name__=='__main__':main()
