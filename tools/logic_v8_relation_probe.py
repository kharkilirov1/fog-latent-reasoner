"""Post-locked-test diagnostic: frozen LM relation classification from original anchors.

Not a fresh locked result. Prompts contain no gold relation, role, or program answer.
No fitting occurs. Six cyclic choice orders are averaged to reduce choice-letter bias.
"""
import argparse
import json
from pathlib import Path
import sys
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'research'/'logic_v8'))
from cache import load_train_dev
from model import TextBank
from window_roles import TypedSemanticReader
from train_typed import state_digest
from run import make_program_refs,predict_instructions,evaluate_predictions,instruction_metrics

@torch.inference_mode()
def classify(net,tok,texts,ref):
    # Descriptions are the EXACT anchors declared in the original baseline.
    descriptions=[' / '.join(ref.REL_ANCHORS[r]) for r in ref.RELATIONS]
    letters='ABCDEF';ids=[tok.encode(x,add_special_tokens=False) for x in letters]
    if not all(len(x)==1 for x in ids):raise ValueError('Choice letters must be single tokens')
    choice_ids=torch.tensor([x[0] for x in ids]);prompts=[];orders=[]
    for text in texts:
        for rotation in range(ref.R):
            order=[(rotation+j)%ref.R for j in range(ref.R)]
            choices='\n'.join(letters[j]+': '+descriptions[k] for j,k in enumerate(order))
            user='Which relationship is expressed by this statement?\nStatement: '+text+'\n'+choices+'\nReply with exactly one letter A, B, C, D, E, or F.'
            prompts.append(tok.apply_chat_template([{'role':'system','content':'Classify the meaning of a statement. Answer only with the choice letter.'},{'role':'user','content':user}],tokenize=False,add_generation_prompt=True))
            orders.append(order)
    scores=torch.zeros(len(texts),ref.R)
    for start in range(0,len(prompts),8):
        enc=tok(prompts[start:start+8],return_tensors='pt',padding=True,add_special_tokens=False)
        h=net.model(**enc,use_cache=False,return_dict=True).last_hidden_state
        last=enc['attention_mask'].sum(-1)-1
        logits=net.lm_head(h[torch.arange(len(last)),last])[:,choice_ids].float()
        logp=logits.log_softmax(-1)
        for j,row in enumerate(logp):
            scores[(start+j)//ref.R,orders[start+j]]+=row.cpu()/ref.R
        if start%80==0:print(json.dumps({'semantic_relation_prompts':start,'total':len(prompts)}),flush=True)
    return scores

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--features',required=True);ap.add_argument('--output',default='relation-probe.json');a=ap.parse_args()
    torch.set_num_threads(2)
    ref,data,programs,banks,refs=load_train_dev(a.features)
    ck=torch.load(a.checkpoint,map_location='cpu',weights_only=True);s=ck['state_dict']
    model=TypedSemanticReader(ref,ck['hidden_size'],s['op.proto_layers'],s['rel.proto_layers'],ck['bind_config']).eval();model.load_state_dict(s)
    digest=state_digest(model)
    from transformers import AutoTokenizer,AutoModelForCausalLM
    tok=AutoTokenizer.from_pretrained(ck['model'],revision=ck['backbone_revision'],use_fast=True);tok.pad_token=tok.pad_token or tok.eos_token;tok.padding_side='right'
    net=AutoModelForCausalLM.from_pretrained(ck['model'],revision=ck['backbone_revision'],torch_dtype=torch.float32).eval()
    for p in net.parameters():p.requires_grad_(False)
    spec=ref.PhraseSpec()
    programs['opened_test']=ref.generate_programs(ref.SEED+9000,(1,2,3,4,5,6,8),72,train=False)
    tb=TextBank(ref.ENTITIES);refs['opened_test']=make_program_refs(programs['opened_test'],'test',tb,ref)
    phrase_refs=[tb.add(spec.texts[r[0]]) for r in spec.test_examples]
    tb.featurize(net,tok,torch.device('cpu'));banks['opened_test']=tb
    predictions={};questions={};records=[]
    for split in ('train','dev','opened_test'):
        textrefs=[r for p in refs[split] for r in p]
        pred=predict_instructions(model,banks[split],textrefs,ref);predictions[split]=(pred,textrefs)
        for i,(ins,tr) in enumerate(zip(pred,textrefs)):
            if ins.op==ref.OID['BIND']:
                text=banks[split].texts[tr.index]
                if text not in questions:questions[text]=len(questions)
    oldphrase=predict_instructions(model,tb,phrase_refs,ref)
    for ins,tr in zip(oldphrase,phrase_refs):
        if ins.op==ref.OID['BIND']:
            text=tb.texts[tr.index]
            if text not in questions:questions[text]=len(questions)
    scores=classify(net,tok,list(questions),ref)
    revised={text:int(scores[i].argmax()) for text,i in questions.items()}
    def replace(pred,textrefs,bank):
        return [ref.Instr(x.op,x.e1,revised[bank.texts[r.index]],x.e2) if x.op==ref.OID['BIND'] else x for x,r in zip(pred,textrefs)]
    out={'protocol':'POST_LOCKED_FROZEN_RELATION_PROBE','independent_locked_result':False,'uses_gold_in_prompt':False,
         'uses_training':False,'definitions':'original immutable REL_ANCHORS','choice_order':'all six cyclic rotations',
         'backbone_revision':ck['backbone_revision'],'checkpoint_state_sha256':digest,'splits':{}}
    for split,(pred,textrefs) in predictions.items():
        new=replace(pred,textrefs,banks[split]);gold=[x for p in programs[split] for x in p.instructions]
        out['splits'][split]={'before':evaluate_predictions(pred,programs[split],ref),'after':evaluate_predictions(new,programs[split],ref),
                              'field_metrics':instruction_metrics(new,gold,ref),'full_shuffle':evaluate_predictions(new,programs[split],ref,mode='full')}
        seen=set()
        for old,updated,target,tr in zip(pred,new,gold,textrefs):
            text=banks[split].texts[tr.index]
            if text in seen or old.op!=ref.OID['BIND']:continue
            seen.add(text)
            records.append({'split':split,'text':text,'gold_relation':ref.RELATIONS[target.rel] if target.rel>=0 else None,
                            'old_relation':ref.RELATIONS[old.rel],'new_relation':ref.RELATIONS[updated.rel],
                            'mean_choice_log_probabilities':scores[questions[text]].tolist()})
    out['opened_phrase_metrics']=instruction_metrics(replace(oldphrase,phrase_refs,tb),[ref.Instr(*r[1:]) for r in spec.test_examples],ref)
    out['diagnostic_rows']=records
    if state_digest(model)!=digest:raise AssertionError('Frozen adapter was changed')
    Path(a.output).write_text(json.dumps(out,indent=2));print(json.dumps({'splits':out['splits'],'opened_phrase_metrics':out['opened_phrase_metrics']}),flush=True)
if __name__=='__main__':main()
