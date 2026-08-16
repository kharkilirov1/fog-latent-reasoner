#!/usr/bin/env python3
import torch
MODEL='Qwen/Qwen2.5-0.5B-Instruct'
LAYERS=[2,0,1]
TEXTS=[
'Before executing any instruction, the cyclic register contains seventeen.',
'Program instruction: Move eleven steps forward on the number cycle.',
'Program instruction: Stretch the quantity to five times what it is now.',
'Program instruction: Move twelve steps backward on the number cycle.',
'Execution has reached its endpoint. Freeze the register and finish.',
'Suppose the register begins with seventeen.',
'Program instruction: Give the running number eleven extra units.',
'Program instruction: Make it five times as large as it is now.',
'Program instruction: Take twelve away from what remains.',
'Program instruction: Add another seven.',
'Program instruction: Triple the result.',
'That is enough arithmetic. Keep the value you have and stop.'
]
from transformers import AutoTokenizer,AutoModelForCausalLM
tok=AutoTokenizer.from_pretrained(MODEL,use_fast=True)
if tok.pad_token_id is None: tok.pad_token=tok.eos_token
tok.padding_side='right'
model=AutoModelForCausalLM.from_pretrained(MODEL,torch_dtype=torch.float32,low_cpu_mem_usage=True).eval()
outmap={}
with torch.inference_mode():
    for st in range(0,len(TEXTS),4):
        rows=TEXTS[st:st+4]
        enc=tok(rows,padding=True,truncation=True,max_length=96,return_tensors='pt',add_special_tokens=False)
        out=model.model(**enc,use_cache=False,return_dict=True,output_hidden_states=True)
        mask=enc['attention_mask'].bool()
        for i,t in enumerate(rows):
            n=int(mask[i].sum())
            outmap[t]={'features':torch.stack([out.hidden_states[li][i,:n].float() for li in LAYERS],0).half(),'mask':torch.ones(n,dtype=torch.bool)}
torch.save({'model':MODEL,'layers':LAYERS,'features':outmap},'qwen_sanity_features.pt')
print('saved',len(outmap),'phrases')
