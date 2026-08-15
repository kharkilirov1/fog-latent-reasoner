import json,random,re
from pathlib import Path
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM
SEED=20260821;P=31;MODEL_ID='HuggingFaceTB/SmolLM2-135M-Instruct'
random.seed(SEED);torch.manual_seed(SEED);torch.set_num_threads(4)
def make(r,d):
 x=r.randrange(P);s=x;ops=[]
 for _ in range(d):
  kind=r.choice(('add','sub','mul'))
  if kind=='add':k=r.randint(1,8);x=(x+k)%P;t=f'Add {k}.'
  elif kind=='sub':k=r.randint(1,8);x=(x-k)%P;t=f'Subtract {k}.'
  else:k=r.randint(2,5);x=(x*k)%P;t=f'Multiply by {k}.'
  ops.append(t)
 return s,ops,x,d
def prompt(e):
 s,ops,a,d=e;return f'Work modulo {P}. Start with {s}. '+ ' '.join(f'Step {i+1}: {o}' for i,o in enumerate(ops))+' What is the final value? Answer with only the integer.\nAnswer:'
def parse(s):
 m=re.search(r'[-+]?\d+',s);return int(m.group()) if m else None
@torch.no_grad()
def main():
 tok=AutoTokenizer.from_pretrained(MODEL_ID);tok.pad_token=tok.pad_token or tok.eos_token;tok.padding_side='left';m=AutoModelForCausalLM.from_pretrained(MODEL_ID).eval()
 r=random.Random(600);ex=sum(([make(r,d) for _ in range(50)] for d in [1,2,3,4,5,6,8,10,12]),[]);by={d:[0,0] for d in [1,2,3,4,5,6,8,10,12]};samples=[]
 for st in range(0,len(ex),10):
  bb=ex[st:st+10];ps=[prompt(e) for e in bb];enc=tok(ps,padding=True,return_tensors='pt');out=m.generate(**enc,max_new_tokens=6,do_sample=False,pad_token_id=tok.eos_token_id)
  for i,e in enumerate(bb):
   gen=tok.decode(out[i,enc['input_ids'].size(1):],skip_special_tokens=True);pred=parse(gen);a=e[2];d=e[3];by[d][0]+=int(pred==a);by[d][1]+=1
   if len(samples)<12:samples.append({'depth':d,'answer':a,'prediction':pred,'generation':gen})
  if st%100==0:print('raw',st+len(bb),'/',len(ex),flush=True)
 res={str(d):c/n for d,(c,n) in by.items()};obj={'model':MODEL_ID,'modulus':P,'n':len(ex),'raw_generation_accuracy':res,'samples':samples};Path('retrofit_raw_baseline.json').write_text(json.dumps(obj,indent=2));print('FINAL_RAW',json.dumps(obj,sort_keys=True),flush=True)
if __name__=='__main__':main()
