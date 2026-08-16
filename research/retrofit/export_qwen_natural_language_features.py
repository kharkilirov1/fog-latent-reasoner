#!/usr/bin/env python3
# Evaluation-only feature export. No training; push below intentionally triggers CI.
from __future__ import annotations
import json
from pathlib import Path
import torch

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
LAYERS = [2, 0, 1]

CASES = [
    {
        "name":"plain_unseen_chain","category":"supported","start":"Suppose the register begins with seventeen.","start_value":17,
        "instructions":[["Give the running number eleven extra units.",0,11],["Make it five times as large as it is now.",2,5],["Take twelve away from what remains.",1,12],["Add another seven.",0,7],["Triple the result.",2,3]],
        "stop":"That is enough arithmetic. Keep the value you have and stop.","expected":2
    },
    {
        "name":"story_style","category":"supported","start":"Mira begins with seventeen tokens in a cyclic counter.","start_value":17,
        "instructions":[["A friend gives her eleven more tokens.",0,11],["The counter is then expanded to five times its amount.",2,5],["She spends twelve tokens from the counter.",1,12]],
        "stop":"The story ends here; preserve the amount now shown.","expected":4
    },
    {
        "name":"colloquial","category":"supported","start":"Start me off at twenty-nine.","start_value":29,
        "instructions":[["Bump it up by four.",0,4],["Double it.",2,2],["Knock three off the result.",1,3],["Give it ten more.",0,10]],
        "stop":"Okay, stop there.","expected":11
    },
    {
        "name":"heldout_pairs","category":"supported","start":"The cyclic register initially contains six.","start_value":6,
        "instructions":[["Put eleven more on top of it.",0,11],["Quintuple what you have.",2,5],["Take away twelve.",1,12]],
        "stop":"Freeze the register now.","expected":11
    },
    {
        "name":"implicit_world_story","category":"supported","start":"A score counter starts on fourteen.","start_value":14,
        "instructions":[["The score gains three points.",0,3],["A penalty removes five points.",1,5],["The score is tripled.",2,3]],
        "stop":"Final whistle: keep this score.","expected":5
    },
    {
        "name":"negation_distractor","category":"stress","start":"Begin at nine.","start_value":9,
        "instructions":[["Do not add eleven; subtract two instead.",1,2],["Ignore the words multiply by five; just add one.",0,1]],
        "stop":"Stop now.","expected":8
    },
    {
        "name":"idiom_quantity","category":"stress","start":"We open with a value of twenty-four.","start_value":24,
        "instructions":[["Increase it by a dozen.",0,12],["Double the current amount.",2,2],["Take away half a dozen.",1,6]],
        "stop":"Finish here.","expected":10
    },
    {
        "name":"russian_arithmetic","category":"crosslingual","start":"Пусть в регистре сначала будет семнадцать.","start_value":17,
        "instructions":[["Прибавь к текущему значению четыре.",0,4],["Удвой результат.",2,2],["Вычти три.",1,3]],
        "stop":"На этом остановись и сохрани значение.","expected":8
    },
    {
        "name":"conditional_branch_UNSUPPORTED","category":"unsupported","start":"Begin at eight.","start_value":8,
        "instructions":[["If the current value is even, add three; otherwise subtract two.",-1,-1]],
        "stop":"Stop after the conditional.","expected":None
    },
    {
        "name":"relational_logic_UNSUPPORTED","category":"unsupported","start":"Alice is taller than Bob, and Bob is taller than Carla.","start_value":0,
        "instructions":[["Who is taller, Alice or Carla?",-1,-1]],
        "stop":"Return the answer.","expected":None
    },
    {
        "name":"division_UNSUPPORTED","category":"unsupported","start":"Begin at twenty.","start_value":20,
        "instructions":[["Halve the current value.",-1,-1]],
        "stop":"Stop.","expected":None
    }
]

def full_prompt(c):
    return c["start"] + " " + " ".join(x[0] for x in c["instructions"]) + " " + c["stop"]

def main():
    from transformers import AutoTokenizer, AutoModel
    device = torch.device("cpu")
    tok = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
    tok.pad_token = tok.pad_token or tok.eos_token
    tok.padding_side = "right"
    model = AutoModel.from_pretrained(MODEL, torch_dtype=torch.float32, low_cpu_mem_usage=True).to(device).eval()
    for p in model.parameters(): p.requires_grad_(False)
    texts=[]
    for c in CASES:
        texts += [c["start"], c["stop"]] + [x[0] for x in c["instructions"]] + [full_prompt(c)]
    texts=list(dict.fromkeys(texts))
    cache={}
    bs=8
    with torch.inference_mode():
        for st in range(0,len(texts),bs):
            rows=texts[st:st+bs]
            enc=tok(rows,padding=True,truncation=True,max_length=160,return_tensors="pt",add_special_tokens=False)
            out=model(**enc,use_cache=False,return_dict=True,output_hidden_states=True)
            mask=enc["attention_mask"].bool()
            for i,t in enumerate(rows):
                n=int(mask[i].sum())
                feat=torch.stack([out.hidden_states[li][i,:n].float().cpu() for li in LAYERS],0).half()
                cache[t]={"features":feat,"mask":torch.ones(n,dtype=torch.bool)}
    payload={"model":MODEL,"layers":LAYERS,"cases":CASES,"features":cache}
    torch.save(payload,"qwen_natural_language_features.pt")
    Path("qwen_natural_language_features_manifest.json").write_text(json.dumps({"model":MODEL,"layers":LAYERS,"n_texts":len(texts),"cases":CASES},ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({"status":"ok","model":MODEL,"layers":LAYERS,"n_texts":len(texts)},ensure_ascii=False))
if __name__ == "__main__": main()
