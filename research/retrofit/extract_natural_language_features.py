#!/usr/bin/env python3
"""Extract ONLY frozen Qwen hidden states for the fixed natural-language audit.

This file intentionally contains no FOG weights and performs no training.
The resulting features can be evaluated later with the exact private/local
FOG checkpoint.
"""
from __future__ import annotations
import json
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModel
from natural_language_audit import CASES, full_prompt

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
LAYERS = [2, 0, 1]
OUT = Path("natural_language_qwen_features.pt")
META = Path("natural_language_qwen_features.json")


def collect_texts():
    texts=[]
    for c in CASES:
        texts.append(c.start)
        texts.extend(x[0] for x in c.instructions)
        texts.append(c.stop)
        texts.append(full_prompt(c))
    # Stable deduplication, preserving evaluator order.
    return list(dict.fromkeys(texts))


def main():
    torch.manual_seed(0)
    tok=AutoTokenizer.from_pretrained(MODEL,use_fast=True)
    if tok.pad_token is None: tok.pad_token=tok.eos_token
    tok.padding_side="right"
    model=AutoModel.from_pretrained(MODEL,torch_dtype=torch.float32,low_cpu_mem_usage=True)
    model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    texts=collect_texts()
    rows={}
    # Small batches keep CPU RAM bounded on the free GitHub runner.
    with torch.inference_mode():
        for st in range(0,len(texts),8):
            batch=texts[st:st+8]
            enc=tok(batch,padding=True,truncation=True,max_length=96,return_tensors="pt",add_special_tokens=False)
            out=model(**enc,use_cache=False,return_dict=True,output_hidden_states=True)
            mask=enc["attention_mask"].bool()
            for i,text in enumerate(batch):
                n=int(mask[i].sum())
                feat=torch.stack([out.hidden_states[j][i,:n].float().cpu() for j in LAYERS],0)
                rows[text]={"features":feat,"mask":torch.ones(n,dtype=torch.bool),"input_ids":enc["input_ids"][i,:n].cpu()}
            print(f"encoded {min(st+len(batch),len(texts))}/{len(texts)}",flush=True)
    torch.save({"model":MODEL,"layers":LAYERS,"texts":rows},OUT)
    META.write_text(json.dumps({"model":MODEL,"layers":LAYERS,"n_texts":len(texts),"texts":texts},ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(f"saved {OUT} ({OUT.stat().st_size} bytes)",flush=True)

if __name__=="__main__": main()
