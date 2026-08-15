import os, re, json, random
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

SEED = 20260816
MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"
P = 101
DEVICE = "cpu"
torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


def mk(rng, depth, p=P):
    x = rng.randrange(p); start = x; ops = []
    for _ in range(depth):
        kind = rng.choice(("add", "sub", "mul"))
        if kind == "add":
            k = rng.randint(1, 12); x = (x + k) % p; ops.append(f"Add {k}.")
        elif kind == "sub":
            k = rng.randint(1, 12); x = (x - k) % p; ops.append(f"Subtract {k}.")
        else:
            k = rng.randint(2, 5); x = (x * k) % p; ops.append(f"Multiply by {k}.")
    return {"start": start, "ops": ops, "answer": x, "depth": depth}


def segment_tokens(tok, e, p=P):
    seg = [f"Work modulo {p}. Start with {e['start']}. "]
    seg += [f"Step {i+1}: {op} " for i, op in enumerate(e["ops"])]
    seg += ["What is the final value? Answer with only the integer.\nAnswer:"]
    ids, marks = [], []
    for j, s in enumerate(seg):
        ids.extend(tok.encode(s, add_special_tokens=False))
        if j <= len(e["ops"]):
            marks.append(len(ids) - 1)
    return ids, marks


@torch.no_grad()
def chain_features(base, tok, examples, h, batch=12):
    slots_n = max(e["depth"] for e in examples) + 1
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    S, M, L, Y, D = [], [], [], [], []
    for st in range(0, len(examples), batch):
        bb = examples[st:st+batch]
        built = [segment_tokens(tok, e) for e in bb]
        mx = max(len(x[0]) for x in built)
        ids = torch.full((len(bb), mx), pad, dtype=torch.long)
        am = torch.zeros_like(ids)
        for i, (a, _) in enumerate(built):
            ids[i, :len(a)] = torch.tensor(a); am[i, :len(a)] = 1
        hs = base(input_ids=ids, attention_mask=am, use_cache=False).last_hidden_state.float()
        for i, ((a, marks), e) in enumerate(zip(built, bb)):
            s = torch.zeros(slots_n, h); m = torch.zeros(slots_n, dtype=torch.bool)
            for j, pos in enumerate(marks):
                s[j] = hs[i, pos]; m[j] = True
            S.append(s); M.append(m); L.append(hs[i, len(a)-1]); Y.append(e["answer"]); D.append(e["depth"])
        if st % 240 == 0:
            print(f"features chain {st+len(bb)}/{len(examples)}", flush=True)
    return TensorDataset(torch.stack(S), torch.stack(M), torch.stack(L), torch.tensor(Y), torch.tensor(D))


def gsm_ans(s):
    m = re.search(r"####\s*([-+]?\d[\d,]*)", s)
    return int(m.group(1).replace(",", "")) if m else None


@torch.no_grad()
def context_features(base, tok, rows, h, kctx=32, batch=8):
    prompts, labels = [], []
    for r in rows:
        y = gsm_ans(r["answer"])
        if y is not None and 0 <= y < P:
            prompts.append("Solve the problem. Answer with only the integer.\n" + r["question"] + "\nAnswer:")
            labels.append(y)
    C, M, L, Y = [], [], [], []
    for st in range(0, len(prompts), batch):
        enc = tok(prompts[st:st+batch], padding=True, truncation=True, max_length=160,
                  return_tensors="pt", add_special_tokens=False)
        hs = base(**enc, use_cache=False).last_hidden_state.float()
        for i in range(len(enc["input_ids"])):
            n = int(enc["attention_mask"][i].sum()); take = min(kctx, n)
            c = torch.zeros(kctx, h); m = torch.zeros(kctx, dtype=torch.bool)
            c[-take:] = hs[i, n-take:n]; m[-take:] = True
            C.append(c); M.append(m); L.append(hs[i, n-1]); Y.append(labels[st+i])
        if st % 80 == 0:
            print(f"features gsm {st+len(enc['input_ids'])}/{len(prompts)}", flush=True)
    if not Y:
        raise RuntimeError("No GSM8K rows in constrained answer range")
    return TensorDataset(torch.stack(C), torch.stack(M), torch.stack(L), torch.tensor(Y))


class Zero(nn.Module):
    def forward(self, s, m, last): return torch.zeros_like(last)


class MLP(nn.Module):
    def __init__(self, h, b=128):
        super().__init__(); self.net = nn.Sequential(nn.Linear(2*h, b), nn.GELU(), nn.Linear(b, h))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)
    def forward(self, s, m, last):
        mf = m.float().unsqueeze(-1); mean = (s*mf).sum(1) / mf.sum(1).clamp_min(1)
        return self.net(torch.cat([last, mean], -1))


class GRU(nn.Module):
    def __init__(self, h, d=96):
        super().__init__(); self.i = nn.Linear(h, d); self.g = nn.GRU(d, d, batch_first=True); self.o = nn.Linear(d, h)
        nn.init.zeros_(self.o.weight); nn.init.zeros_(self.o.bias)
    def forward(self, s, m, last):
        z, _ = self.g(self.i(s)); idx = m.sum(1).clamp_min(1) - 1
        return self.o(z[torch.arange(z.size(0)), idx])


class BOp(nn.Module):
    def __init__(self, d, r=48):
        super().__init__(); self.a = nn.Linear(d, r, bias=False); self.b = nn.Linear(d, r, bias=False)
        self.o = nn.Linear(r, d, bias=False); self.n = nn.LayerNorm(d)
    def forward(self, z, c): return self.n(z + self.o(self.a(z) * self.b(c)))


def bprod(z, c):
    q = z.reshape(z.size(0), -1, 2); w = c.reshape(c.size(0), -1, 2)
    q = F.normalize(q, dim=-1); w = F.normalize(w, dim=-1)
    return torch.stack((q[...,0]*w[...,0]-q[...,1]*w[...,1],
                        q[...,0]*w[...,1]+q[...,1]*w[...,0]), -1).reshape_as(z)


class FOGSeq(nn.Module):
    def __init__(self, h, d=96):
        super().__init__(); self.a = nn.Linear(h, d); self.c = nn.Linear(h, d)
        self.ops = nn.ModuleList([BOp(d) for _ in range(3)]); self.r = nn.Linear(2*d, 6)
        self.n = nn.LayerNorm(d); self.o = nn.Linear(d, h); self.routes = None
        nn.init.zeros_(self.o.weight); nn.init.zeros_(self.o.bias)
    def one(self, z, c):
        cand = [z, c, self.n(bprod(z, c))] + [op(z, c) for op in self.ops]
        p = F.softmax(self.r(torch.cat([z, c], -1)), -1)
        hard = F.one_hot(p.argmax(-1), 6).float(); w = hard + p - p.detach()
        return self.n(sum(w[:,i:i+1]*cand[i] for i in range(6))), p
    def forward(self, s, m, last, force=None, intervention=None):
        z = self.n(self.a(s[:,0])); routes = []
        mx = s.size(1)-1 if force is None else min(force, s.size(1)-1)
        for t in range(mx):
            active = m[:,t+1]
            if not active.any(): break
            zn, p = self.one(z, self.n(self.c(s[:,t+1]))); z = torch.where(active[:,None], zn, z); routes.append(p.detach())
            if intervention == "shuffle2" and t == 1: z = z[torch.randperm(z.size(0))]
            if intervention == "zero2" and t == 1: z = torch.zeros_like(z)
        self.routes = routes
        return self.o(z)


class FOGCtx(nn.Module):
    def __init__(self, h, d=96):
        super().__init__(); self.i = nn.Linear(h, d); self.k = nn.Linear(h, d, bias=False); self.v = nn.Linear(h, d, bias=False)
        self.q = nn.Linear(d, d, bias=False); self.ops = nn.ModuleList([BOp(d) for _ in range(3)])
        self.r = nn.Linear(2*d, 6); self.n = nn.LayerNorm(d); self.o = nn.Linear(d, h)
        nn.init.zeros_(self.o.weight); nn.init.zeros_(self.o.bias)
    def one(self, z, ctx, m):
        sc = torch.einsum("bd,bkd->bk", F.normalize(self.q(z), dim=-1), F.normalize(self.k(ctx), dim=-1))*8
        sc = sc.masked_fill(~m, -1e9); a = F.softmax(sc, -1); c = self.n(torch.einsum("bk,bkd->bd", a, self.v(ctx)))
        cand = [z, c, self.n(bprod(z, c))] + [op(z, c) for op in self.ops]
        p = F.softmax(self.r(torch.cat([z, c], -1)), -1); hard = F.one_hot(p.argmax(-1), 6).float(); w = hard + p - p.detach()
        return self.n(sum(w[:,i:i+1]*cand[i] for i in range(6))), p
    def forward(self, ctx, m, last, steps=4, intervention=None):
        z = self.n(self.i(last))
        for t in range(steps):
            z, _ = self.one(z, ctx, m)
            if intervention == "shuffle2" and t == 1: z = z[torch.randperm(z.size(0))]
            if intervention == "zero2" and t == 1: z = torch.zeros_like(z)
        return self.o(z)


class Classifier(nn.Module):
    def __init__(self, h, adapter, p=P):
        super().__init__(); self.adapter = adapter; self.head = nn.Linear(h, p)
    def forward(self, s, m, last, **kw):
        if isinstance(self.adapter, FOGSeq): delta = self.adapter(s, m, last, **kw)
        else: delta = self.adapter(s, m, last)
        return self.head(last + delta)


class ContextClassifier(nn.Module):
    def __init__(self, h, adapter, p=P):
        super().__init__(); self.adapter = adapter; self.head = nn.Linear(h, p)
    def forward(self, ctx, m, last, steps=4, intervention=None):
        if isinstance(self.adapter, FOGCtx): delta = self.adapter(ctx, m, last, steps=steps, intervention=intervention)
        else: delta = self.adapter(ctx, m, last)
        return self.head(last + delta)


def count_params(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)


def train_chain(name, model, ds, epochs=6):
    seed_all(SEED + sum(map(ord, name))); model.train(); opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    for ep in range(epochs):
        total = n = 0
        for s,m,l,y,d in DataLoader(ds, batch_size=64, shuffle=True):
            loss = F.cross_entropy(model(s,m,l), y); opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1); opt.step()
            total += loss.item()*len(y); n += len(y)
        if ep in {0, epochs-1} or (ep+1)%3 == 0: print(name, ep+1, total/n, flush=True)
    return model


def train_ctx(name, model, ds, epochs=8):
    seed_all(SEED + sum(map(ord, name))); model.train(); opt = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-4)
    for ep in range(epochs):
        total = n = 0
        for c,m,l,y in DataLoader(ds, batch_size=64, shuffle=True):
            loss = F.cross_entropy(model(c,m,l,steps=4), y); opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1); opt.step()
            total += loss.item()*len(y); n += len(y)
        if ep in {0, epochs-1} or (ep+1)%3 == 0: print(name, ep+1, total/n, flush=True)
    return model


@torch.no_grad()
def eval_chain(model, ds, force=None, intervention=None):
    model.eval(); a = defaultdict(lambda:[0,0])
    for s,m,l,y,d in DataLoader(ds, batch_size=128):
        kw = {"force":force,"intervention":intervention} if isinstance(model.adapter, FOGSeq) else {}
        pr = model(s,m,l,**kw).argmax(-1)
        for x in d.unique():
            q = d == x; a[int(x)][0] += int((pr[q] == y[q]).sum()); a[int(x)][1] += int(q.sum())
    return {str(k):v[0]/v[1] for k,v in sorted(a.items())}


@torch.no_grad()
def eval_ctx(model, ds, steps=4, intervention=None):
    model.eval(); c = n = 0
    for x,m,l,y in DataLoader(ds, batch_size=64):
        pr = model(x,m,l,steps=steps,intervention=intervention).argmax(-1); c += int((pr == y).sum()); n += len(y)
    return c/n


def main():
    print("load", MODEL_ID, flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_ID); tok.pad_token = tok.pad_token or tok.eos_token
    lm = AutoModelForCausalLM.from_pretrained(MODEL_ID); lm.eval(); [x.requires_grad_(False) for x in lm.parameters()]
    base = lm.model; h = lm.config.hidden_size
    print("hidden", h, "backbone_params", sum(x.numel() for x in lm.parameters()), "classes", P, flush=True)

    r = random.Random(1); tr = sum(([mk(r,d) for _ in range(240)] for d in [1,2,3,4]), [])
    q = random.Random(2); te = sum(([mk(q,d) for _ in range(80)] for d in [1,2,3,4,5,6,8,10,12]), [])
    trf = chain_features(base, tok, tr, h); tef = chain_features(base, tok, te, h)

    chain_models = {
        "linear": Classifier(h, Zero()),
        "mlp": Classifier(h, MLP(h)),
        "gru": Classifier(h, GRU(h)),
        "fog": Classifier(h, FOGSeq(h)),
    }
    cres, params = {}, {}
    for name, model in chain_models.items():
        train_chain(name, model, trf, epochs=8 if name == "fog" else 6)
        cres[name] = eval_chain(model, tef); params[name] = count_params(model)
        print("CHAIN", name, cres[name], flush=True)
    cres["fog_force1"] = eval_chain(chain_models["fog"], tef, force=1)
    cres["fog_shuffle2"] = eval_chain(chain_models["fog"], tef, intervention="shuffle2")
    cres["fog_zero2"] = eval_chain(chain_models["fog"], tef, intervention="zero2")

    gsm = load_dataset("openai/gsm8k", "main")
    trrows = [x for x in gsm["train"] if (lambda y: y is not None and 0 <= y < P)(gsm_ans(x["answer"]))]
    terows = [x for x in gsm["test"] if (lambda y: y is not None and 0 <= y < P)(gsm_ans(x["answer"]))]
    random.Random(300).shuffle(trrows); random.Random(301).shuffle(terows)
    trrows = trrows[:320]; terows = terows[:120]
    print("gsm filtered", len(trrows), len(terows), flush=True)
    trc = context_features(base, tok, trrows, h); tec = context_features(base, tok, terows, h)

    gsm_models = {
        "linear": ContextClassifier(h, Zero()),
        "mlp": ContextClassifier(h, MLP(h)),
        "fog": ContextClassifier(h, FOGCtx(h)),
    }
    gres = {}
    for name, model in gsm_models.items():
        train_ctx("gsm_"+name, model, trc, epochs=10 if name == "fog" else 8)
        gres[name] = eval_ctx(model, tec, steps=4)
        print("GSM", name, gres[name], flush=True)
    for steps in [1,2,4,8]: gres[f"fog_r{steps}"] = eval_ctx(gsm_models["fog"], tec, steps=steps)
    gres["fog_shuffle2"] = eval_ctx(gsm_models["fog"], tec, steps=4, intervention="shuffle2")
    gres["fog_zero2"] = eval_ctx(gsm_models["fog"], tec, steps=4, intervention="zero2")

    out = {
        "model": MODEL_ID, "frozen_backbone": True, "classes": P,
        "chain": {"train_depths":[1,2,3,4], "test_depths":[1,2,3,4,5,6,8,10,12], "n_train":len(tr), "n_test":len(te), "trainable_params":params, "results":cres},
        "gsm8k_numeric_subset": {"source":"openai/gsm8k official train/test", "answer_range":[0,P-1], "n_train":len(trc), "n_test":len(tec), "results":gres},
        "limitations":["Frozen 135M backbone; only retrofit modules and matched numeric heads are trained", "GSM8K is an official held-out subset restricted to final answers in 0..100", "Numeric classification is constrained rather than free-form generation"]
    }
    Path("retrofit_results.json").write_text(json.dumps(out, indent=2))
    torch.save({"chain_fog":chain_models["fog"].state_dict(), "gsm_fog":gsm_models["fog"].state_dict(), "results":out}, "retrofit_adapters.pt")
    print("FINAL", json.dumps(out, sort_keys=True), flush=True)


if __name__ == "__main__": main()
