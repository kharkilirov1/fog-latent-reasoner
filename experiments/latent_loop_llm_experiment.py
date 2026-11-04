"""Latent-loop reasoning inside a pretrained LLM (no external executor).

No external executor or symbolic parser is used. The pretrained LLM executes
the program in its own residual stream through a recurrent loop of latent
thought tokens:

    prompt (frozen LLM, no grad, KV cached)
      -> [" Step t:" marker tokens][latent token x_t]   (thought-mode LoRA on)
      -> h_t = last hidden of latent token
      -> x_{t+1} = Feedback(h_t)                        (continuous)

Nothing symbolic is parsed: no opcode classifier, no value register, no ALU.
The only supervision is a *shared* linear readout D(h_t) of the intermediate
value (gauge fixing, research/notes/NOTE-016).

Arms (Feedback):
  cont   : x = rescale(W h)                      continuous (Coconut-like)
  attr   : x = sum_v softmax(beta*D(h))_v c_v      attractor clean-up (Hopfield)
  hard   : x = c_argmax (straight-through)        discrete reference only

Theory (NOTE-016): with clean-up contraction kappa, latent recurrence is
stable for all depths iff kappa*L<1 (L = transition Lipschitz), instead of the
L<1 needed without clean-up.
"""
from __future__ import annotations
import argparse, json, math, random, time
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"
P = 31
STYLES_TRAIN = ("base", "para")
STYLE_HELD = "alt"


def op_text(kind, k, style):
    if style == "base":
        return {"add": f"Add {k}.", "sub": f"Subtract {k}.", "mul": f"Multiply by {k}."}[kind]
    if style == "para":
        return {"add": f"Increase the current value by {k}.", "sub": f"Decrease the current value by {k}.",
                "mul": f"Scale the current value by a factor of {k}."}[kind]
    return {"add": f"Now add {k} more.", "sub": f"Now take away {k}.", "mul": f"Now take {k} times the current value."}[kind]


def make_ex(rng, depth, style):
    cur = rng.randrange(P); start = cur; ops = []; vals = []
    for _ in range(depth):
        kind = rng.choice(("add", "sub", "mul"))
        if kind == "add": k = rng.randint(1, 8); cur = (cur + k) % P
        elif kind == "sub": k = rng.randint(1, 8); cur = (cur - k) % P
        else: k = rng.randint(2, 5); cur = (cur * k) % P
        ops.append((kind, k)); vals.append(cur)
    # thought 0 reads the start value (chart bootstrap); thought t executes step t
    return {"start": start, "ops": ops, "vals": [start] + vals, "depth": depth, "style": style}


def prompt_text(e):
    s = f"Work modulo {P}. Start with {e['start']}. "
    s += "".join(f"Step {i+1}: {op_text(k, a, e['style'])} " for i, (k, a) in enumerate(e["ops"]))
    return s + "Think silently, one step at a time."


class LoRA(nn.Module):
    enabled = False  # class-level switch: on only for thought tokens

    def __init__(self, base: nn.Linear, r=8, alpha=16):
        super().__init__()
        self.base = base
        self.A = nn.Parameter(torch.randn(r, base.in_features) / math.sqrt(base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        self.s = alpha / r

    def forward(self, x):
        y = self.base(x)
        if LoRA.enabled:
            y = y + (x @ self.A.t() @ self.B.t()) * self.s
        return y


def add_lora(model, r):
    params = []
    for layer in model.model.layers:
        for parent, names in ((layer.self_attn, ("q_proj", "k_proj", "v_proj", "o_proj")),
                              (layer.mlp, ("gate_proj", "up_proj", "down_proj"))):
            for n in names:
                m = LoRA(getattr(parent, n), r=r); setattr(parent, n, m); params += [m.A, m.B]
    return params


class LatentLoop(nn.Module):
    def __init__(self, model, tok, arm, beta=1.0):
        super().__init__()
        self.model, self.tok, self.arm, self.beta = model, tok, arm, beta
        H = model.config.hidden_size
        emb = model.get_input_embeddings().weight
        self.emb_norm = emb.norm(dim=-1).mean().item()
        self.readout = nn.Linear(H, P)
        self.codes = nn.Parameter(torch.randn(P, H) * emb.std().item())
        self.begin = nn.Parameter(torch.randn(H) * emb.std().item())
        self.fb = nn.Linear(H, H)
        nn.init.eye_(self.fb.weight); nn.init.zeros_(self.fb.bias)
        self._marker_cache = {}

    def rescale(self, x):
        return x * (self.emb_norm / (x.norm(dim=-1, keepdim=True) + 1e-6))

    def feedback(self, h, logits):
        if self.arm == "cont":
            return self.rescale(self.fb(h))
        p = F.softmax(self.beta * logits, -1)
        soft = p @ self.rescale(self.codes)
        if self.arm == "attr":
            return soft
        hard = self.rescale(self.codes)[logits.argmax(-1)]
        return hard + soft - soft.detach()

    def marker_ids(self, t):
        if t not in self._marker_cache:
            self._marker_cache[t] = self.tok.encode(f" Step {t}:", add_special_tokens=False)
        return self._marker_cache[t]

    @torch.no_grad()
    def encode_prompt(self, exs):
        pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.tok.eos_token_id
        ids = [self.tok.encode(prompt_text(e), add_special_tokens=False) for e in exs]
        L = max(map(len, ids))
        inp = torch.full((len(ids), L), pad, dtype=torch.long); am = torch.zeros_like(inp)
        for i, a in enumerate(ids):  # left pad
            inp[i, L - len(a):] = torch.tensor(a); am[i, L - len(a):] = 1
        pos = (am.cumsum(-1) - 1).clamp(min=0)
        LoRA.enabled = False
        out = self.model.model(input_ids=inp, attention_mask=am, position_ids=pos, use_cache=True)
        return out.past_key_values, am, pos[:, -1]

    def forward(self, exs, T, track=False):
        cache, am, lastpos = self.encode_prompt(exs)
        Bsz = len(exs); embed = self.model.get_input_embeddings()
        x = self.rescale(self.begin).expand(Bsz, -1)
        all_logits, xs = [], []
        LoRA.enabled = True
        try:
            for t in range(0, T + 1):
                mk = torch.tensor(self.marker_ids(t))
                tok_emb = embed(mk).unsqueeze(0).expand(Bsz, -1, -1)
                inp = torch.cat([tok_emb, x.unsqueeze(1)], 1)
                n = inp.shape[1]
                am = torch.cat([am, torch.ones(Bsz, n, dtype=am.dtype)], 1)
                pos = lastpos.unsqueeze(1) + torch.arange(1, n + 1).unsqueeze(0)
                lastpos = pos[:, -1]
                out = self.model.model(inputs_embeds=inp, attention_mask=am, position_ids=pos,
                                       past_key_values=cache, use_cache=True)
                cache = out.past_key_values
                h = out.last_hidden_state[:, -1]
                logits = self.readout(h)
                all_logits.append(logits)
                if track: xs.append(x.detach())
                x = self.feedback(h, logits)
        finally:
            LoRA.enabled = False
        return torch.stack(all_logits, 1), xs


def evaluate(net, exs_by_depth, bs=32):
    net.eval(); res = {}
    with torch.no_grad():
        for key, exs in exs_by_depth.items():
            d = exs[0]["depth"]; ok = 0; step_ok = 0; tube = [0.0] * (d + 1)
            for i in range(0, len(exs), bs):
                b = exs[i:i + bs]
                logits, xs = net(b, d, track=True)
                pred = logits.argmax(-1)
                gold = torch.tensor([e["vals"] for e in b])
                ok += (pred[:, -1] == gold[:, -1]).sum().item()
                step_ok += (pred == gold).all(-1).sum().item()
                C = net.rescale(net.codes)
                for t in range(1, d + 1):  # distance of fed-back state to correct code
                    tube[t] += (xs[t] - C[gold[:, t - 1]]).norm(dim=-1).sum().item() / net.emb_norm
            n = len(exs)
            res[key] = {"final": ok / n, "all_steps": step_ok / n, "tube": [v / n for v in tube]}
    net.train(); return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="attr", choices=["cont", "attr", "hard"])
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--terminal_only", action="store_true")
    ap.add_argument("--eval_n", type=int, default=48)
    ap.add_argument("--eval_depths", default="1,2,3,4,5,6,8")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="artifacts/latent-loop-llm")
    a = ap.parse_args()
    torch.manual_seed(a.seed); random.seed(a.seed); torch.set_num_threads(int(__import__("os").environ.get("NT", "4")))

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
    for p in model.parameters(): p.requires_grad_(False)
    lora_params = add_lora(model, a.rank)
    net = LatentLoop(model, tok, a.arm, a.beta)
    head_params = [p for n, p in net.named_parameters() if not n.startswith("model.")]
    opt = torch.optim.AdamW(lora_params + head_params, lr=a.lr, weight_decay=0.0)
    ntrain = sum(p.numel() for p in lora_params + head_params)
    print(f"arm={a.arm} trainable={ntrain}", flush=True)

    rng = random.Random(1000 + a.seed)
    erng = random.Random(777)  # identical eval set across arms
    depths = [int(x) for x in a.eval_depths.split(",")]
    evalset = {}
    for d in depths:
        evalset[f"id_R{d}"] = [make_ex(erng, d, erng.choice(STYLES_TRAIN)) for _ in range(a.eval_n)]
        evalset[f"held_R{d}"] = [make_ex(erng, d, STYLE_HELD) for _ in range(a.eval_n)]

    t0 = time.time(); log = []
    for step in range(1, a.steps + 1):
        d = rng.randint(1, min(4, 1 + step // 150))  # depth curriculum
        b = [make_ex(rng, d, rng.choice(STYLES_TRAIN)) for _ in range(a.bs)]
        logits, _ = net(b, d)
        gold = torch.tensor([e["vals"] for e in b])
        if a.terminal_only:
            loss = F.cross_entropy(logits[:, -1], gold[:, -1])
        else:
            loss = F.cross_entropy(logits.reshape(-1, P), gold.reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(lora_params + head_params, 1.0); opt.step()
        if step % 20 == 0:
            acc = (logits.argmax(-1) == gold).float().mean().item()
            print(f"step {step} R={d} loss={loss.item():.3f} stepacc={acc:.3f} t={time.time()-t0:.0f}s", flush=True)
            log.append({"step": step, "loss": loss.item(), "acc": acc})
    res = evaluate(net, evalset)
    for k, v in res.items():
        print(k, json.dumps({kk: (round(vv, 4) if isinstance(vv, float) else [round(x, 3) for x in vv]) for kk, vv in v.items()}), flush=True)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    tag = f"{a.arm}{'_term' if a.terminal_only else ''}_s{a.seed}"
    (out / f"result_{tag}.json").write_text(json.dumps({"args": vars(a), "trainable": ntrain, "train_log": log,
                                                        "eval": res, "seconds": time.time() - t0}, indent=1))


if __name__ == "__main__":
    main()
