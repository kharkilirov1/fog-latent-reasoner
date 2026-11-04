"""Latent-loop reasoning in a modern pretrained LLM (Qwen3.5), FULL fine-tune with the mx6 latent grid.

Tests the paper's predictions (paper/ "When Is a Latent Thought Reusable?", sec. 8) at a scale the SmolLM2 pilot could not
reach: P2 (capture flattens the depth curve), P3 (one shared read-out), P4 (mid-stack tap beats the final layer).

Task: the pilot's natural-language modular programs (latent_loop_llm_experiment.make_ex / prompt_text, Z_31).
Every decoder nn.Linear becomes an mx6 LatentGridLinear (6-bit index on an NVFP4-aligned grid, update inside backward, no
fp32 master, no Adam); per-weight momentum (bf16) in sparse+reuse mode, because the latent loop calls every layer T+1 times
in one graph: the gradients of all calls are summed, ONE update per layer per optimizer step. Norms / conv / gates and the
small loop heads use AdamW; embeddings and the tied lm_head are frozen.

Arms
  direct   answer only, as text                        (no reasoning baseline)
  cot      " Step 1: v1. ... Answer: vT." as text       (explicit chain of thought)
  cont     latent loop, x_{t+1} = rescale(W h_t)        (continuous feedback, Coconut-like)
  cap      latent loop, x_{t+1} = sum_v softmax(beta cos(U h_t, K_v)) C_v   (Hopfield capture, beta annealed to --beta-max)
  cap_mid  cap, but h_t read from a mid-stack layer (--tap) instead of the last one (P4)

Latent step t (t = 0..T): append the marker " Step t:" and the latent input x_t, run the decoder over the WHOLE sequence
(no KV cache: exact gradients through every layer type, incl. the Gated DeltaNet recurrent state), read h_t at x_t.
Step 0 reads the start value. Supervision: CE of the shared read-out on the value after every step (P3).
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from latent_loop_llm_experiment import P, STYLE_HELD, STYLES_TRAIN, make_ex, prompt_text  # noqa: E402

LATENT_ARMS = ("cont", "cap", "cap_mid")
TEXT_ARMS = ("direct", "cot")


# ── mx6 conversion ────────────────────────────────────────────────────────────────────────────────────────────────────

class LGBias(nn.Module):
    def __init__(self, lg, bias):
        super().__init__()
        self.lg, self.bias = lg, nn.Parameter(bias)

    def forward(self, x):
        return self.lg(x) + self.bias


def convert_mx6(model, lr: float, mom_dtype: str, k: int = 6, headroom: float = 2.0) -> dict:
    """Every nn.Linear inside a decoder layer -> LatentGridLinear (in_features % 16 == 0); momentum sparse+reuse."""
    from mx6 import lg_accum
    from mx6.latent_grid import LatentGridLinear
    todo = []
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and ".layers." in f".{name}" and "lm_head" not in name and "visual" not in name:
            if mod.in_features % 16 == 0:
                todo.append((name, mod))
    n_par = 0
    for name, mod in todo:
        parent = model.get_submodule(name.rsplit(".", 1)[0])
        lg = LatentGridLinear.from_dense(mod.weight.detach(), k=k, headroom=headroom, lr=lr, backend="auto",
                                         update_mode="exact", rms_beta=0.9, rms_eps=1e-3).to(mod.weight.device)
        new = LGBias(lg, mod.bias.detach().float()) if mod.bias is not None else lg
        setattr(parent, name.rsplit(".", 1)[1], new)
        n_par += mod.weight.numel()
        del mod
    st = lg_accum.set_momentum(model, 0.9, mom_dtype, sparse=True, reuse=True)
    return {"n_layers": len(todo), "params_in_lg": n_par, "momentum_bytes": lg_accum.accum_resident_bytes(model),
            "state": st}


def set_counter_lr(model, lr: float):
    from mx6.latent_grid import lg_layers
    for m in lg_layers(model):
        if hasattr(m, "set_counter_lr"):
            m.set_counter_lr(lr)
        else:
            m.lr = lr
            if getattr(m, "_lr_t", None) is not None:
                m._lr_t.fill_(lr)


# ── latent loop ───────────────────────────────────────────────────────────────────────────────────────────────────────

class Loop(nn.Module):
    def __init__(self, model, tok, arm: str, tap: int, dev, code_dim: int = 128):
        super().__init__()
        self.model, self.tok, self.arm, self.dev = model, tok, arm, dev
        emb = model.get_input_embeddings().weight
        H = emb.shape[1]
        self.emb_norm = emb.float().norm(dim=-1).mean().item()
        self.tap = tap if arm == "cap_mid" else -1
        self.U = nn.Linear(H, code_dim, bias=False)
        self.K = nn.Parameter(torch.randn(P, code_dim) / math.sqrt(code_dim))
        self.log_s = nn.Parameter(torch.tensor(math.log(10.0)))          # CE temperature of the cosine read-out
        # value codes C_v: initialised from the embeddings of the number tokens " 0" .. " 30"
        ids = [tok.encode(f" {v}", add_special_tokens=False)[0] for v in range(P)]
        self.C = nn.Parameter(emb[ids].detach().float().clone())
        self.begin = nn.Parameter(emb[tok.encode(" Start", add_special_tokens=False)[0]].detach().float().clone())
        self.W = nn.Linear(H, H)
        nn.init.eye_(self.W.weight)
        nn.init.zeros_(self.W.bias)
        self.beta = 1.0
        self._mk = {}
        self.to(dev)

    def rescale(self, x):
        return x * (self.emb_norm / (x.norm(dim=-1, keepdim=True) + 1e-6))

    def cos(self, h):
        return F.normalize(self.U(h.float()), dim=-1) @ F.normalize(self.K, dim=-1).t()

    def marker(self, t):
        if t not in self._mk:
            self._mk[t] = torch.tensor(self.tok.encode(f" Step {t}:", add_special_tokens=False), device=self.dev)
        return self._mk[t]

    def feedback(self, h, c):
        if self.arm == "cont":
            return self.rescale(self.W(h.float()))
        p = F.softmax(self.beta * c, -1)
        return self.rescale(p @ self.rescale(self.C))

    def forward(self, exs, T, track=False):
        embed = self.model.get_input_embeddings()
        pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.tok.eos_token_id
        ids = [self.tok.encode(prompt_text(e), add_special_tokens=False) for e in exs]
        Lp = max(map(len, ids))
        inp = torch.full((len(ids), Lp), pad, dtype=torch.long, device=self.dev)
        am = torch.zeros_like(inp)
        for i, a in enumerate(ids):                                        # left pad
            inp[i, Lp - len(a):] = torch.tensor(a, device=self.dev)
            am[i, Lp - len(a):] = 1
        dt = embed.weight.dtype
        seq = embed(inp)
        B = len(exs)
        x = self.rescale(self.begin.float()).expand(B, -1)
        logits_all, xs = [], []
        for t in range(T + 1):
            mk = embed(self.marker(t)).unsqueeze(0).expand(B, -1, -1)
            seq = torch.cat([seq, mk, x.to(dt).unsqueeze(1)], 1)
            am = torch.cat([am, torch.ones(B, mk.shape[1] + 1, dtype=am.dtype, device=self.dev)], 1)
            out = self.model.model(inputs_embeds=seq, attention_mask=am, output_hidden_states=self.tap >= 0,
                                   use_cache=False)
            h = (out.hidden_states[self.tap] if self.tap >= 0 else out.last_hidden_state)[:, -1]
            c = self.cos(h)
            logits_all.append(c * self.log_s.exp())
            if track:
                xs.append(x.detach())
            x = self.feedback(h, c)
        return torch.stack(logits_all, 1), xs


# ── text arms ─────────────────────────────────────────────────────────────────────────────────────────────────────────

def target_text(e, arm):
    if arm == "direct":
        return f" Answer: {e['vals'][-1]}."
    return "".join(f" Step {i + 1}: {v}." for i, v in enumerate(e["vals"][1:])) + f" Answer: {e['vals'][-1]}."


def text_loss(model, tok, exs, arm, dev):
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    rows, tg = [], []
    for e in exs:
        a = tok.encode(prompt_text(e), add_special_tokens=False)
        b = tok.encode(target_text(e, arm), add_special_tokens=False) + [tok.eos_token_id]
        rows.append(a + b)
        tg.append(len(a))
    L = max(map(len, rows))
    inp = torch.full((len(rows), L), pad, dtype=torch.long, device=dev)
    am = torch.zeros_like(inp)
    lab = torch.full_like(inp, -100)
    for i, r in enumerate(rows):                                           # right pad for training
        inp[i, :len(r)] = torch.tensor(r, device=dev)
        am[i, :len(r)] = 1
        lab[i, tg[i]:len(r)] = inp[i, tg[i]:len(r)]
    h = model.model(input_ids=inp, attention_mask=am, use_cache=False).last_hidden_state[:, :-1]
    tgt = lab[:, 1:]
    sel = tgt != -100
    logits = model.lm_head(h[sel]).float()
    return F.cross_entropy(logits, tgt[sel])


@torch.no_grad()
def text_eval(model, tok, exs, arm, dev, bs=32):
    tok.padding_side = "left"
    ok = 0
    for i in range(0, len(exs), bs):
        b = exs[i:i + bs]
        enc = tok([prompt_text(e) for e in b], return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
        n_new = 8 if arm == "direct" else 7 * (b[0]["depth"] + 1) + 8
        out = model.generate(**enc, max_new_tokens=n_new, do_sample=False, pad_token_id=tok.pad_token_id)
        for e, row in zip(b, out[:, enc["input_ids"].shape[1]:]):
            m = re.search(r"Answer:\s*(\d+)", tok.decode(row, skip_special_tokens=True))
            ok += int(m is not None and int(m.group(1)) == e["vals"][-1])
    return {"final": ok / len(exs)}


@torch.no_grad()
def latent_eval(net, exs, bs=32):
    d = exs[0]["depth"]
    ok = allok = 0
    tube = [0.0] * (d + 1)
    C = net.rescale(net.C)
    for i in range(0, len(exs), bs):
        b = exs[i:i + bs]
        logits, xs = net(b, d, track=True)
        pred = logits.argmax(-1).cpu()
        gold = torch.tensor([e["vals"] for e in b])
        ok += (pred[:, -1] == gold[:, -1]).sum().item()
        allok += (pred == gold).all(-1).sum().item()
        for t in range(1, d + 1):                    # fed-back state vs the code of the previous step's value
            tube[t] += ((xs[t].float() - C[gold[:, t - 1].to(C.device)]).norm(dim=-1) / net.emb_norm).sum().item()
    n = len(exs)
    return {"final": ok / n, "all_steps": allok / n, "tube": [round(v / n, 4) for v in tube]}


# ── main ──────────────────────────────────────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--arm", default="cap", choices=TEXT_ARMS + LATENT_ARMS)
    ap.add_argument("--mx6", default=None, help="path to memory-native/research/mxfp4-native-no-state/mxfp4-native")
    ap.add_argument("--optim", default="mx6", choices=["mx6", "adamw"], help="decoder linears: mx6 latent grid or AdamW")
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5,
                    help="mx6 counter lr (or AdamW lr of the decoder with --optim adamw). The RMS-normalised lg step moves "
                         "every weight by ~lr per step, like Adam: fine-tuning needs Adam-scale lr (3e-4 wrecked Qwen3.5)")
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--small-lr", type=float, default=1e-5, help="AdamW lr of norms / conv / gates")
    ap.add_argument("--mom-dtype", default="bf16", choices=["bf16", "fp8", "fp32"])
    ap.add_argument("--max-train-depth", type=int, default=6)
    ap.add_argument("--beta-max", type=float, default=30.0)
    ap.add_argument("--tap", type=int, default=-1, help="cap_mid: hidden_states index (default 2/3 of the depth)")
    ap.add_argument("--eval-n", type=int, default=64)
    ap.add_argument("--eval-depths", default="1,2,4,6,8,12,16")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    ap.add_argument("--out", default="artifacts/latent-loop-qwen35")
    ap.add_argument("--tiny", action="store_true", help="CPU smoke test: random tiny Qwen3.5 text model")
    ap.add_argument("--grad-ckpt", action="store_true", help="activation checkpointing per decoder layer (latent loop: T+1 "
                    "full-sequence forwards with grad)")
    ap.add_argument("--eval-before", action="store_true", help="text arms: evaluate the untouched model first")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    random.seed(a.seed)
    if a.mx6:
        sys.path.insert(0, a.mx6)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dt = torch.bfloat16 if a.dtype == "bf16" else torch.float32
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    if a.tiny:
        from transformers import Qwen3_5ForCausalLM
        from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
        cfg = Qwen3_5TextConfig(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                                num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                                layer_types=["linear_attention"] * 3 + ["full_attention"],
                                linear_num_value_heads=4, linear_num_key_heads=2, linear_key_head_dim=16,
                                linear_value_head_dim=16, tie_word_embeddings=True)
        model = Qwen3_5ForCausalLM(cfg).to(dev, dt)
        load_info = {"missing_keys": [], "unexpected_keys": []}
    else:
        model, load_info = AutoModelForCausalLM.from_pretrained(a.model, dtype=dt, output_loading_info=True)
        model.to(dev)
    miss = [k for k in load_info["missing_keys"] if "visual" not in k]
    print(f"model {a.model}: {type(model).__name__} params={sum(p.numel() for p in model.parameters()) / 1e9:.2f}B "
          f"missing={len(miss)} unexpected={len(load_info['unexpected_keys'])}", flush=True)
    if miss:
        raise SystemExit(f"checkpoint keys missing (would train a random model): {miss[:5]}")
    nl = model.config.num_hidden_layers if not hasattr(model.config, "text_config") else model.config.text_config.num_hidden_layers
    tap = a.tap if a.tap >= 0 else round(2 * nl / 3)

    for p in model.parameters():
        p.requires_grad_(False)
    rep = {}
    if a.optim == "mx6":
        rep = convert_mx6(model, a.lr, a.mom_dtype)
        print(f"mx6: {rep['n_layers']} decoder linears -> latent grid ({rep['params_in_lg'] / 1e9:.2f}B params), "
              f"momentum {rep['momentum_bytes'] / 2 ** 30:.2f} GB ({a.mom_dtype}, sparse+reuse)", flush=True)
    emb_w = model.get_input_embeddings().weight
    small = [p for n, p in model.named_parameters()
             if p is not emb_w and "lm_head" not in n and "embed" not in n and "visual" not in n
             and ".lg." not in n and not n.endswith(".idx")]
    dec_linear = []
    if a.optim == "adamw":
        dec_linear = [m.weight for n, m in model.named_modules() if isinstance(m, nn.Linear) and ".layers." in f".{n}"]
        small = [p for p in small if all(p is not w for w in dec_linear)]
    for p in small + dec_linear:
        p.requires_grad_(True)
    net = Loop(model, tok, a.arm, tap, dev) if a.arm in LATENT_ARMS else None
    groups = [{"params": small, "lr": a.small_lr}]
    if dec_linear:
        groups.append({"params": dec_linear, "lr": a.lr})
    if net is not None:
        groups.append({"params": [p for n, p in net.named_parameters() if not n.startswith("model.")], "lr": a.head_lr})
    opt = torch.optim.AdamW(groups, weight_decay=0.0)
    print(f"arm={a.arm} optim={a.optim} tap={tap if a.arm == 'cap_mid' else '-'} small={sum(p.numel() for p in small)} "
          f"adamw_dec={sum(p.numel() for p in dec_linear)}", flush=True)

    from mx6.lg_accum import lg_accumulate
    rng, erng = random.Random(1000 + a.seed), random.Random(777)
    depths = [int(x) for x in a.eval_depths.split(",")]
    evalset = {}
    for d in depths:
        evalset[f"id_R{d}"] = [make_ex(erng, d, erng.choice(STYLES_TRAIN)) for _ in range(a.eval_n)]
        evalset[f"held_R{d}"] = [make_ex(erng, d, STYLE_HELD) for _ in range(a.eval_n)]

    pre = {}
    if a.eval_before and net is None:
        model.eval()
        for k in [k for k in evalset if k.endswith(("R1", "R2", "R4"))]:
            pre[k] = text_eval(model, tok, evalset[k], a.arm, dev)
            print("BEFORE", k, json.dumps(pre[k]), flush=True)
    if a.grad_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.config.use_cache = False
    model.train()
    t0, log = time.time(), []
    warm = max(1, a.steps // 20)
    for step in range(1, a.steps + 1):
        frac = step / a.steps
        lr_mult = min(1.0, step / warm) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * frac)))
        if a.optim == "mx6":
            set_counter_lr(model, a.lr * lr_mult)
        for g, base in zip(opt.param_groups, [a.small_lr] + ([a.lr] if dec_linear else []) + ([a.head_lr] if net else [])):
            g["lr"] = base * lr_mult
        d = rng.randint(1, min(a.max_train_depth, 1 + int(a.max_train_depth * min(1.0, 1.5 * frac))))
        b = [make_ex(rng, d, rng.choice(STYLES_TRAIN)) for _ in range(a.bs)]
        if net is not None and a.arm != "cont":
            net.beta = 1.0 + (a.beta_max - 1.0) * min(1.0, frac / 0.6)
        ctx = lg_accumulate(model, 1) if a.optim == "mx6" else None
        if ctx is not None:
            with ctx as acc:
                for _ in acc.micro_batches():
                    loss, acc_v = _loss(net, model, tok, b, d, a.arm, dev)
                    loss.backward()
        else:
            loss, acc_v = _loss(net, model, tok, b, d, a.arm, dev)
            loss.backward()
        torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if step % 20 == 0 or step == 1:
            rec = {"step": step, "depth": d, "loss": round(loss.item(), 4), "acc": round(acc_v, 4),
                   "beta": round(net.beta, 2) if net else None, "t": round(time.time() - t0, 1),
                   "peak_gb": round(torch.cuda.max_memory_allocated() / 2 ** 30, 1) if dev == "cuda" else 0,
                   "lg_updates": rep["state"].n_updates if rep else None}
            log.append(rec)
            print(json.dumps(rec), flush=True)

    model.eval()
    res = {}
    for k, exs in evalset.items():
        r = latent_eval(net, exs) if net is not None else text_eval(model, tok, exs, a.arm, dev)
        res[k] = r
        print(k, json.dumps(r), flush=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{a.arm}_{a.optim}_s{a.seed}"
    (out / f"result_{tag}.json").write_text(json.dumps(
        {"args": vars(a), "before": pre, "mx6": {k: v for k, v in rep.items() if k != "state"}, "train_log": log, "eval": res,
         "seconds": round(time.time() - t0, 1)}, indent=1))
    print("DONE", tag, round(time.time() - t0, 1), "s", flush=True)


def _loss(net, model, tok, b, d, arm, dev):
    if net is None:
        return text_loss(model, tok, b, arm, dev), float("nan")
    logits, _ = net(b, d)
    gold = torch.tensor([e["vals"] for e in b], device=logits.device)
    loss = F.cross_entropy(logits.reshape(-1, P), gold.reshape(-1))
    return loss, (logits.argmax(-1) == gold).float().mean().item()


if __name__ == "__main__":
    main()
