"""Toward real data: the winning Qwen3.5 latent-loop recipe (Markov, one operation per step, hard snap, operator-compatible
code chart, one-step warm-up) on WHOLE NUMBERS 0..N-1 (default 1000) instead of Z_31, with word-problem phrasings.

Code chart: a multi-period "helix" (cos/sin of 2 pi v / T for many periods T, cf. Kantamneni & Tegmark 2025: LLMs represent
numbers on such helices and add by rotating them). Operations: add/subtract 1..99, multiply by 2..9, always kept inside
[0, N). Reuses experiments/latent_loop_qwen35_mx6.py (MarkovLoop, training loop, eval) by swapping its task globals.
Usage: same flags as latent_loop_qwen35_mx6.py plus --n-values; use --arm msnap --per-step-op --code-init fourier.
"""
import math
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import latent_loop_qwen35_mx6 as L  # noqa: E402

N = 1000
for i, x in enumerate(sys.argv):
    if x == "--n-values":
        N = int(sys.argv[i + 1])
        del sys.argv[i:i + 2]
        break
PERIODS = [2, 3, 4, 5, 7, 8, 9, 10, 11, 13, 16, 20, 25, 32, 50, 64, 100, 128, 200, 250, 500, 1000, 2000]
MULW = {2: "doubled", 3: "tripled", 4: "quadrupled"}

STYLES_TRAIN = ("base", "story")
STYLE_HELD = "alt"


def helix_codes():
    v = torch.arange(N, dtype=torch.float64)[:, None]
    t = torch.tensor([T for T in PERIODS if T <= 2 * N], dtype=torch.float64)[None]
    a = 2 * math.pi * v / t
    f = torch.cat([a.cos(), a.sin()], 1)
    return (f / f.norm(dim=1, keepdim=True)).float()


def op_text(kind, k, style):
    if style == "base":
        return {"add": f"Add {k}.", "sub": f"Subtract {k}.", "mul": f"Multiply by {k}."}[kind]
    if style == "story":
        return {"add": f"The box receives {k} more coins.", "sub": f"{k} coins are taken out of the box.",
                "mul": f"The number of coins in the box is {MULW.get(k, f'multiplied by {k}')}."}[kind]
    return {"add": f"Now add {k} more.", "sub": f"Now take away {k}.", "mul": f"Now take {k} times the current value."}[kind]


def make_ex(rng, depth, style):
    cur = rng.randrange(N)
    start, ops, vals = cur, [], []
    for _ in range(depth):
        while True:
            kind = rng.choice(("add", "sub", "mul"))
            k = rng.randint(1, 99) if kind != "mul" else rng.randint(2, 9)
            nxt = cur + k if kind == "add" else cur - k if kind == "sub" else cur * k
            if 0 <= nxt < N:
                break
        ops.append((kind, k))
        vals.append(nxt)
        cur = nxt
    return {"start": start, "ops": ops, "vals": [start] + vals, "depth": depth, "style": style}


def prompt_step(e, t):
    now = "Read the current value." if t == 0 else op_text(*e["ops"][t - 1], e["style"])
    return f"Work with whole numbers. Now: {now} Think silently."


def prompt_ops(e):
    return "Work with whole numbers. " + "".join(f"Step {i + 1}: {op_text(k, a, e['style'])} "
                                                for i, (k, a) in enumerate(e["ops"])) + "Think silently."


if __name__ == "__main__":
    L.P = N
    L.make_ex, L.prompt_step, L.prompt_ops, L.fourier_codes = make_ex, prompt_step, prompt_ops, helix_codes
    L.STYLES_TRAIN, L.STYLE_HELD = STYLES_TRAIN, STYLE_HELD
    f = helix_codes()
    g = f @ f.t()
    g.fill_diagonal_(-2)
    print(f"helix chart: {N} values, {f.shape[1]} dims, max cos between codes {g.max().item():.4f}", flush=True)
    L.main()
