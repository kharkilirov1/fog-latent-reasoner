"""Width-sharpness trade-off with learned operators (NOTE-016, Prop. E).

Affine grammar over F_31 (A: x->x+1, M: x->3x). For each width d, N=31 codes are
spread on S^{d-1} (d=2: regular polygon; d>2: repulsion minimising max cosine).
A flexible MLP F(h, op) is fitted on the 62 one-step pairs. We measure the
closure defect eps, the local gain L (spectral norm of the Jacobian at codes),
the code gap Delta = 1 - max_{i!=j} <m_i, m_j>, the sufficient inverse
temperature beta* of Theorem A', and the smallest beta on a geometric grid
for which the captured loop h <- Pi_beta(F(h, op)) is exact through depth 256."""
import json, math, pathlib, torch, torch.nn as nn, torch.nn.functional as Fn
torch.set_num_threads(2)
P = 31; DEPTH = 256; B = 512
NXT = torch.stack([(torch.arange(P) + 1) % P, (3 * torch.arange(P)) % P])
BETAS = [2 ** (k / 2) for k in range(0, 27)]  # 1 .. 8192

def spread_codes(d, seed):
    if d == 2:
        th = 2 * math.pi * torch.arange(P) / P
        return torch.stack([th.cos(), th.sin()], 1)
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(P, d, generator=g).requires_grad_(True)
    opt = torch.optim.Adam([X], 1e-2)
    for _ in range(3000):
        C = Fn.normalize(X, dim=-1); G = C @ C.T - 3 * torch.eye(P)
        loss = torch.logsumexp(40 * G.flatten(), 0) / 40
        opt.zero_grad(); loss.backward(); opt.step()
    return Fn.normalize(X.detach(), dim=-1)

def beta_star(eps, L, Delta):
    if eps >= Delta / 2: return math.inf
    Lb = max(L, 1.0); best = math.inf
    for k in range(1, 2000):
        s = eps + (Delta / 2 - eps) * k / 2000
        if Delta - 2 * s <= 0: continue
        best = min(best, max(0.0, math.log(2 * Lb * (P - 1) / (s - eps))) / (Delta - 2 * s))
    return best

def rollout(F, C, mode, beta, ops, i0):
    h = C[i0]; i = i0.clone(); alive = torch.ones(len(i0), dtype=torch.bool)
    with torch.no_grad():
        for t in range(DEPTH):
            h = F(h, ops[t]); i = NXT[ops[t], i]
            if mode == "cap": h = torch.softmax(beta * h @ C.T, -1) @ C
            elif mode == "snap": h = C[(h @ C.T).argmax(-1)]
            alive &= ((h @ C.T).argmax(-1) == i)
    return alive.float().mean().item()

def run(d, seed):
    C = spread_codes(d, seed)
    G = C @ C.T; Delta = 1 - (G - 3 * torch.eye(P)).max().item()
    torch.manual_seed(seed)
    net = nn.Sequential(nn.Linear(d + 2, 256), nn.GELU(), nn.Linear(256, 256), nn.GELU(), nn.Linear(256, d))
    F = lambda h, op: net(torch.cat([h, Fn.one_hot(op, 2).float()], -1))
    X = C.repeat(2, 1); OP = torch.arange(2).repeat_interleave(P); Y = C[NXT.reshape(-1)]
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 6000)
    for _ in range(6000):
        loss = ((F(X, OP) - Y) ** 2).sum(-1).mean(); opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    with torch.no_grad():
        out = F(X, OP); eps = (out - Y).norm(dim=-1).max().item()
        one = ((out @ C.T).argmax(-1) == NXT.reshape(-1)).float().mean().item()
    L = max(torch.linalg.matrix_norm(torch.autograd.functional.jacobian(
        lambda h: F(h[None], OP[k:k + 1])[0], X[k]), 2).item() for k in range(2 * P))
    gen = torch.Generator().manual_seed(1000 + seed)
    ops = torch.randint(0, 2, (DEPTH, B), generator=gen); i0 = torch.randint(0, P, (B,), generator=gen)
    raw = rollout(F, C, "raw", 0, ops, i0); snap = rollout(F, C, "snap", 0, ops, i0)
    sweep = {round(b, 2): rollout(F, C, "cap", b, ops, i0) for b in BETAS}
    b_emp = next((b for b, a in sweep.items() if a == 1.0), math.inf)
    return {"d": d, "seed": seed, "Delta": Delta, "eps": eps, "L": L, "one_step": one,
            "beta_star": beta_star(eps, L, Delta), "beta_emp": b_emp, "raw_d256": raw,
            "snap_d256": snap, "sweep": sweep}

if __name__ == "__main__":
    out = []
    for d in (2, 3, 4, 8, 16, 32):
        for seed in (0, 1, 2):
            r = run(d, seed); out.append(r)
            print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k != "sweep"}), flush=True)
    p = pathlib.Path("artifacts/capture-width-sharpness"); p.mkdir(parents=True, exist_ok=True)
    (p / "results.json").write_text(json.dumps(out, indent=1))
