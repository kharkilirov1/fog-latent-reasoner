"""Builds every figure of the paper from repository artifacts."""
import json, pathlib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
R = pathlib.Path(__file__).resolve().parents[1]; OUT = R / "paper" / "figures"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": .3, "figure.dpi": 150})
BLUE, ORANGE, GREEN, GREY = "#2b6cb0", "#dd6b20", "#2f855a", "#718096"

# Fig 1: one-step exact vs recurrent stability
d = json.load(open(R / "artifacts/research/stability_diagnostic/seed22.json"))
fig, ax = plt.subplots(figsize=(4.6, 2.6))
for row, c, lab in ((d["rows"][0], BLUE, "law-compatible operator"), (d["rows"][1], ORANGE, "flexible operator")):
    tr = row["metrics"]["trajectory"]; xs = sorted(map(int, tr))
    m = row["metrics"]
    ax.plot(xs, [100 * tr[str(x)]["accuracy"] for x in xs], "o-", c=c, ms=3.5,
            label=f"{lab} (gain p95 {m['local_gain_p95']:.2f})")
ax.set_xscale("log", base=2); ax.set_xticks([1, 2, 4, 8, 16, 32, 64]); ax.set_xticklabels([1, 2, 4, 8, 16, 32, 64])
ax.set_xlabel("recurrence depth"); ax.set_ylabel("accuracy (%)"); ax.legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "fig_onestep_vs_recurrent.pdf")

# Fig 2: capture rescues learned flexible operators
runs = json.load(open(R / "artifacts/capture-learned-operator/results.json"))
fig, ax = plt.subplots(figsize=(4.6, 3.1))
depths = [1, 8, 16, 32, 64, 128, 256]
for mode, c, lab in (("raw", ORANGE, "no capture"), ("cap4", GREY, r"capture $\beta=4$ (below $\beta^*$)"),
                     ("cap16", GREEN, r"capture $\beta=16$"), ("cap64", BLUE, r"capture $\beta=64\geq\beta^*$")):
    ys = [100 * sum(r[mode][str(t)] for r in runs) / len(runs) for t in depths]
    ax.plot(depths, ys, marker="o", c=c, ms=3.5, label=lab, ls="--" if mode == "cap16" else "-")
ax.set_xscale("log", base=2); ax.set_xticks(depths); ax.set_xticklabels(depths)
ax.set_xlabel("recurrence depth"); ax.set_ylabel("exact trajectories (%)")
ax.legend(frameon=False, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=2)
fig.tight_layout(); fig.savefig(OUT / "fig_capture_rescue.pdf")

# Fig 3: width-sharpness trade-off
W = json.load(open(R / "artifacts/capture-width-sharpness/results.json"))
fig, ax = plt.subplots(figsize=(4.6, 2.8))
D = [r["Delta"] for r in W]
ax.scatter(D, [r["beta_star"] for r in W], c=ORANGE, s=16, label=r"sufficient $\beta^*$ (theory)", zorder=3)
ax.scatter(D, [r["beta_emp"] for r in W], c=BLUE, s=16, marker="s", label=r"smallest exact $\beta$ (grid)", zorder=3)
import numpy as np
xs = np.logspace(np.log10(0.015), np.log10(1.2), 50)
ax.plot(xs, 4 / xs, c=GREY, lw=1, ls=":", label=r"$\propto \Delta^{-1}$")
for dd in sorted({r["d"] for r in W}):
    r = next(x for x in W if x["d"] == dd)
    lab, off = (f"d={dd}", (4, 4)) if dd < 16 else (("d=16,32", (5, -11)) if dd == 16 else (None, None))
    if lab: ax.annotate(lab, (r["Delta"], r["beta_star"]), textcoords="offset points", xytext=off, fontsize=7)
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel(r"code gap $\Delta$"); ax.set_ylabel(r"inverse temperature $\beta$")
ax.legend(frameon=False, fontsize=7.5); fig.tight_layout(); fig.savefig(OUT / "fig_width_sharpness.pdf")

# Fig 4: linear width threshold
rows = json.load(open(R / "artifacts/research/exp_022/metrics.json"))["rows"]
fig, ax = plt.subplots(figsize=(4.0, 2.4))
by = {}
for r in rows: by.setdefault(r["dimension"], []).append(100 * r["metrics"]["mixed_program_accuracy"])
ks = sorted(by)
ax.plot(ks, [sum(by[k]) / len(by[k]) for k in ks], "-", c=BLUE, lw=1)
for k in ks: ax.scatter([k] * len(by[k]), by[k], c=BLUE, s=14, zorder=3)
ax.axhline(100 / 31, c=GREY, ls=":", lw=1); ax.text(16.3, 8, "chance", fontsize=7, color=GREY)
ax.set_xticks(ks); ax.set_xlabel("register width d"); ax.set_ylabel("mixed programs (%)")
fig.tight_layout(); fig.savefig(OUT / "fig_linear_width.pdf")
print("figures written to", OUT)
