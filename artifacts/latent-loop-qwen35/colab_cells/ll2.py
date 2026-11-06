import subprocess, time, os, json, glob
t0 = time.time()
def sh(cmd, timeout=None, show=40):
    print(f"\n$ {cmd[:220]}", flush=True)
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        out = (r.stdout + r.stderr).splitlines()
        print("\n".join(l[:300] for l in out[-show:] if "arning" not in l), flush=True)
        print(f"[rc={r.returncode}]", flush=True)
        return r.returncode
    except subprocess.TimeoutExpired:
        print("[TIMEOUT]", flush=True)
        return -1
os.makedirs("/content/ll2", exist_ok=True)
sh("cd /content/fog && git fetch -q origin latent-loop-qwen35 && git checkout -q FETCH_HEAD && git log --oneline -1")
sh("nvidia-smi --query-gpu=memory.used --format=csv")
MX6 = "/content/mn/research/mxfp4-native-no-state/mxfp4-native"
BASE = f"cd /content/fog && python -u experiments/latent_loop_qwen35_mx6.py --mx6 {MX6} --out /content/ll2 "
def run(tag, args, timeout=3600):
    sh(f"{BASE}{args} > /content/ll2/{tag}.log 2>&1; echo rc=$?", timeout=timeout, show=2)
    sh(f"grep -v -i -E 'warn|falling back' /content/ll2/{tag}.log | grep -E 'mx6:|arm=|\"step\": (1|20|100|200|300|400|500|600),|_R|BEFORE|DONE|Error|Traceback' | tail -40", show=40)
def final(tag_glob):
    fs = glob.glob(f"/content/ll2/result_{tag_glob}.json")
    if not fs: return -1
    ev = json.load(open(fs[0]))["eval"]
    return sum(v["final"] for k, v in ev.items() if k.endswith(("R1", "R2", "R4"))) / max(1, sum(k.endswith(("R1", "R2", "R4")) for k in ev))
# ---- 1) lr probe on the CoT arm (fast, and it checks the text eval)
probe = {}
for lr in ("1e-5", "3e-5", "1e-4"):
    out = f"/content/ll2/p{lr}"
    sh(f"mkdir -p {out}")
    tag = f"probe_cot_{lr}"
    sh(f"{BASE.replace('/content/ll2 ', out + ' ')}--arm cot --lr {lr} --steps 400 --eval-n 32 --eval-depths 1,2,4 {'--eval-before' if lr == '1e-5' else ''} > /content/ll2/{tag}.log 2>&1", timeout=1800)
    sh(f"grep -E '_R|BEFORE|\"step\": (20|400),|Error' /content/ll2/{tag}.log | tail -12", show=12)
    fs = glob.glob(f"{out}/result_*.json")
    if fs:
        ev = json.load(open(fs[0]))["eval"]
        probe[lr] = sum(v["final"] for v in ev.values()) / len(ev)
print("PROBE", probe, round((time.time() - t0) / 60, 1), "min", flush=True)
best = max(probe, key=probe.get) if probe else "2e-5"
print("BEST LR", best, flush=True)
# ---- 2) arms
arms = [("cap", "--arm cap --grad-ckpt"), ("cont", "--arm cont --grad-ckpt"), ("cap_mid", "--arm cap_mid --grad-ckpt"),
        ("cot", "--arm cot"), ("direct", "--arm direct"), ("cap_adamw", "--arm cap --grad-ckpt --optim adamw")]
for tag, a in arms:
    if time.time() - t0 > 140 * 60:
        print("BUDGET STOP before", tag, flush=True); break
    run(tag, f"{a} --lr {best} --steps 600", timeout=3600)
    print(tag, "done at", round((time.time() - t0) / 60, 1), "min", flush=True)
for f in sorted(glob.glob("/content/ll2/result_*.json")):
    r = json.load(open(f))
    print("RESULT", os.path.basename(f), json.dumps({k: v.get("final") for k, v in r["eval"].items()}), r["seconds"], flush=True)
print("ALL_DONE", round((time.time() - t0) / 60, 1), "min", flush=True)
