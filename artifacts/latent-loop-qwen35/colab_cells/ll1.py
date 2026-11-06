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
os.makedirs("/content/ll", exist_ok=True)
sh("rm -rf /content/mn && git clone -q --depth 1 --branch research/mxfp4-native-no-state --filter=blob:none --sparse https://github.com/kharkilirov1/memory-native /content/mn && cd /content/mn && git sparse-checkout set research/mxfp4-native-no-state/mxfp4-native && git log --oneline -1")
sh("rm -rf /content/fog && git clone -q --depth 1 --branch latent-loop-qwen35 https://github.com/kharkilirov1/fog-latent-reasoner /content/fog && cd /content/fog && git log --oneline -1")
sh("pip install -q -U transformers flash-linear-attention hf_transfer 2>&1 | tail -2; python -c 'import transformers,torch; print(transformers.__version__, torch.__version__); import fla; print(\"fla ok\")'", timeout=900)
sh("HF_HUB_ENABLE_HF_TRANSFER=1 python -c \"from huggingface_hub import snapshot_download as s; print(s('Qwen/Qwen3.5-4B'))\" 2>&1 | tail -1", timeout=1800)
MX6 = "/content/mn/research/mxfp4-native-no-state/mxfp4-native"
BASE = f"cd /content/fog && python -u experiments/latent_loop_qwen35_mx6.py --mx6 {MX6} --out /content/ll "
def run(tag, args, timeout=3600):
    sh(f"{BASE}{args} > /content/ll/{tag}.log 2>&1; echo rc=$?", timeout=timeout, show=2)
    sh(f"grep -v -i -E 'warn|falling back' /content/ll/{tag}.log | grep -E 'model |mx6:|arm=|\"step\": (1|20|40|60|80|100|200|300|400|500|600),|_R|DONE|Error|error|Traceback|OOM' | tail -40", show=40)
# ---- 1) lr probe on the capture arm (short)
probe = {}
for lr in ("3e-4", "1e-3", "3e-3"):
    tag = f"probe_cap_lr{lr}"
    run(tag, f"--arm cap --steps 80 --lr {lr} --eval-n 16 --eval-depths 2,4", timeout=1800)
    try:
        last = [json.loads(l) for l in open(f"/content/ll/{tag}.log") if l.startswith('{"step"')][-1]
        probe[lr] = last["acc"]
    except Exception as e:
        print("probe parse failed", e, flush=True)
print("PROBE", probe, round((time.time() - t0) / 60, 1), "min", flush=True)
best = max(probe, key=probe.get) if probe else "1e-3"
print("BEST LR", best, flush=True)
# ---- 2) arms (budget guard: no new arm after 150 min)
arms = [("cap", "--arm cap"), ("cont", "--arm cont"), ("cap_mid", "--arm cap_mid"), ("cot", "--arm cot"),
        ("direct", "--arm direct"), ("cap_adamw", "--arm cap --optim adamw --lr 1e-5")]
for tag, a in arms:
    if time.time() - t0 > 150 * 60:
        print("BUDGET STOP before", tag, flush=True); break
    lr = "" if "adamw" in tag else f" --lr {best}"
    run(tag, f"{a}{lr} --steps 600", timeout=3600)
    print(tag, "done at", round((time.time() - t0) / 60, 1), "min", flush=True)
for f in sorted(glob.glob("/content/ll/result_*.json")):
    r = json.load(open(f))
    print("RESULT", os.path.basename(f), json.dumps({k: v.get("final") for k, v in r["eval"].items()}), r["seconds"], flush=True)
print("ALL_DONE", round((time.time() - t0) / 60, 1), "min", flush=True)
