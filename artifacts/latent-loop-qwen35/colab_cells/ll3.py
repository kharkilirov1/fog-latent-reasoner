import subprocess, time, os, json, glob
t0 = time.time()
def sh(cmd, timeout=None, show=40):
    print(f"\n$ {cmd[:220]}", flush=True)
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        out = (r.stdout + r.stderr).splitlines()
        print("\n".join(l[:400] for l in out[-show:] if "arning" not in l), flush=True)
        print(f"[rc={r.returncode}]", flush=True)
        return r.returncode
    except subprocess.TimeoutExpired:
        print("[TIMEOUT]", flush=True)
        return -1
os.makedirs("/content/ll3", exist_ok=True)
sh("rm -rf /content/mn && git clone -q --depth 1 --branch research/mxfp4-native-no-state --filter=blob:none --sparse https://github.com/kharkilirov1/memory-native /content/mn && cd /content/mn && git sparse-checkout set research/mxfp4-native-no-state/mxfp4-native && git log --oneline -1")
sh("rm -rf /content/fog && git clone -q --depth 1 --branch latent-loop-qwen35 https://github.com/kharkilirov1/fog-latent-reasoner /content/fog && cd /content/fog && git log --oneline -1")
sh("pip install -q -U flash-linear-attention hf_transfer 2>&1 | tail -2; python -c 'import fla; print(\"fla ok\")'", timeout=900)
sh("HF_HUB_ENABLE_HF_TRANSFER=1 python -c \"from huggingface_hub import snapshot_download as s; print(s('Qwen/Qwen3.5-4B'))\" 2>&1 | tail -1", timeout=1800)
MX6 = "/content/mn/research/mxfp4-native-no-state/mxfp4-native"
BASE = f"cd /content/fog && python -u experiments/latent_loop_qwen35_mx6.py --mx6 {MX6} --out /content/ll3 --lr 1e-5 --steps 600 "
arms = [("cap_adamw", "--arm cap --grad-ckpt --optim adamw"),
        ("snap_adamw", "--arm snap --grad-ckpt --optim adamw"),
        ("cap_adamw_tf", "--arm cap --grad-ckpt --optim adamw --tf 0.5"),
        ("cap_mx6", "--arm cap --grad-ckpt"),
        ("cot_mx6", "--arm cot")]
for tag, a in arms:
    if time.time() - t0 > 55 * 60:
        print("BUDGET STOP before", tag, flush=True); break
    sh(f"{BASE}{a} > /content/ll3/{tag}.log 2>&1; echo rc=$?", timeout=2400, show=2)
    sh(f"grep -v -i -E 'warn|falling back' /content/ll3/{tag}.log | grep -E '^codes|\"step\": (1|200|400|600),|_R(1|2|4|8|16) |DONE|Error|Traceback' | cut -c1-600 | tail -20", show=20)
    print(tag, "done at", round((time.time() - t0) / 60, 1), "min", flush=True)
for f in sorted(glob.glob("/content/ll3/result_*.json")):
    r = json.load(open(f))
    print("RESULT", os.path.basename(f), json.dumps({k: v.get("final") for k, v in r["eval"].items()}), flush=True)
print("ALL_DONE", round((time.time() - t0) / 60, 1), "min", flush=True)
