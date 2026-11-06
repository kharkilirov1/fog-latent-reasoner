import subprocess, time, os, json, glob
t0 = time.time()
def sh(cmd, timeout=None, show=40):
    print(f"\n$ {cmd[:220]}", flush=True)
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        out = (r.stdout + r.stderr).splitlines()
        print("\n".join(l[:900] for l in out[-show:] if "arning" not in l), flush=True)
        print(f"[rc={r.returncode}]", flush=True)
    except subprocess.TimeoutExpired:
        print("[TIMEOUT]", flush=True)
os.makedirs("/content/ll7", exist_ok=True)
sh("rm -rf /content/mn && git clone -q --depth 1 --branch research/mxfp4-native-no-state --filter=blob:none --sparse https://github.com/kharkilirov1/memory-native /content/mn && cd /content/mn && git sparse-checkout set research/mxfp4-native-no-state/mxfp4-native && git log --oneline -1")
sh("rm -rf /content/fog && git clone -q --depth 1 --branch latent-loop-qwen35 https://github.com/kharkilirov1/fog-latent-reasoner /content/fog && cd /content/fog && git log --oneline -1")
sh("pip install -q -U flash-linear-attention hf_transfer 2>&1 | tail -2; python -c 'import fla; print(\"fla ok\")'", timeout=900)
sh("HF_HUB_ENABLE_HF_TRANSFER=1 python -c \"from huggingface_hub import snapshot_download as s; print(s('Qwen/Qwen3.5-4B'))\" 2>&1 | tail -1", timeout=1800)
MX6 = "/content/mn/research/mxfp4-native-no-state/mxfp4-native"
BASE = (f"cd /content/fog && python -u experiments/latent_loop_qwen35_mx6.py --mx6 {MX6} --out /content/ll7 --lr 1e-5 "
        "--eval-depths 1,2,4,8,16,32 --arm msnap --grad-ckpt --tf 0.5 --steps 1200 ")
for tag, a in [("msnap_adamw_tf", "--optim adamw"), ("msnap_mx6_tf", "")]:
    if time.time() - t0 > 60 * 60:
        print("BUDGET STOP before", tag, flush=True); break
    sh(f"{BASE}{a} > /content/ll7/{tag}.log 2>&1; echo rc=$?", timeout=3000, show=2)
    sh(f"grep -v -i -E 'warn|falling back' /content/ll7/{tag}.log | grep -E '\"step\": (1|300|600|900|1200),|_R|DONE|Error|Traceback' | tail -20", show=20)
    print(tag, "done at", round((time.time() - t0) / 60, 1), "min", flush=True)
print("ALL_DONE", round((time.time() - t0) / 60, 1), "min", flush=True)
