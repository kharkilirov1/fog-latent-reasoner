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
os.makedirs("/content/ll4", exist_ok=True)
sh("cd /content/fog && git fetch -q origin latent-loop-qwen35 && git checkout -q FETCH_HEAD && git log --oneline -1")
MX6 = "/content/mn/research/mxfp4-native-no-state/mxfp4-native"
BASE = (f"cd /content/fog && python -u experiments/latent_loop_qwen35_mx6.py --mx6 {MX6} --out /content/ll4 --lr 1e-5 "
        "--eval-depths 1,2,4,8,16,32 ")
arms = [("mcap_adamw", "--arm mcap --grad-ckpt --optim adamw --steps 800"),
        ("msnap_adamw", "--arm msnap --grad-ckpt --optim adamw --steps 800"),
        ("mcap_mx6", "--arm mcap --grad-ckpt --steps 800"),
        ("cot_mx6", "--arm cot --steps 600 --eval-depths 1,2,4,8,12,16")]
for tag, a in arms:
    if time.time() - t0 > 55 * 60:
        print("BUDGET STOP before", tag, flush=True); break
    sh(f"{BASE}{a} > /content/ll4/{tag}.log 2>&1; echo rc=$?", timeout=2400, show=2)
    sh(f"grep -v -i -E 'warn|falling back' /content/ll4/{tag}.log | grep -E '^codes|\"step\": (1|200|400|600|800),|_R|DONE|Error|Traceback' | cut -c1-700 | tail -24", show=24)
    print(tag, "done at", round((time.time() - t0) / 60, 1), "min", flush=True)
for f in sorted(glob.glob("/content/ll4/result_*.json")):
    r = json.load(open(f))
    print("RESULT", os.path.basename(f), json.dumps({k: [v.get("final"), v.get("swap_follows_latent")] for k, v in r["eval"].items()}), flush=True)
print("ALL_DONE", round((time.time() - t0) / 60, 1), "min", flush=True)
