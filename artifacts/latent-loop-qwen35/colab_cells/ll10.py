import subprocess, time, os
t0 = time.time()
def sh(cmd, timeout=None, show=40):
    print(f"\n$ {cmd[:220]}", flush=True)
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        out = (r.stdout + r.stderr).splitlines()
        print("\n".join(l[:1000] for l in out[-show:] if "arning" not in l), flush=True)
        print(f"[rc={r.returncode}]", flush=True)
    except subprocess.TimeoutExpired:
        print("[TIMEOUT]", flush=True)
os.makedirs("/content/ll10", exist_ok=True)
sh("cd /content/fog && git fetch -q origin latent-loop-qwen35 && git checkout -q FETCH_HEAD && git log --oneline -1")
MX6 = "/content/mn/research/mxfp4-native-no-state/mxfp4-native"
cmd = (f"cd /content/fog && python -u experiments/latent_loop_qwen35_mx6.py --mx6 {MX6} --out /content/ll10 --lr 1e-5 "
       "--eval-depths 1,2,4,8,16,32 --arm msnap --per-step-op --code-init fourier --grad-ckpt --optim adamw "
       "--steps 1200 --onestep-steps 400 > /content/ll10/ps_fourier.log 2>&1; echo rc=$?")
sh(cmd, timeout=3000, show=2)
sh("grep -v -i -E 'warn|falling back' /content/ll10/ps_fourier.log | grep -E '^codes|\"step\": (1|400|800|1200),|_R|DONE|Error|Traceback' | cut -c1-1000 | tail -20", show=20)
print("ALL_DONE", round((time.time() - t0) / 60, 1), "min", flush=True)
