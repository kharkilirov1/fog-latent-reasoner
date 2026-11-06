import subprocess, time, json, glob, os
t0 = time.time()
MX6 = "/content/mn/research/mxfp4-native-no-state/mxfp4-native"
cmd = (f"cd /content/fog && python -u experiments/latent_loop_qwen35_mx6.py --mx6 {MX6} --out /content/ll5 --lr 1e-5 "
       "--arm cot --steps 600 --eval-depths 1,2,4,8,12,16 > /content/ll5_cot.log 2>&1")
os.makedirs("/content/ll5", exist_ok=True)
subprocess.run(cmd, shell=True, timeout=1800)
r = subprocess.run("grep -E '_R|DONE|Error|Traceback' /content/ll5_cot.log | tail -14", shell=True, capture_output=True, text=True)
print(r.stdout, flush=True)
print("ALL_DONE", round((time.time() - t0) / 60, 1), "min", flush=True)
