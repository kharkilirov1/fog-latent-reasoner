import subprocess, time, os
t0 = time.time()
MX6 = "/content/mn/research/mxfp4-native-no-state/mxfp4-native"
os.makedirs("/content/ll6", exist_ok=True)
cmd = (f"cd /content/fog && python -u experiments/latent_loop_qwen35_mx6.py --mx6 {MX6} --out /content/ll6 --lr 1e-5 "
       "--eval-depths 1,2,4,8,16,32 --arm msnap --grad-ckpt --optim adamw --steps 800 > /content/ll6_msnap.log 2>&1")
subprocess.run(cmd, shell=True, timeout=2400)
r = subprocess.run("grep -v -i -E 'warn|falling' /content/ll6_msnap.log | grep -E '\"step\": (200|400|600|800),|_R|DONE|Error|Traceback' | cut -c1-900 | tail -20",
                   shell=True, capture_output=True, text=True)
print(r.stdout, flush=True)
print("ALL_DONE", round((time.time() - t0) / 60, 1), "min", flush=True)
