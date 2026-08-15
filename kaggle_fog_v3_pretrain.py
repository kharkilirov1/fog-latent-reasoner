import os
import sys
import subprocess

# 1. Setup Environment & Install Dependencies
print("Installing dependencies...")
subprocess.run([sys.executable, "-m", "pip", "install", "datasets", "tokenizers", "tqdm", "torch"], check=True)

import torch
from pathlib import Path
import json

# 2. Configuration for the 50k "Gold" Pretraining
CONFIG = {
    "vocab_size": 8192,
    "d_model": 512,
    "n_heads": 8,
    "n_layers": 6,
    "d_ff": 2048,
    "max_seq_len": 2048,
    "dropout": 0.1,
    "latent_slots": 16,
    "reasoning_steps": 8,
    "compare_rank": 128,
    "planner_ff": 1024,
    "memory_slots": 64,
    "architecture_version": "register_machine_v3",
    "binding_mode": "query_conditioned",
    "readout_mode": "direct_latent",
    "protected_binding_slots": 16,
    "binding_offsets": [1, 2, 4, 8, 16, 32],
    "binding_query_update": "primary_recurrent",
    "machine_operator_count": 16,
    "machine_operator_rank": 64,
    "machine_ff": 1024,
    "machine_min_steps": 2,
    "machine_hard_routing": True
}

# 3. Kaggle Training Script Logic
def run_training():
    print("Cloning repository...")
    repo_url = "https://github.com/kharkilirov1/fog-latent-reasoner.git"
    # Clone into a subdirectory to avoid mixing with Kaggle working dir
    subprocess.run(["git", "clone", repo_url, "repo"], check=True)
    os.chdir("repo")
    
    print("Starting Scale-Up Pretraining (50,000 steps)...")
    
    # Command to run the pretraining
    cmd = [
        sys.executable, "train_real.py", "pretrain",
        "--architecture", "register_machine_v3",
        "--tokenizer", "tokenizer/tinystories_3k_bpe.json",
        "--checkpoint-dir", "/kaggle/working/checkpoints/gold_pretrain",
        "--device", "cuda",
        "--precision", "bf16",
        "--max-steps", "50000",
        "--batch-size", "4",
        "--gradient-accumulation", "16",
        "--sequence-length", "1024",
        "--lr", "0.0004",
        "--warmup-steps", "2000",
        "--eval-every", "5000",
        "--save-every", "5000",
        "--log-every", "10",
        "--dataset-id", "HuggingFaceFW/fineweb-edu",
        "--dataset-config", "sample-10BT",
        "--shuffle-buffer", "5000"
    ]
    
    print(f"Executing: {' '.join(cmd)}")
    # Run and ensure output is visible
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in process.stdout:
        print(line, end="")
    process.wait()

if __name__ == "__main__":
    # Save the config in the Kaggle output directory
    os.makedirs("/kaggle/working/configs", exist_ok=True)
    with open("/kaggle/working/configs/v3_backbone_50k.json", "w") as f:
        json.dump(CONFIG, f, indent=4)
        
    run_training()
