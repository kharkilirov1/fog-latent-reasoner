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
# We define a high-capacity FOG v3 backbone
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
    # Note: In a real Kaggle environment, we would clone the repo or upload scripts
    # For this script, we assume the environment is set up with the Fog-latent-reasoner files
    
    print("Starting Scale-Up Pretraining (50,000 steps)...")
    
    # Command to run the pretraining
    # We use a larger batch size (e.g., 4) and seq_len (1024) for the "Gold" run
    cmd = [
        sys.executable, "train_real.py", "pretrain",
        "--architecture", "register_machine_v3",
        "--tokenizer", "tokenizer/tinystories_3k_bpe.json", # Using the existing tokenizer for consistency
        "--checkpoint-dir", "checkpoints/gold_pretrain",
        "--device", "cuda",
        "--precision", "bf16",
        "--max-steps", "50000",
        "--batch-size", "4",
        "--gradient-accumulation", "16", # Effective batch size = 64
        "--sequence-length", "1024",
        "--lr", "0.0004",
        "--warmup-steps", "2000",
        "--eval-every", "5000",
        "--save-every", "5000",
        "--log-every", "10",
        "--dataset-id", "HuggingFaceFW/fineweb-edu", # Primary source
        "--dataset-config", "sample-10BT",
        "--shuffle-buffer", "5000"
    ]
    
    print(f"Executing: {' '.join(cmd)}")
    subprocess.run(cmd)

if __name__ == "__main__":
    # Save the config for reference
    os.makedirs("configs", exist_ok=True)
    with open("configs/v3_backbone_50k.json", "w") as f:
        json.dump(CONFIG, f, indent=4)
        
    run_training()
