import os
from datasets import load_dataset
import json
from tqdm import tqdm

def download_and_save():
    # 1. FineWeb-Edu (Educational quality)
    print("Loading FineWeb-Edu...")
    fw_edu = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    
    # 2. OpenWebText (Web diversity)
    print("Loading OpenWebText...")
    owt = load_dataset("Skylion007/openwebtext", split="train", streaming=True)
    
    # 3. Code (Logic/Structure)
    print("Loading Code (The Stack sample)...")
    code = load_dataset("bigcode/the-stack-dedup", split="train", streaming=True)

    output_file = "data/train_gold_500k.jsonl"
    os.makedirs("data", exist_ok=True)
    
    count = 0
    limit_per_source = 166666 # Total ~500k samples for the "Gold" dataset
    
    with open(output_file, "w", encoding="utf-8") as f:
        # Mix FineWeb-Edu
        print(f"Sampling {limit_per_source} from FineWeb-Edu...")
        for i, item in enumerate(tqdm(fw_edu, total=limit_per_source)):
            if i >= limit_per_source: break
            f.write(json.dumps({"text": item["text"]}, ensure_ascii=False) + "\n")
            count += 1
            
        # Mix OpenWebText
        print(f"Sampling {limit_per_source} from OpenWebText...")
        for i, item in enumerate(tqdm(owt, total=limit_per_source)):
            if i >= limit_per_source: break
            f.write(json.dumps({"text": item["text"]}, ensure_ascii=False) + "\n")
            count += 1
            
        # Mix Code
        print(f"Sampling {limit_per_source} from Code...")
        # Note: Code might require more processing, but we take the raw text for pretraining
        for i, item in enumerate(tqdm(code, total=limit_per_source)):
            if i >= limit_per_source: break
            f.write(json.dumps({"text": item["content"]}, ensure_ascii=False) + "\n")
            count += 1

    print(f"Done! Saved {count} samples to {output_file}")

if __name__ == "__main__":
    download_and_save()
