import os
import subprocess
import itertools
import time
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

# ==========================================
# 🛡️ 边缘提示复活专属引擎 (EdgePrompt Resurrection)
# ==========================================
GPUS = ['3'] 
MAX_CONCURRENT_PER_GPU = 5  

PRETRAIN = "GraphMAE"
TRAILS = 30
EPOCHS = 2000
PATIENCE = 20

# 搜索空间：只锁定这两个“大病初愈”的兄弟！
DATASETS = ["cora", "citeseer", "pubmed", "cornell", "texas", "wisconsin", "chameleon", "actor", "squirrel"]
METHODS = ["edgeprompt", "edgeprompt_plus"]
SHOTS = [1, 5]

# 经典大网格
LRS = [0.01, 0.005, 0.001]
WDS = [5e-4, 5e-5, 1e-5]

best_results = {}

def parse_accuracy(log_content):
    lines = log_content.strip().split('\n')[-15:]
    for line in lines:
        match = re.search(r'Accuracy:\s*([0-9.]+)\s*±\s*([0-9.]+)', line)
        if match:
            return float(match.group(1)), float(match.group(2))
    return None, None

def update_leaderboard():
    with open("EdgePrompt_Fixed_Leaderboard.md", "w", encoding="utf-8") as f:
        f.write("# 🛡️ EdgePrompt Fixed SOTA\n\n")
        f.write("| Dataset | Shot | Method | Best Acc | Best Std | Best LR | Best WD |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        
        sorted_keys = sorted(best_results.keys(), key=lambda x: (x[0], x[1], x[2]))
        for key in sorted_keys:
            ds, shot, method = key
            acc, std, lr, wd = best_results[key]
            f.write(f"| **{ds}** | {shot} | {method} | **{acc:.4f}** | ±{std:.4f} | {lr} | {wd} |\n")

def run_task(task):
    ds, method, shot, lr, wd, gpu_id = task
    # 🚨 使用全新的日志文件夹，防止读取到之前那些死掉或者撞分的假缓存！
    log_dir = "logs_edgeprompt_fix"
    os.makedirs(log_dir, exist_ok=True)
    log_file = f"{log_dir}/{ds}_{method}_{shot}shot_lr{lr}_wd{wd}.txt"

    if os.path.exists(log_file):
        with open(log_file, "r", encoding="utf-8", errors='ignore') as f:
            acc, std = parse_accuracy(f.read())
            if acc is not None:
                return task, acc, std, True 

    cmd = [
        "python", "-u", "main.py",
        "--dataset", ds,
        "--method", method,
        "--model", PRETRAIN,
        "--shot", str(shot),
        "--down_lr", str(lr),
        "--down_wd", str(wd),
        "--epochs", str(EPOCHS),
        "--patience", str(PATIENCE),
        "--trails", str(TRAILS)
    ]
    
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_id
    env["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

    try:
        with open(log_file, "w") as f:
            process = subprocess.Popen(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)
            process.wait()
        
        with open(log_file, "r", encoding="utf-8", errors='ignore') as f:
            acc, std = parse_accuracy(f.read())
            return task, acc, std, False
            
    except Exception as e:
        with open(log_file, "a") as f:
            f.write(f"\n[CRASH/OOM ERROR]: {str(e)}\n")
        return task, None, None, False

def main():
    print("==========================================================")
    print("🛡️ EDGEPROMPT RESURRECTION ENGINE INITIATED")
    print(f"GPUs Available: {GPUS} | Concurrent Workers: {MAX_CONCURRENT_PER_GPU}")
    print("==========================================================\n")

    all_combinations = list(itertools.product(DATASETS, METHODS, SHOTS, LRS, WDS))
    
    tasks = []
    for i, combo in enumerate(all_combinations):
        gpu_id = GPUS[i % len(GPUS)]
        tasks.append((*combo, gpu_id))
    
    total_tasks = len(tasks)
    print(f"🎯 Total Configurations to Re-evaluate: {total_tasks}\n")
    
    max_workers = len(GPUS) * MAX_CONCURRENT_PER_GPU
    
    completed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {executor.submit(run_task, task): task for task in tasks}
        
        for future in as_completed(future_to_task):
            task, acc, std, is_cached = future.result()
            ds, method, shot, lr, wd, gpu_id = task
            completed += 1
            
            status_str = "[CACHED]" if is_cached else "[DONE]"
            if acc is not None:
                print(f"{completed}/{total_tasks} {status_str} {ds} | {shot}s | {method} ➡️ {acc:.4f} (lr={lr}, wd={wd})")
                
                key = (ds, shot, method)
                if key not in best_results or acc > best_results[key][0]:
                    best_results[key] = (acc, std, lr, wd)
                    update_leaderboard()
            else:
                print(f"{completed}/{total_tasks} [ERROR] {ds} | {shot}s | {method} | lr={lr} wd={wd} (Check log)")

    print("\n🎉 EDGEPROMPT HAS BEEN RESURRECTED! Check 'EdgePrompt_Fixed_Leaderboard.md'.")

if __name__ == "__main__":
    main()