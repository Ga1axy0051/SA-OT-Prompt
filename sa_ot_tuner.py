import os
import subprocess
import itertools
import time
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

# ==========================================
# 👑 亲儿子专属·局部精确爆破引擎
# ==========================================
GPUS = ['3'] 
MAX_CONCURRENT_PER_GPU = 5  # 极其稳定

PRETRAIN = "GraphMAE"
TRAILS = 30
EPOCHS = 2000
PATIENCE = 20
METHODS = ["sa_ot_prompt"]
WDS = [5e-5, 5e-4] # weight decay 保持两个经典选项
NUM_PROMPTS = [5, 10] # 提示节点数（增加一点多样性）

# ==========================================
# 🎯 为每个数据集量身定制的搜索空间 (Based on Prior Knowledge)
# ==========================================
SPECIFIC_GRIDS = {
    'cora': {
        1: {'tau': [0.95, 0.99, 1.0], 'k': [3, 5, 8], 'lr': [0.0005, 0.001, 0.002], 'beta': [0.001, 0.005, 0.01]},
        5: {'tau': [0.5, 0.6, 0.7], 'k': [3, 4, 6], 'lr': [0.001, 0.002, 0.005], 'beta': [0.001, 0.005, 0.01]}
    },
    'citeseer': {
        1: {'tau': [0.8, 0.85, 0.9], 'k': [60, 80, 100], 'lr': [0.0001, 0.0005, 0.001], 'beta': [0.0001, 0.0005, 0.001]},
        5: {'tau': [0.4, 0.5, 0.6], 'k': [30, 45, 60], 'lr': [0.0005, 0.0008, 0.001], 'beta': [0.0001, 0.0005, 0.001]}
    },
    'pubmed': {
        1: {'tau': [0.9, 0.95, 0.99], 'k': [1, 3, 5], 'lr': [0.001, 0.005, 0.01], 'beta': [0.0001, 0.0005, 0.001]},
        5: {'tau': [0.7, 0.8, 0.9], 'k': [1, 3, 5], 'lr': [0.005, 0.01, 0.02], 'beta': [0.0005, 0.001, 0.005]}
    },
    'cornell': {
        1: {'tau': [0.0, 0.1], 'k': [30, 40, 50], 'lr': [0.005, 0.01, 0.02], 'beta': [0.0005, 0.001, 0.005]},
        5: {'tau': [0.0, 0.1], 'k': [30, 42, 50], 'lr': [0.0005, 0.0008, 0.001], 'beta': [0.0005, 0.001, 0.005]}
    },
    'texas': {
        1: {'tau': [0.0, 0.1], 'k': [20, 26, 35], 'lr': [0.005, 0.009, 0.015], 'beta': [0.0001, 0.0005, 0.001]},
        5: {'tau': [0.0, 0.1], 'k': [40, 48, 60], 'lr': [0.001, 0.003, 0.005], 'beta': [0.0001, 0.0005, 0.001]}
    },
    'wisconsin': {
        1: {'tau': [0.0, 0.1], 'k': [50, 70, 90], 'lr': [0.005, 0.008, 0.01], 'beta': [0.0001, 0.0005, 0.001]},
        5: {'tau': [0.0, 0.1], 'k': [70, 85, 100], 'lr': [0.0005, 0.0008, 0.001], 'beta': [0.0001, 0.0005, 0.001]}
    },
    'chameleon': {
        1: {'tau': [0.0, 0.1], 'k': [30, 40, 50], 'lr': [0.01, 0.015, 0.02], 'beta': [0.0005, 0.001, 0.005]},
        5: {'tau': [0.0, 0.1], 'k': [3, 5, 8], 'lr': [0.0001, 0.0005, 0.001], 'beta': [0.0005, 0.001, 0.005]}
    },
    'squirrel': {
        1: {'tau': [0.0, 0.1], 'k': [40, 55, 70], 'lr': [0.005, 0.009, 0.015], 'beta': [0.001, 0.005, 0.01]},
        5: {'tau': [0.0, 0.1], 'k': [25, 35, 45], 'lr': [0.005, 0.007, 0.01], 'beta': [0.001, 0.005, 0.01]}
    },
    'actor': {
        1: {'tau': [0.0, 0.1], 'k': [40, 60], 'lr': [0.005, 0.01], 'beta': [0.0005, 0.001]},
        5: {'tau': [0.0, 0.1], 'k': [40, 60], 'lr': [0.001, 0.005], 'beta': [0.0005, 0.001]}
    }
}

best_results = {}

def parse_accuracy(log_content):
    lines = log_content.strip().split('\n')[-15:]
    for line in lines:
        match = re.search(r'Accuracy:\s*([0-9.]+)\s*±\s*([0-9.]+)', line)
        if match:
            return float(match.group(1)), float(match.group(2))
    return None, None

def update_leaderboard():
    with open("SA_OT_SOTA_Leaderboard.md", "w", encoding="utf-8") as f:
        f.write("# 👑 SA-OT-Prompt Ultimate SOTA\n\n")
        f.write("| Dataset | Shot | Best Acc | Std | LR | WD | Tau | k | Beta | Prompts |\n")
        f.write("|---|---|---|---|---|---|---|---|---|---|\n")
        
        sorted_keys = sorted(best_results.keys(), key=lambda x: (x[0], x[1]))
        for key in sorted_keys:
            ds, shot = key
            acc, std, lr, wd, tau, k, beta, prompts = best_results[key]
            f.write(f"| **{ds}** | {shot} | **{acc:.4f}** | ±{std:.4f} | {lr} | {wd} | {tau} | {k} | {beta} | {prompts} |\n")

def run_task(task):
    ds, shot, lr, wd, tau, k, beta, prompts, gpu_id = task
    log_dir = "logs_sa_ot_fine"
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = f"{log_dir}/{ds}_{shot}s_lr{lr}_wd{wd}_tau{tau}_k{k}_beta{beta}_p{prompts}.txt"

    if os.path.exists(log_file):
        with open(log_file, "r", encoding="utf-8", errors='ignore') as f:
            acc, std = parse_accuracy(f.read())
            if acc is not None:
                return task, acc, std, True 

    cmd = [
        "python", "-u", "main.py",
        "--dataset", ds,
        "--method", "sa_ot_prompt",
        "--model", PRETRAIN,
        "--shot", str(shot),
        "--down_lr", str(lr),
        "--down_wd", str(wd),
        "--tau", str(tau),
        "--k", str(k),
        "--ot_beta", str(beta),
        "--num_prompts", str(prompts),
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
    print("👑 SA-OT-PROMPT PRECISE BOMBING INITIATED")
    print(f"GPUs Available: {GPUS} | Concurrent Workers: {MAX_CONCURRENT_PER_GPU}")
    print("==========================================================\n")

    tasks = []
    task_idx = 0
    
    # 动态组装每个数据集独特的搜索空间
    for ds in SPECIFIC_GRIDS.keys():
        for shot in [1, 5]:
            grid = SPECIFIC_GRIDS[ds][shot]
            # 生成该数据集/shot专属的所有组合
            combos = list(itertools.product(
                [ds], [shot], grid['lr'], WDS, grid['tau'], grid['k'], grid['beta'], NUM_PROMPTS
            ))
            for combo in combos:
                gpu_id = GPUS[task_idx % len(GPUS)]
                tasks.append((*combo, gpu_id))
                task_idx += 1
    
    total_tasks = len(tasks)
    print(f"🎯 Tailored Configurations to Explore: {total_tasks}\n")
    
    max_workers = len(GPUS) * MAX_CONCURRENT_PER_GPU
    
    completed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {executor.submit(run_task, task): task for task in tasks}
        
        for future in as_completed(future_to_task):
            task, acc, std, is_cached = future.result()
            ds, shot, lr, wd, tau, k, beta, prompts, gpu_id = task
            completed += 1
            
            status_str = "[CACHED]" if is_cached else "[DONE]"
            if acc is not None:
                # 如果这个组合打败了当前数据集的最优记录，就在终端打印出来并且大声提示！
                key = (ds, shot)
                if key not in best_results or acc > best_results[key][0]:
                    print(f"🔥 NEW SOTA for {ds.upper()} {shot}-shot! Acc: {acc:.4f} (lr={lr}, tau={tau}, k={k}, beta={beta})")
                    best_results[key] = (acc, std, lr, wd, tau, k, beta, prompts)
                    update_leaderboard()
                elif not is_cached and completed % 20 == 0:
                    # 避免刷屏，每跑完20个任务打印一次进度
                    print(f"{completed}/{total_tasks} explored...")
            else:
                print(f"{completed}/{total_tasks} [ERROR] {ds} | {shot}s (Check log)")

    print("\n🎉 ALL TUNING COMPLETED! Check 'SA_OT_SOTA_Leaderboard.md'.")

if __name__ == "__main__":
    main()