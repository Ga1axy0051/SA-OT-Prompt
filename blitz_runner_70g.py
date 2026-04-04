import os
import subprocess
import itertools
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

# ==========================================
# ⚖️ 全量基线 & 全量底座 扫表引擎 (3-Shot)
# ==========================================
GPUS = ['0', '1', '2', '3']  # 可用 GPU 列表
MAX_CONCURRENT_PER_GPU = 6  

PRETRAINS = ["GraphMAE2"]
SHOTS = [1] # 🟢 锁死 3-shot

TRAILS = 30         
EPOCHS = 2000       
PATIENCE = 20       

# 数据集全集
DATASETS = ["cora", "citeseer", "pubmed", "cornell", "texas", "wisconsin", "chameleon", "actor", "squirrel"]

METHODS = [
    "linear_probe", "fine_tune", "gpf", "gpf_plus", 
    "edgeprompt", "edgeprompt_plus", "graphprompt", "all_in_one", "gppt"
]

LRS = [0.01, 0.005, 0.001]
WDS = [5e-4, 5e-5, 1e-5]

# 全局字典记录最好成绩: key 变成了 (pretrain, dataset, method)
best_results = {}

def parse_accuracy(log_content):
    lines = log_content.strip().split('\n')[-20:]
    for line in lines:
        match = re.search(r'Accuracy:\s*([0-9.]+)\s*±\s*([0-9.]+)', line)
        if match:
            return float(match.group(1)), float(match.group(2))
    return None, None

def update_leaderboards():
    # 🟢 升级 2：动态为每个底座生成专属的 Leaderboard
    for pt in PRETRAINS:
        lb_file = f"Leaderboard_{pt}_{SHOTS[0]}shot_Baselines.md"
        with open(lb_file, "w", encoding="utf-8") as f:
            f.write(f"# 🏆 {pt} - {SHOTS[0]}-Shot Baselines Leaderboard (Patience=20)\n\n")
            f.write("| Dataset | Method | Best Acc | Best Std | Best LR | Best WD |\n")
            f.write("|---|---|---|---|---|---|\n")
            
            # 过滤出当前底座的数据，并按数据集和方法排序
            sorted_keys = sorted([k for k in best_results.keys() if k[0] == pt], key=lambda x: (x[1], x[2]))
            for key in sorted_keys:
                _, ds, method = key
                acc, std, lr, wd = best_results[key]
                f.write(f"| **{ds}** | {method} | **{acc:.4f}** | ±{std:.4f} | {lr} | {wd} |\n")

def run_task(task):
    # 🟢 升级 3：解析 task 时加入 pretrain
    pretrain, ds, method, shot, lr, wd, gpu_id = task
    log_dir = f"logs_baselines_{pretrain}_{shot}shot" 
    os.makedirs(log_dir, exist_ok=True)
    log_file = f"{log_dir}/{ds}_{method}_lr{lr}_wd{wd}.txt"

    if os.path.exists(log_file):
        with open(log_file, "r", encoding="utf-8", errors='ignore') as f:
            acc, std = parse_accuracy(f.read())
            if acc is not None:
                return task, acc, std, True 

    cmd = [
        "python", "-u", "main.py",
        "--dataset", ds, "--method", method, "--model", pretrain,
        "--shot", str(shot), "--down_lr", str(lr), "--down_wd", str(wd),
        "--epochs", str(EPOCHS), "--patience", str(PATIENCE), "--trails", str(TRAILS)
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
    except Exception:
        return task, None, None, False

def main():
    print(f"🚀 MEGA BASELINE SWEEP INITIATED: Backbones={PRETRAINS}, Shot={SHOTS[0]}")
    # 🟢 升级 4：网格生成器加入 PRETRAINS
    all_combinations = list(itertools.product(PRETRAINS, DATASETS, METHODS, SHOTS, LRS, WDS))
    tasks = [(*combo, GPUS[i % len(GPUS)]) for i, combo in enumerate(all_combinations)]
    
    print(f"📦 Total Tasks Scheduled: {len(tasks)}")

    with ThreadPoolExecutor(max_workers=len(GPUS) * MAX_CONCURRENT_PER_GPU) as executor:
        future_to_task = {executor.submit(run_task, task): task for task in tasks}
        for i, future in enumerate(as_completed(future_to_task)):
            task, acc, std, is_cached = future.result()
            pretrain, ds, method, shot, lr, wd, _ = task
            if acc is not None:
                print(f"{i+1}/{len(tasks)} [{pretrain}] {ds} | {method} | {lr}/{wd} -> {acc:.4f}")
                key = (pretrain, ds, method)
                # 更新最高分逻辑
                if key not in best_results or acc > best_results[key][0]:
                    best_results[key] = (acc, std, lr, wd)
                    update_leaderboards()

if __name__ == "__main__":
    main()