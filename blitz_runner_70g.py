import os
import subprocess
import itertools
import time
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

# ==========================================
# ☢️ 核动力引擎配置 (专为 70G 显存定制)
# ==========================================
GPUS = ['2']  # 🟢 精准指定空闲的卡 2
# 跑 1-shot 显存占用更小，3 个并发在 70G 上绝对稳，如闲置较多可调至 4 或 5
MAX_CONCURRENT_PER_GPU = 3  

# 基础护甲
PRETRAIN = "DGI" # 🟢 随时可改成 "DGI" 或 "GRACE"，脚本会自动区分输出文件！
TRAILS = 30
EPOCHS = 2000
PATIENCE = 20

# 搜索空间 
DATASETS = ["cora", "citeseer", "pubmed", "cornell", "texas", "wisconsin", "chameleon", "actor", "squirrel"]
METHODS = ["linear_probe", "fine_tune", "gpf", "gpf_plus", "edgeprompt", "edgeprompt_plus", "graphprompt", "all_in_one", "gppt"]
SHOTS = [1]  # 🟢 现阶段仅跑 1-shot

# 核心超参网格
LRS = [0.01, 0.005, 0.001]
WDS = [5e-4, 5e-5, 1e-5]

best_results = {}

# 🟢 动态生成排行榜文件名，防止不同预训练底座的数据互相覆盖
LEADERBOARD_FILE = f"Leaderboard_{PRETRAIN}_{SHOTS[0]}shot.md"

# ==========================================
# 🛡️ 核心调度逻辑
# ==========================================
def parse_accuracy(log_content):
    # 解析倒数 15 行内的准确率输出
    lines = log_content.strip().split('\n')[-15:]
    for line in lines:
        match = re.search(r'Accuracy:\s*([0-9.]+)\s*±\s*([0-9.]+)', line)
        if match:
            return float(match.group(1)), float(match.group(2))
    return None, None

def update_leaderboard():
    with open(LEADERBOARD_FILE, "w", encoding="utf-8") as f:
        # 🟢 标题栏高亮当前的预训练底座
        f.write(f"# 🏆 {PRETRAIN} - 1-Shot Baselines Leaderboard\n\n")
        f.write("| Dataset | Shot | Method | Best Acc | Best Std | Best LR | Best WD |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        
        sorted_keys = sorted(best_results.keys(), key=lambda x: (x[0], x[1], x[2]))
        for key in sorted_keys:
            ds, shot, method = key
            acc, std, lr, wd = best_results[key]
            f.write(f"| **{ds}** | {shot} | {method} | **{acc:.4f}** | ±{std:.4f} | {lr} | {wd} |\n")

def run_task(task):
    ds, method, shot, lr, wd, gpu_id = task
    # 🟢 独立的日志文件夹
    log_dir = f"logs_{PRETRAIN}_1shot_blitz" 
    os.makedirs(log_dir, exist_ok=True)
    # 🟢 独立且详细的日志文件名
    log_file = f"{log_dir}/{PRETRAIN}_{ds}_{method}_{shot}shot_lr{lr}_wd{wd}.txt"

    # 防弹断点续传
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
    
    # 挂载 GPU 环境变量并优化显存碎片分配
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
    print(f"🚀 ENGINE INITIATED: [ {PRETRAIN} ] Base | 1-Shot Only")
    print(f"GPUs Available: {GPUS} | Concurrent Workers: {MAX_CONCURRENT_PER_GPU}")
    print("==========================================================\n")

    all_combinations = list(itertools.product(DATASETS, METHODS, SHOTS, LRS, WDS))
    
    tasks = []
    for i, combo in enumerate(all_combinations):
        gpu_id = GPUS[i % len(GPUS)]
        tasks.append((*combo, gpu_id))
    
    total_tasks = len(tasks)
    print(f"🎯 Total Configurations to Annihilate: {total_tasks}\n")
    
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
                # 🟢 终端打印也会显示预训练模型名，防止你看花了眼
                print(f"{completed}/{total_tasks} {status_str} [{PRETRAIN}] | {ds} | {method} | lr={lr} wd={wd} ➡️ {acc:.4f}")
                
                # 更新打擂台最高分
                key = (ds, shot, method)
                if key not in best_results or acc > best_results[key][0]:
                    best_results[key] = (acc, std, lr, wd)
                    update_leaderboard()
            else:
                print(f"{completed}/{total_tasks} [ERROR] [{PRETRAIN}] | {ds} | {method} | lr={lr} wd={wd} (Check log)")

    print(f"\n🎉 ALL TASKS ANNIHILATED! Check '{LEADERBOARD_FILE}' for the absolute SOTAs.")

if __name__ == "__main__":
    main()