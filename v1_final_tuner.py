import os
import subprocess
import itertools
import time
import re
import json
import threading
import queue
import random

# ==========================================
# 👑 GraphMAE (v1) 绝对公平·微雕爆破引擎 (动态耐心排队版)
# ==========================================
GPUS = ['5']  # 你的显卡
# 🟢 并发上限调到 3，配合错峰发车，绝对安全且高效
MAX_CONCURRENT_PER_GPU = 3   

# 🔴 核心纪律：所有架构参数彻底锁死，不再动摇！
PRETRAIN = "GraphMAE"
TRAILS = 30             
EPOCHS = 2000
PATIENCE = 100           
WDS = [5e-5]            
NUM_PROMPTS = [10]      

# ==========================================
# 🎯 V1 第二轮：中继平滑过渡网格 (Gradual Narrowing)
# 核心策略：解开死锁，在最优值两侧建立缓冲带
# ==========================================

SPECIFIC_GRIDS = {
    # ================= 强异配图：K=120 绝不能锁死，向两边看 =================
    # Texas, Wisconsin, Actor, Chameleon 1s 的最佳 K 都在 120。
    # 策略：以 120 为中心，探 80 和 160。LR 在 0.001 和 0.05 之间插入过渡值。
    'wisconsin': {
        1: {'tau': [0.0, 0.1, 0.2], 'k': [80, 120, 160], 'lr': [0.0005, 0.001, 0.005], 'beta': [0.0005, 0.001, 0.005]},
        3: {'tau': [0.0, 0.1, 0.2], 'k': [80, 120, 160], 'lr': [0.0005, 0.001, 0.005], 'beta': [0.0005, 0.001, 0.005]},
        5: {'tau': [0.0, 0.1, 0.2], 'k': [80, 120, 160], 'lr': [0.0005, 0.001, 0.005], 'beta': [0.0005, 0.001, 0.005]}
    },
    'texas': {
        1: {'tau': [0.0, 0.1, 0.2], 'k': [80, 120, 160], 'lr': [0.005, 0.01, 0.05], 'beta': [0.0005, 0.001, 0.005]},
        3: {'tau': [0.0, 0.1, 0.2], 'k': [80, 120, 160], 'lr': [0.005, 0.01, 0.05], 'beta': [0.005, 0.01, 0.05]},
        5: {'tau': [0.0, 0.1, 0.2], 'k': [40, 60, 80], 'lr': [0.01, 0.05, 0.08], 'beta': [0.005, 0.01, 0.05]}
    },
    'actor': {
        1: {'tau': [0.0, 0.1, 0.2], 'k': [80, 120, 160], 'lr': [0.01, 0.05, 0.08], 'beta': [0.0005, 0.001, 0.005]},
        3: {'tau': [0.0, 0.1, 0.2], 'k': [80, 120, 160], 'lr': [0.005, 0.01, 0.05], 'beta': [0.005, 0.01, 0.05]},
        5: {'tau': [0.0, 0.1, 0.2], 'k': [80, 120, 160], 'lr': [0.0005, 0.001, 0.005], 'beta': [0.05, 0.1, 0.2]}
    },
    'chameleon': {
        1: {'tau': [0.0, 0.1, 0.2], 'k': [80, 120, 160], 'lr': [0.0005, 0.001, 0.005], 'beta': [0.005, 0.01, 0.05]},
        3: {'tau': [0.0, 0.1, 0.2], 'k': [40, 60, 80], 'lr': [0.0005, 0.001, 0.005], 'beta': [0.05, 0.1, 0.2]},
        5: {'tau': [0.0, 0.1, 0.2], 'k': [10, 20, 40], 'lr': [0.0005, 0.001, 0.005], 'beta': [0.05, 0.1, 0.2]}
    },

    # ================= 异配回调图：锁定中等范围 =================
    # Cornell 5s 在 60，Squirrel 在 20~60。
    'cornell': {
        1: {'tau': [0.0, 0.1, 0.2], 'k': [40, 60, 80], 'lr': [0.0005, 0.001, 0.005], 'beta': [0.005, 0.01, 0.05]},
        3: {'tau': [0.0, 0.1, 0.2], 'k': [40, 60, 80], 'lr': [0.005, 0.01, 0.05], 'beta': [0.0005, 0.001, 0.005]},
        5: {'tau': [0.0, 0.1, 0.2], 'k': [40, 60, 80], 'lr': [0.01, 0.05, 0.08], 'beta': [0.0005, 0.001, 0.005]}
    },
    'squirrel': {
        1: {'tau': [0.0, 0.1, 0.2], 'k': [40, 60, 80], 'lr': [0.005, 0.01, 0.05], 'beta': [0.005, 0.01, 0.05]},
        3: {'tau': [0.0, 0.1, 0.2], 'k': [20, 40, 60], 'lr': [0.005, 0.01, 0.05], 'beta': [0.0005, 0.001, 0.005]},
        5: {'tau': [0.0, 0.1, 0.2], 'k': [10, 20, 40], 'lr': [0.005, 0.01, 0.05], 'beta': [0.0005, 0.001, 0.005]}
    },

    # ================= 强同配图：不要错过近邻 =================
    # Cora, Pubmed 最佳是 K=2，Citeseer 最佳是 K=50。
    # 策略：增加 1, 5, 10 阶梯，防止过拟合到 K=2。
    'cora': {
        1: {'tau': [0.5, 0.8, 1.0], 'k': [1, 2, 5, 10], 'lr': [0.0005, 0.001, 0.005], 'beta': [0.001, 0.005, 0.01]},
        3: {'tau': [0.5, 0.8, 1.0], 'k': [1, 2, 5, 10], 'lr': [0.0005, 0.001, 0.005], 'beta': [0.01, 0.05, 0.1]},
        5: {'tau': [0.5, 0.8, 1.0], 'k': [1, 2, 5, 10], 'lr': [0.0005, 0.001, 0.005], 'beta': [0.01, 0.05, 0.1]}
    },
    'pubmed': {
        # Pubmed 依然要小心 OOM，所以 K 最大给到 10
        1: {'tau': [0.5, 0.8, 1.0], 'k': [1, 2, 5, 10], 'lr': [0.005, 0.01, 0.05], 'beta': [0.0001, 0.0005, 0.001]},
        3: {'tau': [0.5, 0.8, 1.0], 'k': [1, 2, 5, 10], 'lr': [0.005, 0.01, 0.05], 'beta': [0.01, 0.05, 0.1]},
        5: {'tau': [0.5, 0.8, 1.0], 'k': [1, 2, 5, 10], 'lr': [0.005, 0.01, 0.05], 'beta': [0.01, 0.05, 0.1]}
    },
    'citeseer': {
        1: {'tau': [0.5, 0.8, 1.0], 'k': [30, 50, 70], 'lr': [0.0005, 0.001, 0.005], 'beta': [0.0001, 0.0005, 0.001]},
        3: {'tau': [0.5, 0.8, 1.0], 'k': [30, 50, 70], 'lr': [0.0005, 0.001, 0.005], 'beta': [0.001, 0.005, 0.01]},
        5: {'tau': [0.5, 0.8, 1.0], 'k': [30, 50, 70], 'lr': [0.0005, 0.001, 0.005], 'beta': [0.001, 0.005, 0.01]}
    }
}

RESULT_DB = "v1_tuning_results_db.json"
LOG_DIR = "logs_v1_coarse"
MD_FILE = "SA_OT_v1_Leaderboard.md"

best_results = {}
if os.path.exists(RESULT_DB):
    try:
        with open(RESULT_DB, 'r') as f:
            best_results = json.load(f)
    except: pass

def parse_accuracy(log_content):
    lines = log_content.strip().split('\n')[-15:]
    for line in lines:
        match = re.search(r'Accuracy:\s*([0-9.]+)\s*±\s*([0-9.]+)', line)
        if match:
            return float(match.group(1)), float(match.group(2))
    return None, None

# ==========================================
# 🧠 核心：无差别显存碰撞引擎 (引入动态限流)
# ==========================================
task_queue = queue.Queue()
active_counts = {g: 0 for g in GPUS}
exclusive_mode = {g: False for g in GPUS}  
gpu_full_flag = {g: False for g in GPUS}  # 🔴 新增：显卡吃撑限流阀门
active_lock = threading.Lock()
print_lock = threading.Lock()

completed_tasks = 0
total_tasks = 0

def worker(gpu_id):
    global completed_tasks
    while True:
        # 🔴 [核心改动 1]：限流等位机制。如果当前卡吃撑了，且还有任务在跑，就耐心等位。
        while True:
            with active_lock:
                is_full = gpu_full_flag[gpu_id]
                active = active_counts[gpu_id]
            if is_full and active > 0:
                time.sleep(5) # 耐心等里面的任务跑完腾出空间
            else:
                break

        try:
            task_item = task_queue.get(timeout=3)
        except queue.Empty:
            break

        # 🔴 [核心改动 2]：不再直接包场，而是记录失败次数
        task_tuple, fail_count = task_item
        req_empty = (fail_count >= 3) # 事不过三，连续 OOM 3 次才允许包场独占
        ds, shot, lr, wd, tau, k, beta, prompts = task_tuple

        os.makedirs(LOG_DIR, exist_ok=True)
        log_file = f"{LOG_DIR}/{ds}_{shot}s_lr{lr}_tau{tau}_k{k}_beta{beta}.txt"

        # 命中缓存，直接跳过
        if not req_empty and os.path.exists(log_file):
            with open(log_file, "r", encoding="utf-8", errors='ignore') as f:
                acc, std = parse_accuracy(f.read())
                if acc is not None:
                    with print_lock:
                        completed_tasks += 1
                    task_queue.task_done()
                    continue

        # ---------------- 🛡️ 智能准入与错峰调度系统 ----------------
        can_run = False
        with active_lock:
            if exclusive_mode[gpu_id] or active_counts[gpu_id] >= MAX_CONCURRENT_PER_GPU:
                can_run = False
            elif req_empty:
                if active_counts[gpu_id] > 0:
                    can_run = False 
                else:
                    can_run = True
                    active_counts[gpu_id] += 1
                    exclusive_mode[gpu_id] = True 
            else:
                can_run = True
                active_counts[gpu_id] += 1

        if not can_run:
            task_queue.put(task_item)
            time.sleep(random.uniform(2, 5)) 
            task_queue.task_done()
            continue

        # 🚀 [终极防抖] 错峰发车，防止 data.to(device) 瞬间爆炸
        if not req_empty:
            wait_time = active_counts[gpu_id] * 4 
            time.sleep(wait_time) 

        # ---------------- 🛡️ 严格锁死运行命令 ----------------
        cmd = [
            "python", "-u", "main.py",
            "--dataset", ds, "--method", "sa_ot_prompt", "--model", PRETRAIN,
            "--shot", str(shot), "--down_lr", str(lr), 
            "--clf_lr", "0.05",         # 🔴 分类器学习率锁死
            "--down_wd", str(wd), 
            "--tau", str(tau), "--k", str(k), "--ot_beta", str(beta), 
            "--num_prompts", str(prompts),
            "--epochs", "2000",         
            "--down_epochs", "2000",    # 💣 彻底放开上限
            "--patience", "100",        # 🟡 耐心加倍
            "--trails", str(TRAILS), 
            "--hid_dim", "256"          # 🟢 维度焊死
        ]
        
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_id
        env["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

        try:
            # 正常启动子进程
            process = subprocess.run(cmd, env=env, capture_output=True, text=True)
            output = process.stdout + process.stderr
            
            with open(log_file, "w") as f:
                f.write(output)

            # 🧠 暴力小写化匹配
            output_lower = output.lower()

            # 🧠 增强型 OOM 探测
            is_oom = (
                "out of memory" in output_lower or 
                "oom" in output_lower or 
                "allocation on device" in output_lower or
                process.returncode == 137 
            )
            
            # 🔴 [核心改动 3]：精准的状态标记
            with active_lock:
                active_counts[gpu_id] -= 1
                if req_empty:
                    exclusive_mode[gpu_id] = False 

                if is_oom:
                    # 如果包场还炸，或者唯一在跑也炸，说明物理显存装不下
                    if req_empty or active_counts[gpu_id] == 0:
                        fatal_oom = True
                    else:
                        fatal_oom = False
                        gpu_full_flag[gpu_id] = True # 📢 广播限流：这张卡吃撑了，门外排队！
                else:
                    fatal_oom = False
                    if process.returncode == 0:
                        gpu_full_flag[gpu_id] = False # 📢 广播放行：跑成功腾出空间了，可以进新任务了！

            if is_oom:
                if fatal_oom:
                    with print_lock:
                        completed_tasks += 1
                        print(f"{completed_tasks}/{total_tasks} 💀 [FATAL OOM] {ds} {shot}s (Exceeds Physical GPU Memory!)")
                    task_queue.task_done()
                else:
                    with print_lock:
                        print(f"⏳ [GPU FULL] {ds} {shot}s - Squeezed out. Retrying normally later (Fail count: {fail_count+1})")
                    # 作为普通任务重新排队，失败次数 +1
                    task_queue.put((task_tuple, fail_count + 1))
                    task_queue.task_done()
                continue

            # 检查是否正常退出
            if process.returncode != 0:
                acc, std = None, None
            else:
                acc, std = parse_accuracy(output)
            
        except Exception as e:
            with active_lock:
                active_counts[gpu_id] -= 1
                if req_empty:
                    exclusive_mode[gpu_id] = False
            with open(log_file, "a") as f:
                f.write(f"\n[CRASH ERROR]: {str(e)}\n")
            acc, std = None, None

        # ---------------- 📊 成绩写入 ----------------
        with print_lock:
            completed_tasks += 1
            if acc is not None:
                key = f"{ds},{shot}"
                if key not in best_results or acc > best_results[key][0]:
                    print(f"🔥 NEW V1 SOTA for {ds.upper()} {shot}-shot! Acc: {acc:.4f} (lr={lr}, tau={tau}, k={k}, beta={beta})")
                    best_results[key] = (acc, std, lr, wd, tau, k, beta, prompts)
                    
                    with open(MD_FILE, "w", encoding="utf-8") as f:
                        f.write("# 🛡️ GraphMAE (v1) Absolute Baseline\n\n")
                        f.write("| Dataset | Shot | Best Acc | Std | LR | WD | Tau | k | Beta | Prompts |\n")
                        f.write("|---|---|---|---|---|---|---|---|---|---|\n")
                        sorted_keys = sorted(best_results.keys(), key=lambda x: (x.split(',')[0], int(x.split(',')[1])))
                        for k_md in sorted_keys:
                            d_md, s_md = k_md.split(',')
                            a_md, st_md, l_md, w_md, t_md, k_val, b_md, p_md = best_results[k_md]
                            f.write(f"| **{d_md.capitalize()}** | {s_md} | **{a_md:.4f}** | ±{st_md:.4f} | {l_md} | {w_md} | {t_md} | {k_val} | {b_md} | {p_md} |\n")
                    with open(RESULT_DB, 'w') as f:
                        json.dump(best_results, f, indent=4)
                        
                elif completed_tasks % 10 == 0:
                    print(f"{completed_tasks}/{total_tasks} explored...")
            else:
                print(f"{completed_tasks}/{total_tasks} [ERROR/CRASH] {ds} {shot}s (Check log)")

        task_queue.task_done()

def main():
    global total_tasks
    print("==========================================================")
    print("🛡️ GraphMAE (v1) AUTO-VRAM DETECT ENGINE (DYNAMIC BACKPRESSURE)")
    print(f"GPUs: {GPUS} | Workers/GPU: {MAX_CONCURRENT_PER_GPU}")
    print("==========================================================\n")

    all_tasks = []
    for ds in SPECIFIC_GRIDS.keys():
        for shot in [1, 3, 5]:
            if shot not in SPECIFIC_GRIDS[ds]: continue
            grid = SPECIFIC_GRIDS[ds][shot]
            combos = list(itertools.product(
                [ds], [shot], grid['lr'], WDS, grid['tau'], grid['k'], grid['beta'], NUM_PROMPTS
            ))
            all_tasks.extend(combos)
    
    total_tasks = len(all_tasks)
    print(f"🎯 Total Configs to Explore: {total_tasks}")
    
    # 🔀 打乱任务，实现自然插队和缝隙填补
    random.seed(42)
    random.shuffle(all_tasks)
    
    # 🚀 绝对纯净的入队：没有任何经验预判，初始 fail_count 为 0
    for t in all_tasks:
        task_queue.put((t, 0))

    threads = []
    for i in range(len(GPUS) * MAX_CONCURRENT_PER_GPU):
        gpu_id = GPUS[i % len(GPUS)]
        t = threading.Thread(target=worker, args=(gpu_id,))
        
        # 🟢 给线程启动本身加个微小间隔，防止瞬间并发拉起
        time.sleep(1.5) 
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    print(f"\n🎉 V1 TUNING COMPLETED! Check '{MD_FILE}'.")

if __name__ == "__main__":
    main()